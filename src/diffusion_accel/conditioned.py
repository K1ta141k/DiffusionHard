"""Conditioned DDPM generation with reusable prefix K/V."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Callable, Dict, List

from .mdlm import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    _load_mdlm_model,
    _mdlm_log_probabilities,
    _resolve_device,
    _sample_categorical,
    _synchronize,
    prefix_isolation,
)
from .prefix_drift import (
    _benchmark,
    _cached_suffix_forward,
    _capture_prefix_qkv,
    _extract_logits,
    _rotate_prefix_cache,
)
from .quality import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_REVISION,
    collect_token_windows,
)


def _run_conditioned_ddpm_cache(
    logits_function: Callable[[Any, Any], Any],
    suffix_ids: Any,
    *,
    mask_token_id: int,
    steps: int,
    device: str,
    probability_cache_enabled: bool = True,
    sampling_epsilon: float = 1e-5,
) -> tuple[Any, Dict[str, object]]:
    """Run ancestral MDLM DDPM updates over only the generated suffix."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    import torch

    noise_epsilon = 1e-3
    timesteps = torch.linspace(1.0, sampling_epsilon, steps + 1, device=device)
    probability_cache = None
    probability_cache_hits = 0
    model_evaluations = 0
    last_model_input = None
    changed_tokens: List[int] = []
    active_tokens: List[int] = []

    _synchronize(device)
    started = time.perf_counter()
    for transition_id in range(steps):
        active_count = int(suffix_ids.eq(mask_token_id).sum().item())
        if active_count == 0:
            break
        active_tokens.append(active_count)
        time_t = timesteps[transition_id].reshape(1)
        time_s = timesteps[transition_id + 1].reshape(1)
        sigma_t = -torch.log1p(-(1 - noise_epsilon) * time_t)

        if probability_cache is None:
            last_model_input = suffix_ids.detach().clone()
            logits = logits_function(suffix_ids, sigma_t)
            probability_cache = _mdlm_log_probabilities(
                logits,
                mask_token_id,
            ).exp()
            model_evaluations += 1
        else:
            probability_cache_hits += 1

        move_t = ((1 - noise_epsilon) * time_t)[:, None, None]
        move_s = ((1 - noise_epsilon) * time_s)[:, None, None]
        transition = probability_cache * (move_t - move_s)
        transition[:, :, mask_token_id] = move_s[:, :, 0]
        proposal = _sample_categorical(transition)
        next_ids = torch.where(
            suffix_ids.eq(mask_token_id),
            proposal,
            suffix_ids,
        )
        changed_count = int(suffix_ids.ne(next_ids).sum().item())
        changed_tokens.append(changed_count)
        if changed_count or not probability_cache_enabled:
            probability_cache = None
        suffix_ids = next_ids

    remaining_mask = suffix_ids.eq(mask_token_id)
    remaining_count = int(remaining_mask.sum().item())
    if remaining_count:
        sigma = -torch.log1p(
            -(1 - noise_epsilon)
            * torch.tensor([sampling_epsilon], device=device)
        )
        last_model_input = suffix_ids.detach().clone()
        logits = logits_function(suffix_ids, sigma)
        predicted = _mdlm_log_probabilities(
            logits,
            mask_token_id,
        ).argmax(dim=-1)
        suffix_ids = torch.where(remaining_mask, predicted, suffix_ids)
        model_evaluations += 1

    _synchronize(device)
    wall_time = time.perf_counter() - started
    return suffix_ids, {
        "sampling_transitions": len(active_tokens),
        "model_evaluations": model_evaluations,
        "probability_cache_hits": probability_cache_hits,
        "transition_active_tokens": active_tokens,
        "transition_changed_tokens": changed_tokens,
        "wall_latency_ms": wall_time * 1e3,
        "all_tokens_committed": not bool(
            suffix_ids.eq(mask_token_id).any().item()
        ),
        "terminal_kv_valid": bool(
            last_model_input is not None
            and last_model_input.eq(suffix_ids).all().item()
        ),
    }


def _run_conditioned_ddpm_candidate_cache(
    logits_function: Callable[[Any, Any], Any],
    suffix_ids: Any,
    *,
    mask_token_id: int,
    steps: int,
    device: str,
    sampling_epsilon: float = 1e-5,
) -> tuple[Any, Dict[str, object]]:
    """Use one categorical candidate plus scalar reveal draws per position.

    Conditional on a masked position becoming unmasked, its token distribution
    is the unchanged model distribution. A candidate may therefore be sampled
    once and retained across transitions until any input token changes.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    import torch

    noise_epsilon = 1e-3
    timesteps = torch.linspace(1.0, sampling_epsilon, steps + 1, device=device)
    candidate_cache = None
    candidate_cache_hits = 0
    model_evaluations = 0
    last_model_input = None
    changed_tokens: List[int] = []
    active_tokens: List[int] = []

    _synchronize(device)
    started = time.perf_counter()
    for transition_id in range(steps):
        active_mask = suffix_ids.eq(mask_token_id)
        active_count = int(active_mask.sum().item())
        if active_count == 0:
            break
        active_tokens.append(active_count)
        time_t = timesteps[transition_id].reshape(1)
        time_s = timesteps[transition_id + 1].reshape(1)
        sigma_t = -torch.log1p(-(1 - noise_epsilon) * time_t)

        if candidate_cache is None:
            last_model_input = suffix_ids.detach().clone()
            logits = logits_function(suffix_ids, sigma_t)
            probabilities = _mdlm_log_probabilities(
                logits,
                mask_token_id,
            ).exp()
            candidate_cache = _sample_categorical(probabilities)
            model_evaluations += 1
        else:
            candidate_cache_hits += 1

        move_t = ((1 - noise_epsilon) * time_t).reshape(1, 1)
        move_s = ((1 - noise_epsilon) * time_s).reshape(1, 1)
        reveal_probability = (move_t - move_s) / move_t
        reveal = torch.rand(suffix_ids.shape, device=device) < reveal_probability
        proposal = torch.where(
            reveal,
            candidate_cache,
            torch.full_like(candidate_cache, mask_token_id),
        )
        next_ids = torch.where(active_mask, proposal, suffix_ids)
        changed_count = int(suffix_ids.ne(next_ids).sum().item())
        changed_tokens.append(changed_count)
        if changed_count:
            candidate_cache = None
        suffix_ids = next_ids

    remaining_mask = suffix_ids.eq(mask_token_id)
    remaining_count = int(remaining_mask.sum().item())
    if remaining_count:
        sigma = -torch.log1p(
            -(1 - noise_epsilon)
            * torch.tensor([sampling_epsilon], device=device)
        )
        last_model_input = suffix_ids.detach().clone()
        logits = logits_function(suffix_ids, sigma)
        predicted = _mdlm_log_probabilities(
            logits,
            mask_token_id,
        ).argmax(dim=-1)
        suffix_ids = torch.where(remaining_mask, predicted, suffix_ids)
        model_evaluations += 1

    _synchronize(device)
    wall_time = time.perf_counter() - started
    positions = suffix_ids.numel()
    candidate_id_bytes = max(1, math.ceil(mask_token_id.bit_length() / 8))
    hardware_candidate_bytes = (
        positions * candidate_id_bytes + 2 * math.ceil(positions / 8)
    )
    return suffix_ids, {
        "sampling_transitions": len(active_tokens),
        "model_evaluations": model_evaluations,
        "candidate_cache_hits": candidate_cache_hits,
        "probability_cache_hits": candidate_cache_hits,
        "transition_active_tokens": active_tokens,
        "transition_changed_tokens": changed_tokens,
        "wall_latency_ms": wall_time * 1e3,
        "all_tokens_committed": not bool(
            suffix_ids.eq(mask_token_id).any().item()
        ),
        "terminal_kv_valid": bool(
            last_model_input is not None
            and last_model_input.eq(suffix_ids).all().item()
        ),
        "hardware_candidate_cache_bytes": hardware_candidate_bytes,
        "candidate_id_bytes": candidate_id_bytes,
        "candidate_token_bytes": positions * candidate_id_bytes,
        "active_and_valid_bitmap_bytes": 2 * math.ceil(positions / 8),
    }


def _timed_logits_function(
    operation: Callable[[Any, Any], Any],
    *,
    device: str,
) -> tuple[Callable[[Any, Any], Any], List[float]]:
    latencies: List[float] = []

    def timed(suffix_ids: Any, sigma: Any) -> Any:
        _synchronize(device)
        started = time.perf_counter()
        logits = operation(suffix_ids, sigma)
        _synchronize(device)
        latencies.append(time.perf_counter() - started)
        return logits

    return timed, latencies


def _sequence_agreement(left: Any, right: Any) -> float:
    return float(left.eq(right).float().mean().item())


def _target_agreement(generated: Any, target: Any) -> float:
    return _sequence_agreement(generated, target)


def _path_result(
    generated: Any,
    target: Any,
    tokenizer: Any,
    metadata: Dict[str, object],
    forward_latencies: List[float],
) -> Dict[str, object]:
    total_forward = sum(forward_latencies)
    return {
        **metadata,
        "model_forward_latency_ms": total_forward * 1e3,
        "median_model_evaluation_latency_ms": (
            statistics.median(forward_latencies) * 1e3
            if forward_latencies
            else 0.0
        ),
        "target_token_agreement": _target_agreement(generated, target),
        "generated_token_ids": generated[0].detach().cpu().tolist(),
        "generated_text": tokenizer.decode(
            generated[0].detach().cpu().tolist()
        ),
    }


def evaluate_conditioned_mdlm_ddpm(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: str = "auto",
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    split: str = "test",
    prefix_tokens: int = 64,
    suffix_tokens: int = 32,
    samples: int = 3,
    steps: int = 64,
    seed: int = 0,
    local_files_only: bool = False,
) -> Dict[str, object]:
    """Compare original, isolated, and cached conditioned DDPM generation."""
    if min(prefix_tokens, suffix_tokens, samples, steps) <= 0:
        raise ValueError("token counts, samples, and steps must be positive")

    import torch
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoTokenizer

    resolved_device = _resolve_device(device)
    torch.manual_seed(seed)
    model = _load_mdlm_model(
        model_id=model_id,
        revision=revision,
        device=resolved_device,
        local_files_only=local_files_only,
    )
    import flash_attn

    if not getattr(flash_attn, "_diffusion_accel_compat", False):
        raise RuntimeError(
            "prefix isolation currently requires the macOS attention compatibility path"
        )
    if prefix_tokens + suffix_tokens > int(model.config.model_length):
        raise ValueError("conditioned sequence exceeds checkpoint context length")
    if bool(model.config.time_conditioning):
        raise ValueError("probability caching requires a non-time-conditioned model")

    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        local_files_only=local_files_only,
    )
    dataset = load_dataset(
        dataset_id,
        dataset_config,
        split=split,
        revision=dataset_revision,
        download_config=DownloadConfig(local_files_only=local_files_only),
    )
    windows = collect_token_windows(
        dataset["text"],
        tokenizer,
        window_tokens=prefix_tokens + suffix_tokens,
        samples=samples,
        seed=seed,
    )
    mask_token_id = int(model.config.vocab_size) - 1
    heads = int(model.config.n_heads)

    first_prefix = torch.tensor(
        windows[0][:prefix_tokens],
        dtype=torch.long,
        device=resolved_device,
    ).unsqueeze(0)
    first_suffix = torch.full(
        (1, suffix_tokens),
        mask_token_id,
        dtype=torch.long,
        device=resolved_device,
    )
    first_sigma = torch.ones(1, device=resolved_device)
    first_full = torch.cat([first_prefix, first_suffix], dim=1)
    model(input_ids=first_full, timesteps=first_sigma)
    with prefix_isolation(prefix_tokens):
        model(input_ids=first_full, timesteps=first_sigma)
    model(input_ids=first_prefix, timesteps=first_sigma)
    _, warm_captures = _capture_prefix_qkv(
        model,
        first_prefix,
        first_sigma,
        prefix_tokens=prefix_tokens,
        heads=heads,
    )
    _cached_suffix_forward(
        model,
        first_suffix,
        first_sigma,
        prefix_tokens=prefix_tokens,
        prefix_kv=_rotate_prefix_cache(model, warm_captures),
    )
    _synchronize(resolved_device)

    sample_results = []
    for sample_index, window in enumerate(windows):
        prefix_ids = torch.tensor(
            window[:prefix_tokens],
            dtype=torch.long,
            device=resolved_device,
        ).unsqueeze(0)
        target_ids = torch.tensor(
            window[prefix_tokens:],
            dtype=torch.long,
            device=resolved_device,
        ).unsqueeze(0)
        initial_suffix = torch.full_like(target_ids, mask_token_id)
        sample_seed = seed + sample_index

        def original_operation(suffix_ids: Any, sigma: Any) -> Any:
            full_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
            return _extract_logits(
                model(input_ids=full_ids, timesteps=sigma)
            )[:, prefix_tokens:]

        original_logits, original_latencies = _timed_logits_function(
            original_operation,
            device=resolved_device,
        )
        torch.manual_seed(sample_seed)
        original_generated, original_metadata = _run_conditioned_ddpm_cache(
            original_logits,
            initial_suffix.clone(),
            mask_token_id=mask_token_id,
            steps=steps,
            device=resolved_device,
        )

        def isolated_operation(suffix_ids: Any, sigma: Any) -> Any:
            full_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
            with prefix_isolation(prefix_tokens):
                output = model(input_ids=full_ids, timesteps=sigma)
            return _extract_logits(output)[:, prefix_tokens:]

        isolated_logits, isolated_latencies = _timed_logits_function(
            isolated_operation,
            device=resolved_device,
        )
        torch.manual_seed(sample_seed)
        isolated_generated, isolated_metadata = _run_conditioned_ddpm_cache(
            isolated_logits,
            initial_suffix.clone(),
            mask_token_id=mask_token_id,
            steps=steps,
            device=resolved_device,
        )

        _, prefill_latency, _ = _benchmark(
            lambda: model(input_ids=prefix_ids, timesteps=first_sigma),
            device=resolved_device,
            repeats=1,
        )
        _, prefix_captures = _capture_prefix_qkv(
            model,
            prefix_ids,
            first_sigma,
            prefix_tokens=prefix_tokens,
            heads=heads,
        )
        rotated_prefix_cache = _rotate_prefix_cache(model, prefix_captures)

        def cached_operation(suffix_ids: Any, sigma: Any) -> Any:
            return _cached_suffix_forward(
                model,
                suffix_ids,
                sigma,
                prefix_tokens=prefix_tokens,
                prefix_kv=rotated_prefix_cache,
            )

        cached_logits, cached_latencies = _timed_logits_function(
            cached_operation,
            device=resolved_device,
        )
        torch.manual_seed(sample_seed)
        cached_generated, cached_metadata = _run_conditioned_ddpm_cache(
            cached_logits,
            initial_suffix.clone(),
            mask_token_id=mask_token_id,
            steps=steps,
            device=resolved_device,
        )

        original_result = _path_result(
            original_generated,
            target_ids,
            tokenizer,
            original_metadata,
            original_latencies,
        )
        isolated_result = _path_result(
            isolated_generated,
            target_ids,
            tokenizer,
            isolated_metadata,
            isolated_latencies,
        )
        cached_result = _path_result(
            cached_generated,
            target_ids,
            tokenizer,
            cached_metadata,
            cached_latencies,
        )
        cached_result["one_time_prefix_prefill_latency_ms"] = (
            prefill_latency * 1e3
        )
        sample_results.append(
            {
                "sample": sample_index,
                "seed": sample_seed,
                "prefix_text": tokenizer.decode(
                    prefix_ids[0].detach().cpu().tolist()
                ),
                "target_suffix_text": tokenizer.decode(
                    target_ids[0].detach().cpu().tolist()
                ),
                "original_full_attention": original_result,
                "prefix_isolated_recompute": isolated_result,
                "prefix_isolated_cache": cached_result,
                "cached_vs_recompute_token_agreement": _sequence_agreement(
                    cached_generated,
                    isolated_generated,
                ),
                "cached_vs_recompute_exact_sequence": bool(
                    cached_generated.eq(isolated_generated).all().item()
                ),
                "cached_vs_recompute_transition_schedule_equal": bool(
                    cached_metadata["transition_active_tokens"]
                    == isolated_metadata["transition_active_tokens"]
                    and cached_metadata["transition_changed_tokens"]
                    == isolated_metadata["transition_changed_tokens"]
                ),
                "original_vs_isolated_token_agreement": _sequence_agreement(
                    original_generated,
                    isolated_generated,
                ),
            }
        )

    cache_agreements = [
        float(result["cached_vs_recompute_token_agreement"])
        for result in sample_results
    ]
    isolated_forward_latencies = [
        float(result["prefix_isolated_recompute"]["model_forward_latency_ms"])
        for result in sample_results
    ]
    cached_forward_latencies = [
        float(result["prefix_isolated_cache"]["model_forward_latency_ms"])
        for result in sample_results
    ]
    isolated_wall_latencies = [
        float(result["prefix_isolated_recompute"]["wall_latency_ms"])
        for result in sample_results
    ]
    cached_wall_latencies = [
        float(result["prefix_isolated_cache"]["wall_latency_ms"])
        for result in sample_results
    ]
    prefix_prefill_latencies = [
        float(
            result["prefix_isolated_cache"][
                "one_time_prefix_prefill_latency_ms"
            ]
        )
        for result in sample_results
    ]
    median_isolated = statistics.median(isolated_forward_latencies)
    median_cached = statistics.median(cached_forward_latencies)
    median_isolated_wall = statistics.median(isolated_wall_latencies)
    median_cached_wall = statistics.median(cached_wall_latencies)
    element_bytes = next(model.parameters()).element_size()
    prefix_cache_bytes = (
        2
        * int(model.config.n_blocks)
        * prefix_tokens
        * int(model.config.hidden_dim)
        * element_bytes
    )
    return {
        "model_id": model_id,
        "revision": revision,
        "device": resolved_device,
        "dataset": {
            "id": dataset_id,
            "config": dataset_config,
            "revision": dataset_revision,
            "split": split,
        },
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "steps": steps,
        "samples": samples,
        "sampler": "ancestral-ddpm-with-exact-no-change-probability-cache",
        "aggregate": {
            "exact_cached_sequences": sum(
                bool(result["cached_vs_recompute_exact_sequence"])
                for result in sample_results
            ),
            "identical_cached_transition_schedules": sum(
                bool(result["cached_vs_recompute_transition_schedule_equal"])
                for result in sample_results
            ),
            "minimum_cached_token_agreement": min(cache_agreements),
            "median_isolated_model_forward_latency_ms": median_isolated,
            "median_cached_model_forward_latency_ms": median_cached,
            "median_model_forward_latency_reduction_fraction": (
                (median_isolated - median_cached) / median_isolated
            ),
            "median_isolated_wall_latency_ms": median_isolated_wall,
            "median_cached_wall_latency_ms": median_cached_wall,
            "median_wall_latency_reduction_fraction": (
                (median_isolated_wall - median_cached_wall)
                / median_isolated_wall
            ),
            "median_one_time_prefix_prefill_latency_ms": statistics.median(
                prefix_prefill_latencies
            ),
            "median_isolated_model_evaluations": statistics.median(
                int(result["prefix_isolated_recompute"]["model_evaluations"])
                for result in sample_results
            ),
            "median_probability_cache_hits": statistics.median(
                int(result["prefix_isolated_cache"]["probability_cache_hits"])
                for result in sample_results
            ),
            "terminal_kv_valid_sequences": sum(
                bool(result["prefix_isolated_cache"]["terminal_kv_valid"])
                for result in sample_results
            ),
            "mean_original_vs_isolated_token_agreement": statistics.mean(
                float(result["original_vs_isolated_token_agreement"])
                for result in sample_results
            ),
            "mean_original_target_token_agreement": statistics.mean(
                float(result["original_full_attention"]["target_token_agreement"])
                for result in sample_results
            ),
            "mean_isolated_target_token_agreement": statistics.mean(
                float(
                    result["prefix_isolated_recompute"][
                        "target_token_agreement"
                    ]
                )
                for result in sample_results
            ),
            "prefix_kv_bytes": prefix_cache_bytes,
            "prefix_kv_mib": prefix_cache_bytes / 1024**2,
        },
        "sample_results": sample_results,
    }
