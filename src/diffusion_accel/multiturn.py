"""Two-turn block-causal MDLM session with finalized K/V state."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List

from .conditioned import (
    _path_result,
    _run_conditioned_ddpm_cache,
    _sequence_agreement,
    _timed_logits_function,
)
from .mdlm import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    _load_mdlm_model,
    _resolve_device,
    _synchronize,
    block_isolation,
)
from .prefix_drift import (
    _benchmark,
    _cached_suffix_forward,
    _capture_prefix_qkv,
    _extract_logits,
    _materialize_cached_block,
    _rotate_prefix_cache,
)
from .quality import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_REVISION,
    collect_token_windows,
)


def _cache_bytes(model: Any, tokens: int) -> int:
    return (
        2
        * int(model.config.n_blocks)
        * tokens
        * int(model.config.hidden_dim)
        * next(model.parameters()).element_size()
    )


def _generate_block(
    logits_operation: Any,
    initial_ids: Any,
    *,
    mask_token_id: int,
    steps: int,
    device: str,
    seed: int,
) -> tuple[Any, Dict[str, object], List[float]]:
    import torch

    timed_logits, latencies = _timed_logits_function(
        logits_operation,
        device=device,
    )
    torch.manual_seed(seed)
    generated, metadata = _run_conditioned_ddpm_cache(
        timed_logits,
        initial_ids.clone(),
        mask_token_id=mask_token_id,
        steps=steps,
        device=device,
    )
    return generated, metadata, latencies


def _warm_two_turn_shapes(
    *,
    model: Any,
    prefix_ids: Any,
    first_block_ids: Any,
    followup_ids: Any,
    second_block_ids: Any,
    timestep: Any,
    heads: int,
) -> None:
    import torch

    prefix_tokens = prefix_ids.shape[1]
    first_tokens = first_block_ids.shape[1]
    followup_tokens = followup_ids.shape[1]
    first_full = torch.cat([prefix_ids, first_block_ids], dim=1)
    with block_isolation([prefix_tokens]):
        model(input_ids=first_full, timesteps=timestep)

    second_history = torch.cat(
        [prefix_ids, first_block_ids, followup_ids], dim=1
    )
    second_full = torch.cat([second_history, second_block_ids], dim=1)
    with block_isolation(
        [
            prefix_tokens,
            prefix_tokens + first_tokens,
            prefix_tokens + first_tokens + followup_tokens,
        ]
    ):
        model(input_ids=second_full, timesteps=timestep)

    model(input_ids=prefix_ids, timesteps=timestep)
    _, prefix_captures = _capture_prefix_qkv(
        model,
        prefix_ids,
        timestep,
        prefix_tokens=prefix_tokens,
        heads=heads,
    )
    cache = _rotate_prefix_cache(model, prefix_captures)
    _cached_suffix_forward(
        model,
        first_block_ids,
        timestep,
        prefix_tokens=prefix_tokens,
        prefix_kv=cache,
    )
    cache = _materialize_cached_block(
        model,
        first_block_ids,
        timestep,
        prefix_tokens=prefix_tokens,
        prefix_kv=cache,
    )
    cache = _materialize_cached_block(
        model,
        followup_ids,
        timestep,
        prefix_tokens=prefix_tokens + first_tokens,
        prefix_kv=cache,
    )
    _cached_suffix_forward(
        model,
        second_block_ids,
        timestep,
        prefix_tokens=prefix_tokens + first_tokens + followup_tokens,
        prefix_kv=cache,
    )
    _synchronize(prefix_ids.device.type)


def evaluate_two_turn_mdlm_session(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: str = "auto",
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    split: str = "test",
    prefix_tokens: int = 64,
    first_answer_tokens: int = 16,
    followup_tokens: int = 16,
    second_answer_tokens: int = 16,
    samples: int = 3,
    steps: int = 64,
    seed: int = 0,
    local_files_only: bool = False,
) -> Dict[str, object]:
    """Generate two blocks while retaining and finalizing session K/V."""
    lengths = (
        prefix_tokens,
        first_answer_tokens,
        followup_tokens,
        second_answer_tokens,
        samples,
        steps,
    )
    if min(lengths) <= 0:
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
            "block isolation currently requires the macOS attention compatibility path"
        )
    total_tokens = (
        prefix_tokens
        + first_answer_tokens
        + followup_tokens
        + second_answer_tokens
    )
    if total_tokens > int(model.config.model_length):
        raise ValueError("two-turn session exceeds checkpoint context length")
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
        window_tokens=total_tokens,
        samples=samples,
        seed=seed,
    )
    mask_token_id = int(model.config.vocab_size) - 1
    heads = int(model.config.n_heads)
    timestep = torch.ones(1, device=resolved_device)

    warm_window = windows[0]
    warm_prefix = torch.tensor(
        warm_window[:prefix_tokens], dtype=torch.long, device=resolved_device
    ).unsqueeze(0)
    warm_first = torch.tensor(
        warm_window[prefix_tokens : prefix_tokens + first_answer_tokens],
        dtype=torch.long,
        device=resolved_device,
    ).unsqueeze(0)
    followup_start = prefix_tokens + first_answer_tokens
    warm_followup = torch.tensor(
        warm_window[followup_start : followup_start + followup_tokens],
        dtype=torch.long,
        device=resolved_device,
    ).unsqueeze(0)
    warm_second = torch.full(
        (1, second_answer_tokens),
        mask_token_id,
        dtype=torch.long,
        device=resolved_device,
    )
    _warm_two_turn_shapes(
        model=model,
        prefix_ids=warm_prefix,
        first_block_ids=warm_first,
        followup_ids=warm_followup,
        second_block_ids=warm_second,
        timestep=timestep,
        heads=heads,
    )

    sample_results = []
    for sample_index, window in enumerate(windows):
        prefix_ids = torch.tensor(
            window[:prefix_tokens], dtype=torch.long, device=resolved_device
        ).unsqueeze(0)
        first_target = torch.tensor(
            window[prefix_tokens : prefix_tokens + first_answer_tokens],
            dtype=torch.long,
            device=resolved_device,
        ).unsqueeze(0)
        followup_start = prefix_tokens + first_answer_tokens
        followup_ids = torch.tensor(
            window[followup_start : followup_start + followup_tokens],
            dtype=torch.long,
            device=resolved_device,
        ).unsqueeze(0)
        second_target = torch.tensor(
            window[
                followup_start
                + followup_tokens : followup_start
                + followup_tokens
                + second_answer_tokens
            ],
            dtype=torch.long,
            device=resolved_device,
        ).unsqueeze(0)
        first_initial = torch.full_like(first_target, mask_token_id)
        second_initial = torch.full_like(second_target, mask_token_id)
        first_seed = seed + 2 * sample_index
        second_seed = first_seed + 1

        def first_recompute_operation(active_ids: Any, sigma: Any) -> Any:
            full_ids = torch.cat([prefix_ids, active_ids], dim=1)
            with block_isolation([prefix_tokens]):
                output = model(input_ids=full_ids, timesteps=sigma)
            return _extract_logits(output)[:, prefix_tokens:]

        (
            first_recomputed,
            first_recompute_metadata,
            first_recompute_latencies,
        ) = _generate_block(
            first_recompute_operation,
            first_initial,
            mask_token_id=mask_token_id,
            steps=steps,
            device=resolved_device,
            seed=first_seed,
        )

        _, initial_prefill_latency, _ = _benchmark(
            lambda: model(input_ids=prefix_ids, timesteps=timestep),
            device=resolved_device,
            repeats=1,
        )
        _, prefix_captures = _capture_prefix_qkv(
            model,
            prefix_ids,
            timestep,
            prefix_tokens=prefix_tokens,
            heads=heads,
        )
        session_kv = _rotate_prefix_cache(model, prefix_captures)

        def first_cached_operation(active_ids: Any, sigma: Any) -> Any:
            return _cached_suffix_forward(
                model,
                active_ids,
                sigma,
                prefix_tokens=prefix_tokens,
                prefix_kv=session_kv,
            )

        first_cached, first_cached_metadata, first_cached_latencies = (
            _generate_block(
                first_cached_operation,
                first_initial,
                mask_token_id=mask_token_id,
                steps=steps,
                device=resolved_device,
                seed=first_seed,
            )
        )

        finalized_output, first_finalization_latency, _ = _benchmark(
            lambda: _materialize_cached_block(
                model,
                first_cached,
                timestep,
                prefix_tokens=prefix_tokens,
                prefix_kv=session_kv,
            ),
            device=resolved_device,
            repeats=1,
        )
        session_kv = finalized_output
        history_after_first = prefix_tokens + first_answer_tokens
        followup_output, followup_prefill_latency, _ = _benchmark(
            lambda: _materialize_cached_block(
                model,
                followup_ids,
                timestep,
                prefix_tokens=history_after_first,
                prefix_kv=session_kv,
            ),
            device=resolved_device,
            repeats=1,
        )
        session_kv = followup_output
        history_before_second = history_after_first + followup_tokens

        full_history = torch.cat(
            [prefix_ids, first_recomputed, followup_ids], dim=1
        )
        completed_boundaries = [
            prefix_tokens,
            history_after_first,
            history_before_second,
        ]

        def second_recompute_operation(active_ids: Any, sigma: Any) -> Any:
            full_ids = torch.cat([full_history, active_ids], dim=1)
            with block_isolation(completed_boundaries):
                output = model(input_ids=full_ids, timesteps=sigma)
            return _extract_logits(output)[:, history_before_second:]

        (
            second_recomputed,
            second_recompute_metadata,
            second_recompute_latencies,
        ) = _generate_block(
            second_recompute_operation,
            second_initial,
            mask_token_id=mask_token_id,
            steps=steps,
            device=resolved_device,
            seed=second_seed,
        )

        def second_cached_operation(active_ids: Any, sigma: Any) -> Any:
            return _cached_suffix_forward(
                model,
                active_ids,
                sigma,
                prefix_tokens=history_before_second,
                prefix_kv=session_kv,
            )

        second_cached, second_cached_metadata, second_cached_latencies = (
            _generate_block(
                second_cached_operation,
                second_initial,
                mask_token_id=mask_token_id,
                steps=steps,
                device=resolved_device,
                seed=second_seed,
            )
        )
        final_cache_output, second_finalization_latency, _ = _benchmark(
            lambda: _materialize_cached_block(
                model,
                second_cached,
                timestep,
                prefix_tokens=history_before_second,
                prefix_kv=session_kv,
            ),
            device=resolved_device,
            repeats=1,
        )
        session_kv = final_cache_output

        first_recompute_result = _path_result(
            first_recomputed,
            first_target,
            tokenizer,
            first_recompute_metadata,
            first_recompute_latencies,
        )
        first_cached_result = _path_result(
            first_cached,
            first_target,
            tokenizer,
            first_cached_metadata,
            first_cached_latencies,
        )
        second_recompute_result = _path_result(
            second_recomputed,
            second_target,
            tokenizer,
            second_recompute_metadata,
            second_recompute_latencies,
        )
        second_cached_result = _path_result(
            second_cached,
            second_target,
            tokenizer,
            second_cached_metadata,
            second_cached_latencies,
        )

        final_tokens = history_before_second + second_answer_tokens
        sample_results.append(
            {
                "sample": sample_index,
                "first_seed": first_seed,
                "second_seed": second_seed,
                "initial_prefix_text": tokenizer.decode(
                    prefix_ids[0].detach().cpu().tolist()
                ),
                "first_target_text": tokenizer.decode(
                    first_target[0].detach().cpu().tolist()
                ),
                "followup_text": tokenizer.decode(
                    followup_ids[0].detach().cpu().tolist()
                ),
                "second_target_text": tokenizer.decode(
                    second_target[0].detach().cpu().tolist()
                ),
                "turn_1": {
                    "recompute": first_recompute_result,
                    "cached": first_cached_result,
                    "exact_sequence": bool(
                        first_recomputed.eq(first_cached).all().item()
                    ),
                    "token_agreement": _sequence_agreement(
                        first_recomputed, first_cached
                    ),
                    "transition_schedule_equal": bool(
                        first_recompute_metadata["transition_active_tokens"]
                        == first_cached_metadata["transition_active_tokens"]
                        and first_recompute_metadata["transition_changed_tokens"]
                        == first_cached_metadata["transition_changed_tokens"]
                    ),
                    "initial_prefix_prefill_latency_ms": (
                        initial_prefill_latency * 1e3
                    ),
                    "answer_finalization_latency_ms": (
                        first_finalization_latency * 1e3
                    ),
                    "session_kv_mib_after_answer": (
                        _cache_bytes(model, history_after_first) / 1024**2
                    ),
                },
                "followup_prefill": {
                    "tokens": followup_tokens,
                    "latency_ms": followup_prefill_latency * 1e3,
                    "session_kv_mib_before_second_answer": (
                        _cache_bytes(model, history_before_second) / 1024**2
                    ),
                },
                "turn_2": {
                    "recompute": second_recompute_result,
                    "cached": second_cached_result,
                    "exact_sequence": bool(
                        second_recomputed.eq(second_cached).all().item()
                    ),
                    "token_agreement": _sequence_agreement(
                        second_recomputed, second_cached
                    ),
                    "transition_schedule_equal": bool(
                        second_recompute_metadata["transition_active_tokens"]
                        == second_cached_metadata["transition_active_tokens"]
                        and second_recompute_metadata["transition_changed_tokens"]
                        == second_cached_metadata["transition_changed_tokens"]
                    ),
                    "answer_finalization_latency_ms": (
                        second_finalization_latency * 1e3
                    ),
                    "final_session_kv_mib": (
                        _cache_bytes(model, final_tokens) / 1024**2
                    ),
                    "final_cache_tokens": session_kv[0].shape[1],
                },
            }
        )

    turn_one_exact = sum(
        bool(result["turn_1"]["exact_sequence"]) for result in sample_results
    )
    turn_two_exact = sum(
        bool(result["turn_2"]["exact_sequence"]) for result in sample_results
    )
    second_recompute_forward = [
        float(result["turn_2"]["recompute"]["model_forward_latency_ms"])
        for result in sample_results
    ]
    second_cached_forward = [
        float(result["turn_2"]["cached"]["model_forward_latency_ms"])
        for result in sample_results
    ]
    median_recompute = statistics.median(second_recompute_forward)
    median_cached = statistics.median(second_cached_forward)
    cached_followup_request = [
        float(result["followup_prefill"]["latency_ms"])
        + float(result["turn_2"]["cached"]["model_forward_latency_ms"])
        for result in sample_results
    ]
    cached_cross_request_cost = [
        float(result["turn_1"]["answer_finalization_latency_ms"])
        + followup_request
        for result, followup_request in zip(
            sample_results,
            cached_followup_request,
        )
    ]
    cached_ready_for_third_turn = [
        cross_request
        + float(result["turn_2"]["answer_finalization_latency_ms"])
        for result, cross_request in zip(
            sample_results,
            cached_cross_request_cost,
        )
    ]
    median_followup_request = statistics.median(cached_followup_request)
    median_cross_request = statistics.median(cached_cross_request_cost)
    median_ready_for_third = statistics.median(cached_ready_for_third_turn)
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
        "session_shape": {
            "initial_prefix_tokens": prefix_tokens,
            "first_answer_tokens": first_answer_tokens,
            "followup_tokens": followup_tokens,
            "second_answer_tokens": second_answer_tokens,
            "final_session_tokens": total_tokens,
        },
        "steps_per_answer": steps,
        "samples": samples,
        "aggregate": {
            "exact_turn_1_sequences": turn_one_exact,
            "exact_turn_2_sequences": turn_two_exact,
            "identical_turn_1_transition_schedules": sum(
                bool(result["turn_1"]["transition_schedule_equal"])
                for result in sample_results
            ),
            "identical_turn_2_transition_schedules": sum(
                bool(result["turn_2"]["transition_schedule_equal"])
                for result in sample_results
            ),
            "minimum_turn_2_token_agreement": min(
                float(result["turn_2"]["token_agreement"])
                for result in sample_results
            ),
            "median_turn_2_recompute_model_forward_latency_ms": median_recompute,
            "median_turn_2_cached_model_forward_latency_ms": median_cached,
            "median_turn_2_model_forward_latency_reduction_fraction": (
                (median_recompute - median_cached) / median_recompute
            ),
            "median_cached_followup_request_latency_ms": median_followup_request,
            "median_cached_followup_request_reduction_fraction": (
                (median_recompute - median_followup_request) / median_recompute
            ),
            "median_cross_request_cost_including_first_finalization_ms": (
                median_cross_request
            ),
            "median_cross_request_reduction_including_first_finalization_fraction": (
                (median_recompute - median_cross_request) / median_recompute
            ),
            "median_cached_cost_ready_for_third_turn_ms": median_ready_for_third,
            "median_ready_for_third_turn_reduction_fraction": (
                (median_recompute - median_ready_for_third) / median_recompute
            ),
            "median_first_answer_finalization_latency_ms": statistics.median(
                float(result["turn_1"]["answer_finalization_latency_ms"])
                for result in sample_results
            ),
            "median_followup_prefill_latency_ms": statistics.median(
                float(result["followup_prefill"]["latency_ms"])
                for result in sample_results
            ),
            "median_second_answer_finalization_latency_ms": statistics.median(
                float(result["turn_2"]["answer_finalization_latency_ms"])
                for result in sample_results
            ),
            "final_session_kv_mib": _cache_bytes(model, total_tokens) / 1024**2,
        },
        "sample_results": sample_results,
    }
