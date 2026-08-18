"""Held-out text quality gate for prefix-isolated MDLM inference."""

from __future__ import annotations

import random
import statistics
from typing import Any, Dict, Iterable, List, Sequence

from .mdlm import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    _load_mdlm_model,
    _resolve_device,
    _synchronize,
    prefix_isolation,
)
from .prefix_drift import (
    _benchmark,
    _cached_suffix_forward,
    _capture_prefix_qkv,
    _extract_logits,
    _rotate_prefix_cache,
    _top1_ids,
    tensor_drift_metrics,
)

DEFAULT_DATASET_ID = "Salesforce/wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def collect_token_windows(
    texts: Iterable[str],
    tokenizer: Any,
    *,
    window_tokens: int,
    samples: int,
    seed: int,
) -> List[List[int]]:
    """Create deterministic held-out windows from a sequence of documents."""
    if window_tokens <= 0 or samples <= 0:
        raise ValueError("window_tokens and samples must be positive")
    token_ids: List[int] = []
    separator = tokenizer.eos_token_id
    for text in texts:
        if not text or not text.strip():
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            continue
        if token_ids and separator is not None:
            token_ids.append(int(separator))
        token_ids.extend(int(token) for token in encoded)
    if len(token_ids) < window_tokens:
        raise ValueError("dataset does not contain one complete token window")

    possible_starts = len(token_ids) - window_tokens + 1
    if possible_starts < samples:
        raise ValueError("dataset is too small for the requested sample count")
    starts = random.Random(seed).sample(range(possible_starts), samples)
    return [token_ids[start : start + window_tokens] for start in starts]


def masked_token_metrics(
    logits: Any,
    targets: Any,
    *,
    mask_token_id: int,
) -> Dict[str, float]:
    """Measure true-token accuracy and negative log likelihood."""
    import torch

    logits = logits.detach().float().clone()
    logits[..., mask_token_id] = -torch.inf
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets).sum()
    log_probabilities = torch.log_softmax(logits, dim=-1)
    target_log_probability = log_probabilities.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    return {
        "correct": float(correct.item()),
        "tokens": float(targets.numel()),
        "negative_log_likelihood_sum": float(
            (-target_log_probability).sum().item()
        ),
    }


def evaluate_quality_gates(
    *,
    original_accuracy: float,
    isolated_accuracy: float,
    original_nll: float,
    isolated_nll: float,
    cached_top1_agreement: float,
    maximum_cached_logit_nrmse: float,
    maximum_accuracy_drop_fraction: float,
    maximum_nll_increase_fraction: float,
    maximum_logit_nrmse: float,
) -> Dict[str, object]:
    accuracy_drop = original_accuracy - isolated_accuracy
    accuracy_drop_fraction = (
        accuracy_drop / original_accuracy
        if original_accuracy > 0
        else float("inf")
    )
    nll_increase_fraction = (
        (isolated_nll - original_nll) / original_nll
        if original_nll > 0
        else float("inf")
    )
    checks = {
        "isolated_accuracy_drop_within_limit": (
            accuracy_drop_fraction <= maximum_accuracy_drop_fraction
        ),
        "isolated_nll_increase_within_limit": (
            nll_increase_fraction <= maximum_nll_increase_fraction
        ),
        "cached_top1_agreement_is_exact": cached_top1_agreement == 1.0,
        "cached_logit_nrmse_within_limit": (
            maximum_cached_logit_nrmse <= maximum_logit_nrmse
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "accuracy_drop": accuracy_drop,
        "accuracy_drop_fraction": accuracy_drop_fraction,
        "maximum_accuracy_drop_fraction": maximum_accuracy_drop_fraction,
        "nll_increase_fraction": nll_increase_fraction,
        "maximum_nll_increase_fraction": maximum_nll_increase_fraction,
        "maximum_cached_logit_nrmse": maximum_cached_logit_nrmse,
        "maximum_logit_nrmse": maximum_logit_nrmse,
    }


def _median_milliseconds(values_s: Sequence[float]) -> float:
    return statistics.median(values_s) * 1e3


def _evaluate_configuration(
    *,
    model: Any,
    windows: Sequence[Sequence[int]],
    prefix_tokens: int,
    suffix_tokens: int,
    device: str,
    tokenizer: Any,
    example_limit: int = 0,
) -> Dict[str, object]:
    import torch

    config = model.config
    mask_token_id = int(config.vocab_size) - 1
    timestep = torch.ones(1, device=device)
    heads = int(config.n_heads)
    original_totals = {"correct": 0.0, "tokens": 0.0, "nll": 0.0}
    isolated_totals = {"correct": 0.0, "tokens": 0.0, "nll": 0.0}
    cache_agreement_correct = 0
    cache_agreement_tokens = 0
    maximum_cached_logit_nrmse = 0.0
    original_latencies: List[float] = []
    isolated_latencies: List[float] = []
    prefix_prefill_latencies: List[float] = []
    cached_suffix_latencies: List[float] = []
    examples = []

    # Warm every shape and attention path before collecting samples.
    first_window = windows[0]
    first_prefix = torch.tensor(
        first_window[:prefix_tokens], dtype=torch.long, device=device
    ).unsqueeze(0)
    first_masked = torch.cat(
        [
            first_prefix,
            torch.full(
                (1, suffix_tokens),
                mask_token_id,
                dtype=torch.long,
                device=device,
            ),
        ],
        dim=1,
    )
    model(input_ids=first_masked, timesteps=timestep)
    with prefix_isolation(prefix_tokens):
        model(input_ids=first_masked, timesteps=timestep)
    model(input_ids=first_prefix, timesteps=timestep)
    _, warm_captures = _capture_prefix_qkv(
        model,
        first_prefix,
        timestep,
        prefix_tokens=prefix_tokens,
        heads=heads,
    )
    _cached_suffix_forward(
        model,
        first_masked[:, prefix_tokens:],
        timestep,
        prefix_tokens=prefix_tokens,
        prefix_kv=_rotate_prefix_cache(model, warm_captures),
    )
    _synchronize(device)

    for sample_index, window in enumerate(windows):
        prefix_ids = torch.tensor(
            window[:prefix_tokens], dtype=torch.long, device=device
        ).unsqueeze(0)
        targets = torch.tensor(
            window[prefix_tokens : prefix_tokens + suffix_tokens],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        masked_ids = torch.cat(
            [
                prefix_ids,
                torch.full_like(targets, mask_token_id),
            ],
            dim=1,
        )

        original_output, original_latency, _ = _benchmark(
            lambda: model(input_ids=masked_ids, timesteps=timestep),
            device=device,
            repeats=1,
        )
        original_logits = _extract_logits(original_output)[:, prefix_tokens:]
        original_latencies.append(original_latency)

        with prefix_isolation(prefix_tokens):
            isolated_output, isolated_latency, _ = _benchmark(
                lambda: model(input_ids=masked_ids, timesteps=timestep),
                device=device,
                repeats=1,
            )
        isolated_logits = _extract_logits(isolated_output)[:, prefix_tokens:]
        isolated_latencies.append(isolated_latency)

        _, prefix_prefill_latency, _ = _benchmark(
            lambda: model(input_ids=prefix_ids, timesteps=timestep),
            device=device,
            repeats=1,
        )
        prefix_prefill_latencies.append(prefix_prefill_latency)
        _, prefix_captures = _capture_prefix_qkv(
            model,
            prefix_ids,
            timestep,
            prefix_tokens=prefix_tokens,
            heads=heads,
        )
        rotated_prefix_cache = _rotate_prefix_cache(model, prefix_captures)
        cached_output, cached_latency, _ = _benchmark(
            lambda: _cached_suffix_forward(
                model,
                masked_ids[:, prefix_tokens:],
                timestep,
                prefix_tokens=prefix_tokens,
                prefix_kv=rotated_prefix_cache,
            ),
            device=device,
            repeats=1,
        )
        cached_suffix_latencies.append(cached_latency)

        original_metrics = masked_token_metrics(
            original_logits,
            targets,
            mask_token_id=mask_token_id,
        )
        isolated_metrics = masked_token_metrics(
            isolated_logits,
            targets,
            mask_token_id=mask_token_id,
        )
        for totals, metrics in (
            (original_totals, original_metrics),
            (isolated_totals, isolated_metrics),
        ):
            totals["correct"] += metrics["correct"]
            totals["tokens"] += metrics["tokens"]
            totals["nll"] += metrics["negative_log_likelihood_sum"]

        isolated_top1 = _top1_ids(isolated_logits, mask_token_id)
        cached_top1 = _top1_ids(cached_output, mask_token_id)
        original_top1 = _top1_ids(original_logits, mask_token_id)
        cache_agreement_correct += int(isolated_top1.eq(cached_top1).sum().item())
        cache_agreement_tokens += isolated_top1.numel()
        cached_drift = tensor_drift_metrics(isolated_logits, cached_output)
        maximum_cached_logit_nrmse = max(
            maximum_cached_logit_nrmse,
            float(cached_drift["normalized_rmse"]),
        )
        if sample_index < example_limit:
            examples.append(
                {
                    "prefix_text": tokenizer.decode(
                        prefix_ids[0].detach().cpu().tolist()
                    ),
                    "target_suffix_text": tokenizer.decode(
                        targets[0].detach().cpu().tolist()
                    ),
                    "original_one_step_top1_text": tokenizer.decode(
                        original_top1[0].detach().cpu().tolist()
                    ),
                    "prefix_isolated_one_step_top1_text": tokenizer.decode(
                        isolated_top1[0].detach().cpu().tolist()
                    ),
                    "cached_one_step_top1_text": tokenizer.decode(
                        cached_top1[0].detach().cpu().tolist()
                    ),
                }
            )

    original_accuracy = original_totals["correct"] / original_totals["tokens"]
    isolated_accuracy = isolated_totals["correct"] / isolated_totals["tokens"]
    original_nll = original_totals["nll"] / original_totals["tokens"]
    isolated_nll = isolated_totals["nll"] / isolated_totals["tokens"]
    cache_bytes = (
        2
        * int(config.n_blocks)
        * prefix_tokens
        * int(config.hidden_dim)
        * next(model.parameters()).element_size()
    )
    full_latency_ms = _median_milliseconds(isolated_latencies)
    cached_latency_ms = _median_milliseconds(cached_suffix_latencies)
    return {
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "samples": len(windows),
        "scored_tokens": int(original_totals["tokens"]),
        "original_full_attention": {
            "top1_accuracy": original_accuracy,
            "mean_negative_log_likelihood": original_nll,
            "median_latency_ms": _median_milliseconds(original_latencies),
        },
        "prefix_isolated_recompute": {
            "top1_accuracy": isolated_accuracy,
            "mean_negative_log_likelihood": isolated_nll,
            "median_latency_ms": full_latency_ms,
        },
        "prefix_isolated_cache": {
            "top1_agreement_with_recompute": (
                cache_agreement_correct / cache_agreement_tokens
            ),
            "maximum_logit_normalized_rmse": maximum_cached_logit_nrmse,
            "median_one_time_prefix_prefill_latency_ms": _median_milliseconds(
                prefix_prefill_latencies
            ),
            "median_cached_suffix_latency_ms": cached_latency_ms,
            "steady_state_latency_reduction_fraction": (
                (full_latency_ms - cached_latency_ms) / full_latency_ms
            ),
            "prefix_kv_bytes": cache_bytes,
            "prefix_kv_mib": cache_bytes / 1024**2,
        },
        "examples": examples,
    }


def evaluate_mdlm_prefix_isolation_quality(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: str = "auto",
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    split: str = "test",
    prefix_lengths: Sequence[int] = (16, 64, 128),
    suffix_lengths: Sequence[int] = (16, 32, 64),
    samples: int = 8,
    seed: int = 0,
    maximum_accuracy_drop_fraction: float = 0.05,
    maximum_nll_increase_fraction: float = 0.05,
    maximum_logit_nrmse: float = 1e-5,
    local_files_only: bool = False,
) -> Dict[str, object]:
    """Evaluate reconstruction quality and cached execution on held-out text."""
    if not prefix_lengths or not suffix_lengths:
        raise ValueError("prefix_lengths and suffix_lengths must not be empty")
    if min(prefix_lengths) <= 0 or min(suffix_lengths) <= 0:
        raise ValueError("prefix and suffix lengths must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if (
        maximum_accuracy_drop_fraction < 0
        or maximum_nll_increase_fraction < 0
        or maximum_logit_nrmse < 0
    ):
        raise ValueError("gate limits must be non-negative")

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
    maximum_window = max(prefix_lengths) + max(suffix_lengths)
    if maximum_window > int(model.config.model_length):
        raise ValueError("evaluation window exceeds checkpoint context length")
    windows = collect_token_windows(
        dataset["text"],
        tokenizer,
        window_tokens=maximum_window,
        samples=samples,
        seed=seed,
    )

    configurations = []
    representative_prefix = prefix_lengths[len(prefix_lengths) // 2]
    representative_suffix = suffix_lengths[len(suffix_lengths) // 2]
    for prefix_tokens in prefix_lengths:
        for suffix_tokens in suffix_lengths:
            configurations.append(
                _evaluate_configuration(
                    model=model,
                    windows=windows,
                    prefix_tokens=prefix_tokens,
                    suffix_tokens=suffix_tokens,
                    device=resolved_device,
                    tokenizer=tokenizer,
                    example_limit=(
                        2
                        if prefix_tokens == representative_prefix
                        and suffix_tokens == representative_suffix
                        else 0
                    ),
                )
            )

    total_original_correct = 0.0
    total_isolated_correct = 0.0
    total_original_nll = 0.0
    total_isolated_nll = 0.0
    total_tokens = 0
    minimum_cache_agreement = 1.0
    maximum_cache_nrmse = 0.0
    for configuration in configurations:
        tokens = int(configuration["scored_tokens"])
        total_tokens += tokens
        total_original_correct += (
            configuration["original_full_attention"]["top1_accuracy"] * tokens
        )
        total_isolated_correct += (
            configuration["prefix_isolated_recompute"]["top1_accuracy"] * tokens
        )
        total_original_nll += (
            configuration["original_full_attention"][
                "mean_negative_log_likelihood"
            ]
            * tokens
        )
        total_isolated_nll += (
            configuration["prefix_isolated_recompute"][
                "mean_negative_log_likelihood"
            ]
            * tokens
        )
        minimum_cache_agreement = min(
            minimum_cache_agreement,
            configuration["prefix_isolated_cache"][
                "top1_agreement_with_recompute"
            ],
        )
        maximum_cache_nrmse = max(
            maximum_cache_nrmse,
            configuration["prefix_isolated_cache"][
                "maximum_logit_normalized_rmse"
            ],
        )
    original_accuracy = total_original_correct / total_tokens
    isolated_accuracy = total_isolated_correct / total_tokens
    original_nll = total_original_nll / total_tokens
    isolated_nll = total_isolated_nll / total_tokens
    gates = evaluate_quality_gates(
        original_accuracy=original_accuracy,
        isolated_accuracy=isolated_accuracy,
        original_nll=original_nll,
        isolated_nll=isolated_nll,
        cached_top1_agreement=minimum_cache_agreement,
        maximum_cached_logit_nrmse=maximum_cache_nrmse,
        maximum_accuracy_drop_fraction=maximum_accuracy_drop_fraction,
        maximum_nll_increase_fraction=maximum_nll_increase_fraction,
        maximum_logit_nrmse=maximum_logit_nrmse,
    )
    representative_examples = next(
        configuration["examples"]
        for configuration in configurations
        if configuration["examples"]
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
            "samples": samples,
            "seed": seed,
        },
        "protocol": (
            "one-step true-token reconstruction with a fully masked held-out suffix"
        ),
        "quality_scope": (
            "reconstruction gate, not complete DDPM generation or chat quality"
        ),
        "aggregate": {
            "scored_tokens": total_tokens,
            "original_top1_accuracy": original_accuracy,
            "prefix_isolated_top1_accuracy": isolated_accuracy,
            "original_mean_negative_log_likelihood": original_nll,
            "prefix_isolated_mean_negative_log_likelihood": isolated_nll,
            "minimum_cached_top1_agreement": minimum_cache_agreement,
            "maximum_cached_logit_normalized_rmse": maximum_cache_nrmse,
        },
        "gates": gates,
        "representative_examples": representative_examples,
        "configurations": configurations,
    }
