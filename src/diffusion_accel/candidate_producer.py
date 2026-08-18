"""Streaming categorical candidate selection and output-head cost model."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence


def analyze_output_head_lane_sweep(
    active_tokens: Sequence[int],
    *,
    vocabulary_size: int = 50_258,
    hidden_size: int = 768,
    weight_bits: Sequence[int] = (16, 8),
    mac_lanes: Sequence[int] = (256, 512, 768, 1_024, 1_248),
    clock_mhz: float = 300.0,
    ddr_bandwidth_gbps: float = 19.2,
    ddr_efficiency: float = 0.65,
) -> Dict[str, object]:
    """Sweep output-head lanes over a measured active-position schedule."""
    if not active_tokens or min(active_tokens) <= 0:
        raise ValueError("active_tokens must contain positive counts")
    if min(vocabulary_size, hidden_size) <= 0:
        raise ValueError("model dimensions must be positive")
    if not weight_bits or min(weight_bits) <= 0 or any(bits % 8 for bits in weight_bits):
        raise ValueError("weight_bits must contain positive byte-aligned widths")
    if not mac_lanes or min(mac_lanes) <= 0:
        raise ValueError("mac_lanes must contain positive counts")
    if clock_mhz <= 0 or ddr_bandwidth_gbps <= 0:
        raise ValueError("clock and DDR bandwidth must be positive")
    if not 0 < ddr_efficiency <= 1:
        raise ValueError("ddr_efficiency must be in (0, 1]")

    effective_ddr_bytes_per_second = ddr_bandwidth_gbps * 1e9 * ddr_efficiency
    results = []
    for bits in weight_bits:
        output_weight_bytes = vocabulary_size * hidden_size * (bits // 8)
        memory_ms_per_evaluation = (
            output_weight_bytes / effective_ddr_bytes_per_second * 1e3
        )
        for lanes in mac_lanes:
            compute_ms = [
                tokens
                * vocabulary_size
                * hidden_size
                / (lanes * clock_mhz * 1e6)
                * 1e3
                for tokens in active_tokens
            ]
            roofline_ms = [
                max(compute, memory_ms_per_evaluation)
                for compute in compute_ms
            ]
            results.append(
                {
                    "weight_bits": bits,
                    "mac_lanes": lanes,
                    "output_weight_bytes_per_evaluation": output_weight_bytes,
                    "total_output_weight_traffic_bytes": (
                        output_weight_bytes * len(active_tokens)
                    ),
                    "compute_only_total_ms": sum(compute_ms),
                    "memory_only_total_ms": (
                        memory_ms_per_evaluation * len(active_tokens)
                    ),
                    "roofline_total_ms": sum(roofline_ms),
                    "ddr_bound_evaluations": sum(
                        memory_ms_per_evaluation >= compute
                        for compute in compute_ms
                    ),
                    "compute_bound_evaluations": sum(
                        compute > memory_ms_per_evaluation
                        for compute in compute_ms
                    ),
                    "crossover_active_tokens": (
                        memory_ms_per_evaluation
                        * lanes
                        * clock_mhz
                        * 1e6
                        / 1e3
                        / (vocabulary_size * hidden_size)
                    ),
                }
            )
    return {
        "schedule": {
            "model_evaluations": len(active_tokens),
            "active_token_positions": sum(active_tokens),
            "minimum_active_tokens": min(active_tokens),
            "maximum_active_tokens": max(active_tokens),
            "active_tokens": list(active_tokens),
        },
        "configuration": {
            "vocabulary_size": vocabulary_size,
            "hidden_size": hidden_size,
            "clock_mhz": clock_mhz,
            "ddr_bandwidth_gbps": ddr_bandwidth_gbps,
            "ddr_efficiency": ddr_efficiency,
            "weight_bits": list(weight_bits),
            "mac_lanes": list(mac_lanes),
        },
        "design_points": results,
        "scope": (
            "output-projection-roofline-only; assumes-one-weight-matrix-read-"
            "per-evaluation-and-one-mac-per-lane-per-cycle"
        ),
    }


def streaming_exponential_race_candidates(
    logits: Any,
    uniforms: Any,
    *,
    mask_token_id: int,
    chunk_size: int,
) -> Any:
    """Select categorical samples without materializing softmax probabilities.

    MDLM samples ``argmax(softmax(logits) / exponential_noise)``. The shared
    softmax normalization does not affect the argmax, so a streaming producer
    can instead maximize ``logit - log(exponential_noise)`` over vocabulary
    chunks while retaining only one score and token ID per position.
    """
    import torch

    if logits.shape != uniforms.shape or logits.ndim < 1:
        raise ValueError("logits and uniforms must have the same non-empty shape")
    vocabulary_size = logits.shape[-1]
    if not 0 <= mask_token_id < vocabulary_size:
        raise ValueError("mask_token_id must be inside the vocabulary")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if bool(((uniforms < 0) | (uniforms >= 1)).any().item()):
        raise ValueError("uniforms must be in [0, 1)")

    flattened_logits = logits.reshape(-1, vocabulary_size)
    flattened_uniforms = uniforms.reshape(-1, vocabulary_size)
    best_scores = torch.full(
        (flattened_logits.shape[0],),
        -torch.inf,
        dtype=logits.dtype,
        device=logits.device,
    )
    best_ids = torch.zeros(
        (flattened_logits.shape[0],),
        dtype=torch.long,
        device=logits.device,
    )
    for start in range(0, vocabulary_size, chunk_size):
        end = min(start + chunk_size, vocabulary_size)
        exponential = 1e-10 - torch.log(
            flattened_uniforms[:, start:end] + 1e-10
        )
        scores = flattened_logits[:, start:end] - torch.log(exponential)
        if start <= mask_token_id < end:
            scores[:, mask_token_id - start] = -torch.inf
        chunk_scores, chunk_ids = scores.max(dim=-1)
        replace = chunk_scores > best_scores
        best_scores = torch.where(replace, chunk_scores, best_scores)
        best_ids = torch.where(replace, chunk_ids + start, best_ids)
    return best_ids.reshape(logits.shape[:-1])


def validate_streaming_candidate_producer(
    *,
    positions: int = 64,
    vocabulary_size: int = 50_258,
    hidden_size: int = 768,
    trials: int = 8,
    chunk_size: int = 512,
    seed: int = 0,
    weight_bits: int = 16,
    activation_bits: int = 16,
    score_bits: int = 32,
    mac_lanes: int = 1_024,
    clock_mhz: float = 300.0,
    ddr_bandwidth_gbps: float = 19.2,
    ddr_efficiency: float = 0.65,
) -> Dict[str, object]:
    """Validate streaming samples and bound a fused output-head design point."""
    integer_values = {
        "positions": positions,
        "vocabulary_size": vocabulary_size,
        "hidden_size": hidden_size,
        "trials": trials,
        "chunk_size": chunk_size,
        "weight_bits": weight_bits,
        "activation_bits": activation_bits,
        "score_bits": score_bits,
        "mac_lanes": mac_lanes,
    }
    if min(integer_values.values()) <= 0:
        raise ValueError("dimensions, bit widths, trials, and lanes must be positive")
    if weight_bits % 8 or activation_bits % 8:
        raise ValueError("weight_bits and activation_bits must be byte aligned")
    if clock_mhz <= 0 or ddr_bandwidth_gbps <= 0:
        raise ValueError("clock and DDR bandwidth must be positive")
    if not 0 < ddr_efficiency <= 1:
        raise ValueError("ddr_efficiency must be in (0, 1]")

    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    matching = 0
    mask_token_id = vocabulary_size - 1
    for _ in range(trials):
        logits = torch.randn(
            (positions, vocabulary_size),
            generator=generator,
            dtype=torch.float64,
        )
        uniforms = torch.rand(
            (positions, vocabulary_size),
            generator=generator,
            dtype=torch.float64,
        )
        masked_logits = logits.clone()
        masked_logits[:, mask_token_id] = -torch.inf
        probabilities = torch.softmax(masked_logits, dim=-1)
        exponential = 1e-10 - torch.log(uniforms + 1e-10)
        reference = (probabilities / exponential).argmax(dim=-1)
        streamed = streaming_exponential_race_candidates(
            logits,
            uniforms,
            mask_token_id=mask_token_id,
            chunk_size=chunk_size,
        )
        matching += int(reference.eq(streamed).sum().item())

    samples = trials * positions
    candidate_id_bits = max(1, math.ceil(math.log2(vocabulary_size)))
    candidate_id_bytes = math.ceil(candidate_id_bits / 8)
    bitmap_bytes = math.ceil(positions / 8)
    candidate_cache_bytes = positions * candidate_id_bytes + 2 * bitmap_bytes
    accumulator_bytes = math.ceil(
        positions * (score_bits + candidate_id_bits + 1) / 8
    )
    logits_bytes = positions * vocabulary_size * (activation_bits // 8)
    output_weight_bytes = (
        hidden_size * vocabulary_size * (weight_bits // 8)
    )
    hidden_activation_bytes = positions * hidden_size * (activation_bits // 8)
    baseline_ddr_bytes = (
        output_weight_bytes + hidden_activation_bytes + 2 * logits_bytes
    )
    fused_ddr_bytes = (
        output_weight_bytes + hidden_activation_bytes + candidate_cache_bytes
    )
    effective_ddr_bytes_per_second = (
        ddr_bandwidth_gbps * 1e9 * ddr_efficiency
    )
    projection_macs = positions * vocabulary_size * hidden_size
    compute_lower_bound_ms = (
        projection_macs / (mac_lanes * clock_mhz * 1e6) * 1e3
    )
    baseline_memory_lower_bound_ms = (
        baseline_ddr_bytes / effective_ddr_bytes_per_second * 1e3
    )
    fused_memory_lower_bound_ms = (
        fused_ddr_bytes / effective_ddr_bytes_per_second * 1e3
    )
    baseline_roofline_ms = max(
        compute_lower_bound_ms,
        baseline_memory_lower_bound_ms,
    )
    fused_roofline_ms = max(
        compute_lower_bound_ms,
        fused_memory_lower_bound_ms,
    )
    return {
        "configuration": {
            **integer_values,
            "seed": seed,
            "clock_mhz": clock_mhz,
            "ddr_bandwidth_gbps": ddr_bandwidth_gbps,
            "ddr_efficiency": ddr_efficiency,
            "mask_token_id": mask_token_id,
        },
        "pathwise_validation": {
            "samples": samples,
            "matching_candidates": matching,
            "agreement": matching / samples,
            "passed": matching == samples,
            "numeric_format": "float64-reference",
        },
        "state": {
            "full_probability_tensor_bytes": logits_bytes,
            "external_probability_roundtrip_bytes": 2 * logits_bytes,
            "streaming_accumulator_bytes": accumulator_bytes,
            "candidate_cache_bytes": candidate_cache_bytes,
            "external_roundtrip_to_candidate_cache_ratio": (
                2 * logits_bytes / candidate_cache_bytes
            ),
        },
        "output_projection": {
            "projection_macs": projection_macs,
            "output_weight_bytes_per_evaluation": output_weight_bytes,
            "hidden_activation_bytes": hidden_activation_bytes,
            "baseline_ddr_bytes": baseline_ddr_bytes,
            "fused_ddr_bytes": fused_ddr_bytes,
            "ddr_bytes_removed": baseline_ddr_bytes - fused_ddr_bytes,
            "compute_lower_bound_ms": compute_lower_bound_ms,
            "baseline_memory_lower_bound_ms": baseline_memory_lower_bound_ms,
            "fused_memory_lower_bound_ms": fused_memory_lower_bound_ms,
            "baseline_roofline_lower_bound_ms": baseline_roofline_ms,
            "fused_roofline_lower_bound_ms": fused_roofline_ms,
            "roofline_speedup": baseline_roofline_ms / fused_roofline_ms,
        },
        "scope": (
            "software-equivalence-and-analytical-output-head-model; "
            "rng-transform-fixed-point-rtl-not-yet-validated"
        ),
        "passed": matching == samples,
    }
