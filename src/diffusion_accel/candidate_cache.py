"""Distribution-equivalence validation for compact DDPM candidate caching."""

from __future__ import annotations

import math
from typing import Dict


def analyze_candidate_reveal_kernel(
    *,
    positions: int = 64,
    vocabulary_size: int = 50_258,
    cache_hit_transitions: int = 23,
    clock_mhz: float = 300.0,
    initiation_interval: int = 1,
    measured_model_forward_ms: float = 0.0,
) -> Dict[str, object]:
    """Estimate the streaming reveal controller's cycle and traffic budget."""
    if min(
        positions,
        vocabulary_size,
        cache_hit_transitions,
        initiation_interval,
    ) <= 0:
        raise ValueError("dimensions, hits, and initiation_interval must be positive")
    if clock_mhz <= 0 or measured_model_forward_ms < 0:
        raise ValueError("clock_mhz must be positive and measured time non-negative")

    candidate_id_bits = max(1, math.ceil(math.log2(vocabulary_size)))
    candidate_id_bytes = math.ceil(candidate_id_bits / 8)
    bitmap_bytes = math.ceil(positions / 8)
    command_cycles = 1
    stream_cycles = positions * initiation_interval
    downstream_invalidation_cycles = math.ceil(positions / 64)
    cycles_per_hit = command_cycles + stream_cycles + downstream_invalidation_cycles
    total_cycles = cycles_per_hit * cache_hit_transitions
    total_latency_us = total_cycles / clock_mhz

    input_bytes_per_hit = (
        positions * (2 * candidate_id_bytes + 4)
        + 2 * bitmap_bytes
        + 8
    )
    output_bytes_per_hit = positions * candidate_id_bytes + bitmap_bytes + 8
    stream_bytes_per_hit = input_bytes_per_hit + output_bytes_per_hit
    probability_cache_bytes_per_hit = positions * vocabulary_size * 2
    model_forward_fraction = (
        (total_latency_us / 1000) / measured_model_forward_ms
        if measured_model_forward_ms
        else None
    )
    return {
        "configuration": {
            "positions": positions,
            "vocabulary_size": vocabulary_size,
            "candidate_id_bits": candidate_id_bits,
            "cache_hit_transitions": cache_hit_transitions,
            "clock_mhz": clock_mhz,
            "assumed_initiation_interval": initiation_interval,
            "measured_model_forward_ms": measured_model_forward_ms,
        },
        "cycles": {
            "command_cycles_per_hit": command_cycles,
            "stream_cycles_per_hit": stream_cycles,
            "downstream_invalidation_cycles_per_hit": (
                downstream_invalidation_cycles
            ),
            "total_cycles_per_hit": cycles_per_hit,
            "total_cycles_all_hits": total_cycles,
            "total_latency_us": total_latency_us,
            "fraction_of_measured_model_forward_time": model_forward_fraction,
        },
        "traffic": {
            "stream_input_bytes_per_hit": input_bytes_per_hit,
            "stream_output_bytes_per_hit": output_bytes_per_hit,
            "stream_bytes_per_hit": stream_bytes_per_hit,
            "full_fp16_probability_cache_bytes_per_hit": (
                probability_cache_bytes_per_hit
            ),
            "full_probability_to_stream_traffic_ratio": (
                probability_cache_bytes_per_hit / stream_bytes_per_hit
            ),
        },
        "provenance": (
            "derived-cycle-model-not-post-synthesis-timing"
        ),
    }


def validate_candidate_cache_equivalence(
    *,
    positions: int = 64,
    vocabulary_size: int = 50_258,
    probability_bits: int = 16,
    analytic_vocabulary_size: int = 257,
    transitions: int = 16,
    monte_carlo_trials: int = 200_000,
    monte_carlo_vocabulary_size: int = 16,
    seed: int = 0,
    maximum_total_variation: float = 0.01,
) -> Dict[str, object]:
    """Check the exact factorization and a sampled categorical experiment."""
    values = {
        "positions": positions,
        "vocabulary_size": vocabulary_size,
        "probability_bits": probability_bits,
        "analytic_vocabulary_size": analytic_vocabulary_size,
        "transitions": transitions,
        "monte_carlo_trials": monte_carlo_trials,
        "monte_carlo_vocabulary_size": monte_carlo_vocabulary_size,
    }
    if min(values.values()) <= 0:
        raise ValueError("all dimensions and trial counts must be positive")
    if probability_bits % 8:
        raise ValueError("probability_bits must be byte aligned")
    if not 0 < maximum_total_variation < 1:
        raise ValueError("maximum_total_variation must be between zero and one")

    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(
        (positions, analytic_vocabulary_size),
        generator=generator,
        dtype=torch.float64,
    )
    token_probabilities = torch.softmax(logits, dim=-1)
    times = torch.linspace(1.0, 1e-5, transitions + 1, dtype=torch.float64)
    maximum_absolute_error = 0.0
    maximum_row_sum_error = 0.0
    for transition in range(transitions):
        move_t = 0.999 * times[transition]
        move_s = 0.999 * times[transition + 1]
        token_mass = move_t - move_s
        original_tokens = token_probabilities * token_mass / move_t
        original_mask = torch.full(
            (positions, 1),
            move_s / move_t,
            dtype=token_probabilities.dtype,
        )
        original = torch.cat([original_tokens, original_mask], dim=-1)

        reveal_probability = token_mass / move_t
        factored = torch.cat(
            [
                token_probabilities * reveal_probability,
                torch.full(
                    (positions, 1),
                    1 - reveal_probability,
                    dtype=token_probabilities.dtype,
                ),
            ],
            dim=-1,
        )
        maximum_absolute_error = max(
            maximum_absolute_error,
            float((original - factored).abs().max().item()),
        )
        maximum_row_sum_error = max(
            maximum_row_sum_error,
            float((factored.sum(dim=-1) - 1).abs().max().item()),
        )

    monte_carlo_logits = torch.randn(
        monte_carlo_vocabulary_size,
        generator=generator,
        dtype=torch.float64,
    )
    monte_carlo_probabilities = torch.softmax(monte_carlo_logits, dim=-1)
    move_t = torch.tensor(0.73, dtype=torch.float64)
    move_s = torch.tensor(0.41, dtype=torch.float64)
    original_distribution = torch.cat(
        [
            monte_carlo_probabilities * (move_t - move_s) / move_t,
            (move_s / move_t).reshape(1),
        ]
    )
    baseline_samples = torch.multinomial(
        original_distribution,
        monte_carlo_trials,
        replacement=True,
        generator=generator,
    )

    candidates = torch.multinomial(
        monte_carlo_probabilities,
        monte_carlo_trials,
        replacement=True,
        generator=generator,
    )
    reveal = torch.rand(monte_carlo_trials, generator=generator) < (
        (move_t - move_s) / move_t
    )
    factored_samples = torch.where(
        reveal,
        candidates,
        torch.full_like(candidates, monte_carlo_vocabulary_size),
    )
    baseline_histogram = torch.bincount(
        baseline_samples,
        minlength=monte_carlo_vocabulary_size + 1,
    ).to(torch.float64) / monte_carlo_trials
    factored_histogram = torch.bincount(
        factored_samples,
        minlength=monte_carlo_vocabulary_size + 1,
    ).to(torch.float64) / monte_carlo_trials
    total_variation = float(
        (baseline_histogram - factored_histogram).abs().sum().item() / 2
    )

    probability_cache_bytes = (
        positions * vocabulary_size * (probability_bits // 8)
    )
    candidate_id_bits = max(1, math.ceil(math.log2(vocabulary_size)))
    candidate_id_bytes = math.ceil(candidate_id_bits / 8)
    candidate_token_bytes = positions * candidate_id_bytes
    bitmap_bytes = 2 * math.ceil(positions / 8)
    candidate_cache_bytes = candidate_token_bytes + bitmap_bytes
    return {
        "method": (
            "sample-token-on-model-evaluation-and-reuse-until-input-change"
        ),
        "correctness_class": "distribution-equivalent",
        "configuration": {
            **values,
            "seed": seed,
            "maximum_total_variation": maximum_total_variation,
        },
        "analytic_factorization": {
            "maximum_absolute_probability_error": maximum_absolute_error,
            "maximum_row_sum_error": maximum_row_sum_error,
            "passed": maximum_absolute_error < 1e-12,
        },
        "monte_carlo": {
            "total_variation_distance": total_variation,
            "passed": total_variation <= maximum_total_variation,
        },
        "storage": {
            "full_probability_cache_bytes": probability_cache_bytes,
            "full_probability_cache_mib": probability_cache_bytes / 1024**2,
            "candidate_token_bytes": candidate_token_bytes,
            "candidate_id_bits": candidate_id_bits,
            "candidate_id_bytes": candidate_id_bytes,
            "active_and_valid_bitmap_bytes": bitmap_bytes,
            "candidate_cache_bytes": candidate_cache_bytes,
            "storage_reduction_factor": (
                probability_cache_bytes / candidate_cache_bytes
            ),
        },
        "passed": (
            maximum_absolute_error < 1e-12
            and total_variation <= maximum_total_variation
        ),
    }
