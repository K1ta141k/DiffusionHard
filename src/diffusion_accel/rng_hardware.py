"""Cycle and structural-resource model for hardware Philox alternatives."""

from __future__ import annotations

import math
from typing import Dict, Mapping


def _latency_ms(cycles: int, clock_mhz: float) -> float:
    return cycles / (clock_mhz * 1_000.0)


def analyze_rng_hardware(
    *,
    positions: int = 64,
    vocabulary_size: int = 50_258,
    hidden_size: int = 768,
    output_head_mac_lanes: int = 1_024,
    clock_mhz: float = 300.0,
    philox_rounds: int = 10,
    words_per_block: int = 4,
    gumbel_lanes: int = 2,
    maximum_iterative_cores: int = 8,
    unrolled_usage: Mapping[str, int] | None = None,
    iterative_core_usage: Mapping[str, int] | None = None,
    four_core_farm_usage: Mapping[str, int] | None = None,
    dual_gumbel_usage: Mapping[str, int] | None = None,
    integrated_stream_usage: Mapping[str, int] | None = None,
) -> Dict[str, object]:
    """Compare a combinational stream with a small iterative Philox farm."""
    positive_values = {
        "positions": positions,
        "vocabulary_size": vocabulary_size,
        "hidden_size": hidden_size,
        "output_head_mac_lanes": output_head_mac_lanes,
        "clock_mhz": clock_mhz,
        "philox_rounds": philox_rounds,
        "words_per_block": words_per_block,
        "gumbel_lanes": gumbel_lanes,
        "maximum_iterative_cores": maximum_iterative_cores,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    score_count = positions * vocabulary_size
    blocks_per_position = math.ceil(vocabulary_size / words_per_block)
    block_count = positions * blocks_per_position
    output_head_macs = score_count * hidden_size
    output_head_cycles = math.ceil(output_head_macs / output_head_mac_lanes)
    required_score_rate = score_count / output_head_cycles

    iterative_points = []
    minimum_non_bottleneck_cores = None
    for core_count in range(1, maximum_iterative_cores + 1):
        generation_cycles = math.ceil(block_count / core_count) * philox_rounds
        transform_cycles = math.ceil(score_count / gumbel_lanes)
        rng_cycles = max(generation_cycles, transform_cycles)
        point = {
            "core_count": core_count,
            "average_generated_scores_per_cycle": (
                core_count * words_per_block / philox_rounds
            ),
            "generation_cycles": generation_cycles,
            "transform_cycles": transform_cycles,
            "rng_cycles": rng_cycles,
            "rng_latency_ms": _latency_ms(rng_cycles, clock_mhz),
            "output_head_bottleneck": rng_cycles > output_head_cycles,
        }
        if iterative_core_usage is not None:
            point["projected_core_only_resources"] = {
                name: count * core_count
                for name, count in iterative_core_usage.items()
            }
            point["resource_scope"] = (
                "linear core-only projection; excludes dispatcher, FIFOs, "
                "and Gumbel transform"
            )
        iterative_points.append(point)
        if rng_cycles <= output_head_cycles and minimum_non_bottleneck_cores is None:
            minimum_non_bottleneck_cores = core_count

    unrolled_cycles = score_count
    unrolled = {
        "score_rate_per_cycle": 1.0,
        "rng_cycles": unrolled_cycles,
        "rng_latency_ms": _latency_ms(unrolled_cycles, clock_mhz),
        "output_head_bottleneck": unrolled_cycles > output_head_cycles,
        "timing_validated": False,
    }
    if unrolled_usage is not None:
        unrolled["mapped_resources"] = dict(unrolled_usage)

    farm_summary: Dict[str, object] = {
        "minimum_non_bottleneck_cores": minimum_non_bottleneck_cores,
        "design_points": iterative_points,
    }
    if four_core_farm_usage is not None:
        farm_summary["mapped_four_core_farm_resources"] = dict(
            four_core_farm_usage
        )
    if four_core_farm_usage is not None and dual_gumbel_usage is not None:
        resource_names = set(four_core_farm_usage) | set(dual_gumbel_usage)
        farm_summary["mapped_farm_plus_dual_gumbel_resources"] = {
            name: four_core_farm_usage.get(name, 0)
            + dual_gumbel_usage.get(name, 0)
            for name in sorted(resource_names)
        }
        farm_summary["mapped_combined_resource_scope"] = (
            "sum of separately mapped four-core farm and dual Gumbel pipeline; "
            "excludes burst FIFO and block-to-pair adapter"
        )
    if integrated_stream_usage is not None:
        farm_summary["mapped_integrated_stream_resources"] = dict(
            integrated_stream_usage
        )
        farm_summary["mapped_integrated_stream_scope"] = (
            "four iterative Philox cores, four-block burst FIFO, block-to-pair "
            "adapter, and dual Q5.10 Gumbel pipeline"
        )

    return {
        "configuration": positive_values,
        "workload": {
            "score_count": score_count,
            "philox_block_count": block_count,
            "output_head_macs": output_head_macs,
            "output_head_ideal_cycles": output_head_cycles,
            "output_head_ideal_latency_ms": _latency_ms(
                output_head_cycles, clock_mhz
            ),
            "required_rng_scores_per_cycle": required_score_rate,
        },
        "unrolled_stream": unrolled,
        "iterative_farm": farm_summary,
        "scope": (
            "analytical steady-state cycle bound plus linear core-only "
            "resource projection; no place-and-route timing"
        ),
    }
