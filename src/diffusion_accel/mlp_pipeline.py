"""Cycle and storage model for the fixed MDLM ping-pong MLP engine."""

from __future__ import annotations

import math
from typing import Dict


def _layer_model(
    *,
    name: str,
    tokens: int,
    input_size: int,
    output_size: int,
    token_lanes: int,
    output_lanes: int,
    inner_lanes: int,
    bytes_per_cycle: float,
    stream_bytes: int,
) -> Dict[str, object]:
    token_groups = math.ceil(tokens / token_lanes)
    inner_tiles = math.ceil(input_size / inner_lanes)
    output_tiles = math.ceil(output_size / output_lanes)
    compute_cycles_per_tile = token_groups * inner_tiles
    weight_bytes_per_tile = output_lanes * input_size
    metadata_bytes_per_tile = token_lanes * output_lanes * (3 + 4)
    load_bytes_per_tile = weight_bytes_per_tile + metadata_bytes_per_tile
    load_cycles_per_tile = math.ceil(load_bytes_per_tile / bytes_per_cycle)
    adapter_beats_per_inner_tile = math.ceil(
        (output_lanes * inner_lanes) / stream_bytes
    )
    adapter_cycles_per_output_tile = inner_tiles * adapter_beats_per_inner_tile
    overlap_spill_cycles = max(0, load_cycles_per_tile - compute_cycles_per_tile)
    single_buffer_cycles = output_tiles * (
        compute_cycles_per_tile + load_cycles_per_tile
    )
    pingpong_cycles = (
        load_cycles_per_tile
        + output_tiles * compute_cycles_per_tile
        + (output_tiles - 1) * (1 + overlap_spill_cycles)
    )
    return {
        "name": name,
        "shape": [tokens, input_size, output_size],
        "token_groups": token_groups,
        "inner_tiles": inner_tiles,
        "output_tiles": output_tiles,
        "compute_cycles_per_output_tile": compute_cycles_per_tile,
        "weight_bytes_per_output_tile": weight_bytes_per_tile,
        "requant_metadata_bytes_per_output_tile": metadata_bytes_per_tile,
        "modeled_ddr_load_cycles_per_output_tile": load_cycles_per_tile,
        "stream_adapter_beats_per_inner_tile": adapter_beats_per_inner_tile,
        "stream_adapter_cycles_per_output_tile": adapter_cycles_per_output_tile,
        "compute_to_load_cycle_ratio": (
            compute_cycles_per_tile / load_cycles_per_tile
        ),
        "overlap_spill_cycles_per_output_tile": overlap_spill_cycles,
        "activation_store_bytes": tokens * input_size,
        "single_weight_buffer_bytes": weight_bytes_per_tile,
        "pingpong_weight_buffer_bytes": 2 * weight_bytes_per_tile,
        "single_buffer_issue_cycles": single_buffer_cycles,
        "pingpong_issue_cycles": pingpong_cycles,
        "pingpong_saved_issue_cycles": single_buffer_cycles - pingpong_cycles,
    }


def analyze_mlp_pingpong(
    *,
    tokens: int = 64,
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    token_lanes: int = 4,
    output_lanes: int = 6,
    inner_lanes: int = 32,
    clock_mhz: float = 250.0,
    effective_ddr_gbps: float = 12.48,
    stream_width_bits: int = 512,
    blocks: int = 12,
    evaluations: int = 8,
) -> Dict[str, object]:
    """Model double-buffer overlap for both fixed MDLM MLP matrices."""
    positive_integers = (
        tokens,
        hidden_size,
        intermediate_size,
        token_lanes,
        output_lanes,
        inner_lanes,
        blocks,
        evaluations,
        stream_width_bits,
    )
    if any(value <= 0 for value in positive_integers):
        raise ValueError("all sizes and counts must be positive")
    if clock_mhz <= 0 or effective_ddr_gbps <= 0:
        raise ValueError("clock and DDR bandwidth must be positive")
    bytes_per_cycle = (
        effective_ddr_gbps * 1_000_000_000.0
        / (clock_mhz * 1_000_000.0)
    )
    if stream_width_bits % 8 != 0:
        raise ValueError("stream width must be byte aligned")
    stream_bytes = stream_width_bits // 8
    up = _layer_model(
        name="mlp_up",
        tokens=tokens,
        input_size=hidden_size,
        output_size=intermediate_size,
        token_lanes=token_lanes,
        output_lanes=output_lanes,
        inner_lanes=inner_lanes,
        bytes_per_cycle=bytes_per_cycle,
        stream_bytes=stream_bytes,
    )
    down = _layer_model(
        name="mlp_down",
        tokens=tokens,
        input_size=intermediate_size,
        output_size=hidden_size,
        token_lanes=token_lanes,
        output_lanes=output_lanes,
        inner_lanes=inner_lanes,
        bytes_per_cycle=bytes_per_cycle,
        stream_bytes=stream_bytes,
    )
    single_cycles_per_block = int(up["single_buffer_issue_cycles"]) + int(
        down["single_buffer_issue_cycles"]
    )
    pingpong_cycles_per_block = int(up["pingpong_issue_cycles"]) + int(
        down["pingpong_issue_cycles"]
    )
    saved_cycles_per_block = single_cycles_per_block - pingpong_cycles_per_block
    total_saved_cycles = saved_cycles_per_block * blocks * evaluations
    return {
        "configuration": {
            "tokens": tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "token_lanes": token_lanes,
            "output_lanes": output_lanes,
            "inner_lanes": inner_lanes,
            "clock_mhz": clock_mhz,
            "effective_ddr_gbps": effective_ddr_gbps,
            "effective_ddr_bytes_per_cycle": bytes_per_cycle,
            "stream_width_bits": stream_width_bits,
            "blocks": blocks,
            "evaluations": evaluations,
        },
        "layers": [up, down],
        "summary": {
            "single_buffer_issue_cycles_per_block": single_cycles_per_block,
            "pingpong_issue_cycles_per_block": pingpong_cycles_per_block,
            "saved_issue_cycles_per_block": saved_cycles_per_block,
            "saved_latency_ms_per_block": (
                saved_cycles_per_block / (clock_mhz * 1_000_000.0) * 1_000.0
            ),
            "saved_latency_ms_for_generation": (
                total_saved_cycles / (clock_mhz * 1_000_000.0) * 1_000.0
            ),
            "all_weight_loads_hidden_after_initial_tile": all(
                int(layer["overlap_spill_cycles_per_output_tile"]) == 0
                for layer in (up, down)
            ),
        },
        "scope": (
            "issue-cycle model with one launch bubble between output tiles; "
            "excludes AXI burst setup, bank conflicts, vendor timing, and final drain"
        ),
    }
