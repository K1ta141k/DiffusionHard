"""Tiled output projection with fused candidate reduction."""

from __future__ import annotations

import math
from typing import Any, Dict


def tiled_fused_output_candidates(
    hidden: Any,
    weights: Any,
    bias: Any,
    race_noise: Any,
    *,
    mask_token_id: int,
    position_tile: int,
    vocabulary_tile: int,
) -> Any:
    """Project tiled logits and retain only the best noisy token per position."""
    import torch

    if hidden.ndim != 2 or weights.ndim != 2:
        raise ValueError("hidden and weights must be matrices")
    positions, hidden_size = hidden.shape
    vocabulary_size, weight_hidden_size = weights.shape
    if hidden_size != weight_hidden_size:
        raise ValueError("hidden and weight dimensions do not match")
    if bias.shape != (vocabulary_size,):
        raise ValueError("bias must have one value per vocabulary token")
    if race_noise.shape != (positions, vocabulary_size):
        raise ValueError("race_noise must have shape [positions, vocabulary]")
    if not 0 <= mask_token_id < vocabulary_size:
        raise ValueError("mask_token_id must be inside the vocabulary")
    if position_tile <= 0 or vocabulary_tile <= 0:
        raise ValueError("tile sizes must be positive")

    candidates = torch.zeros(positions, dtype=torch.long, device=hidden.device)
    for position_start in range(0, positions, position_tile):
        position_end = min(position_start + position_tile, positions)
        position_hidden = hidden[position_start:position_end]
        best_scores = torch.full(
            (position_end - position_start,),
            -torch.inf,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        best_ids = torch.zeros_like(candidates[position_start:position_end])
        for vocabulary_start in range(0, vocabulary_size, vocabulary_tile):
            vocabulary_end = min(
                vocabulary_start + vocabulary_tile,
                vocabulary_size,
            )
            scores = (
                position_hidden
                @ weights[vocabulary_start:vocabulary_end].transpose(0, 1)
            )
            scores = scores + bias[vocabulary_start:vocabulary_end]
            scores = scores + race_noise[
                position_start:position_end,
                vocabulary_start:vocabulary_end,
            ]
            if vocabulary_start <= mask_token_id < vocabulary_end:
                scores[:, mask_token_id - vocabulary_start] = -torch.inf
            tile_scores, tile_ids = scores.max(dim=-1)
            replace = tile_scores > best_scores
            best_scores = torch.where(replace, tile_scores, best_scores)
            best_ids = torch.where(
                replace,
                tile_ids + vocabulary_start,
                best_ids,
            )
        candidates[position_start:position_end] = best_ids
    return candidates


def validate_tiled_output_head(
    *,
    positions: int = 7,
    hidden_size: int = 19,
    vocabulary_size: int = 37,
    trials: int = 32,
    position_tile: int = 4,
    vocabulary_tile: int = 8,
    seed: int = 0,
    hardware_positions: int = 64,
    hardware_hidden_size: int = 768,
    hardware_vocabulary_size: int = 50_258,
    hardware_vocabulary_tile: int = 16,
) -> Dict[str, object]:
    """Validate tiled equivalence and report full MDLM local-state bounds."""
    dimensions = {
        "positions": positions,
        "hidden_size": hidden_size,
        "vocabulary_size": vocabulary_size,
        "trials": trials,
        "position_tile": position_tile,
        "vocabulary_tile": vocabulary_tile,
        "hardware_positions": hardware_positions,
        "hardware_hidden_size": hardware_hidden_size,
        "hardware_vocabulary_size": hardware_vocabulary_size,
        "hardware_vocabulary_tile": hardware_vocabulary_tile,
    }
    if min(dimensions.values()) <= 0:
        raise ValueError("dimensions, trials, and tile sizes must be positive")

    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    matching = 0
    fp16_matching = 0
    mask_token_id = vocabulary_size - 1
    for _ in range(trials):
        hidden = torch.randn(
            (positions, hidden_size),
            generator=generator,
            dtype=torch.float64,
        )
        weights = torch.randn(
            (vocabulary_size, hidden_size),
            generator=generator,
            dtype=torch.float64,
        )
        bias = torch.randn(
            vocabulary_size,
            generator=generator,
            dtype=torch.float64,
        )
        noise = torch.randn(
            (positions, vocabulary_size),
            generator=generator,
            dtype=torch.float64,
        )
        reference_scores = hidden @ weights.transpose(0, 1) + bias + noise
        reference_scores[:, mask_token_id] = -torch.inf
        reference = reference_scores.argmax(dim=-1)
        tiled = tiled_fused_output_candidates(
            hidden,
            weights,
            bias,
            noise,
            mask_token_id=mask_token_id,
            position_tile=position_tile,
            vocabulary_tile=vocabulary_tile,
        )
        matching += int(reference.eq(tiled).sum().item())
        fp16_hidden = hidden.to(torch.float16)
        fp16_weights = weights.to(torch.float16)
        fp16_bias = bias.to(torch.float16)
        fp16_noise = noise.to(torch.float16)
        fp16_reference_scores = (
            fp16_hidden @ fp16_weights.transpose(0, 1)
            + fp16_bias
            + fp16_noise
        )
        fp16_reference_scores[:, mask_token_id] = -torch.inf
        fp16_tiled = tiled_fused_output_candidates(
            fp16_hidden,
            fp16_weights,
            fp16_bias,
            fp16_noise,
            mask_token_id=mask_token_id,
            position_tile=position_tile,
            vocabulary_tile=vocabulary_tile,
        )
        fp16_matching += int(
            fp16_reference_scores.argmax(dim=-1).eq(fp16_tiled).sum().item()
        )

    samples = positions * trials
    candidate_id_bits = max(
        1,
        math.ceil(math.log2(hardware_vocabulary_size)),
    )
    candidate_cache_bytes = (
        hardware_positions * math.ceil(candidate_id_bits / 8)
        + 2 * math.ceil(hardware_positions / 8)
    )
    accumulator_bytes = hardware_positions * hardware_vocabulary_tile * 4
    hidden_buffer_bytes = hardware_positions * hardware_hidden_size
    weight_tile_bytes = hardware_vocabulary_tile * hardware_hidden_size
    return {
        "functional_validation": {
            "configuration": {
                "positions": positions,
                "hidden_size": hidden_size,
                "vocabulary_size": vocabulary_size,
                "trials": trials,
                "position_tile": position_tile,
                "vocabulary_tile": vocabulary_tile,
                "seed": seed,
            },
            "samples": samples,
            "matching_candidates": matching,
            "agreement": matching / samples,
            "passed": matching == samples,
            "numeric_format": "float64-reference",
        },
        "fp16_baseline_validation": {
            "samples": samples,
            "matching_candidates": fp16_matching,
            "agreement": fp16_matching / samples,
            "passed": fp16_matching == samples,
            "numeric_format": "ieee-fp16-software-baseline",
        },
        "hardware_shape": {
            "positions": hardware_positions,
            "hidden_size": hardware_hidden_size,
            "vocabulary_size": hardware_vocabulary_size,
            "vocabulary_tile": hardware_vocabulary_tile,
            "int8_hidden_buffer_bytes": hidden_buffer_bytes,
            "fp16_hidden_buffer_bytes": 2 * hidden_buffer_bytes,
            "int8_weight_tile_bytes": weight_tile_bytes,
            "int32_accumulator_bytes": accumulator_bytes,
            "candidate_cache_bytes": candidate_cache_bytes,
            "total_local_state_bytes": (
                hidden_buffer_bytes
                + weight_tile_bytes
                + accumulator_bytes
                + candidate_cache_bytes
            ),
            "fp16_activation_local_state_bytes": (
                2 * hidden_buffer_bytes
                + weight_tile_bytes
                + accumulator_bytes
                + candidate_cache_bytes
            ),
            "full_int8_output_weight_bytes": (
                hardware_hidden_size * hardware_vocabulary_size
            ),
        },
        "scope": (
            "tiled-projection-and-fused-reduction-equivalence; integer-kernel-"
            "noise-arrives-from-a-separate-stream"
        ),
        "passed": matching == samples and fp16_matching == samples,
    }
