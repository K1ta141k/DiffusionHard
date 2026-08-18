"""Exact algebraic folds used by the fixed MDLM hardware graph."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .model_spec import ModelSpec


def fold_normalized_input_affine(
    weight: "object",
    bias: Optional["object"],
    norm_weight: "object",
    shift: "object",
    scale: "object",
) -> Tuple["object", "object"]:
    """Fold AdaLN scale and shift into a following linear operation.

    Inputs use PyTorch's [output, input] linear-weight convention. The function
    intentionally imports torch lazily so architecture-only commands stay light.
    """
    import torch

    combined_scale = norm_weight * (1 + scale)
    folded_weight = weight * combined_scale.unsqueeze(0)
    folded_bias = torch.mv(weight, shift)
    if bias is not None:
        folded_bias = folded_bias + bias
    return folded_weight, folded_bias


def fold_output_gate(
    weight: "object", bias: Optional["object"], gate: "object"
) -> Tuple["object", Optional["object"]]:
    """Fold a constant output gate into linear rows and its bias."""
    folded_weight = weight * gate.unsqueeze(1)
    folded_bias = None if bias is None else bias * gate
    return folded_weight, folded_bias


def specialization_inventory(spec: ModelSpec) -> Dict[str, int]:
    """Report parameters and traffic removed by constant time conditioning."""
    hidden = spec.hidden_size
    condition = spec.conditioning_size
    block_adaln = condition * 6 * hidden + 6 * hidden
    final_adaln = condition * 2 * hidden + 2 * hidden
    timestep = 256 * condition + condition + condition * condition + condition
    removed = timestep + spec.transformer_blocks * block_adaln + final_adaln
    offline_constants = spec.transformer_blocks * 6 * hidden + 2 * hidden
    folded_norm_values = spec.transformer_blocks * 2 * hidden + hidden
    new_qkv_bias_values = spec.transformer_blocks * 3 * hidden
    runtime_model_values = (
        spec.parameter_count - removed - folded_norm_values + new_qkv_bias_values
    )
    fixed_rotary_values = spec.first_hardware_canvas * spec.head_size
    embedding_rows = spec.first_hardware_canvas * hidden
    evaluation_model_values = (
        runtime_model_values - spec.vocabulary_size * hidden + embedding_rows
    )
    return {
        "constant_conditioning_parameters_removed": removed,
        "constant_values_evaluated_offline": offline_constants,
        "separate_normalization_values_folded": folded_norm_values,
        "new_qkv_bias_values": new_qkv_bias_values,
        "fixed_rotary_values": fixed_rotary_values,
        "runtime_model_values": runtime_model_values,
        "fp16_export_values_with_rotary": runtime_model_values
        + fixed_rotary_values,
        "embedding_values_read_per_evaluation_upper_bound": embedding_rows,
        "fp16_bytes_per_evaluation_after_folding": (
            evaluation_model_values + fixed_rotary_values
        )
        * 2,
        "analytical_int8_plus_fp16_rotary_bytes_per_evaluation": (
            evaluation_model_values + fixed_rotary_values * 2
        ),
    }
