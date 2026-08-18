from pathlib import Path

import torch
import torch.nn.functional as functional

from diffusion_accel.model_spec import load_model_spec
from diffusion_accel.model_specialization import (
    fold_normalized_input_affine,
    fold_output_gate,
    specialization_inventory,
)


ROOT = Path(__file__).resolve().parents[1]


def test_constant_adaln_folds_match_original_algebra() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(5, 8, generator=generator)
    weight = torch.randn(12, 8, generator=generator)
    bias = torch.randn(12, generator=generator)
    norm_weight = torch.randn(8, generator=generator)
    shift = torch.randn(8, generator=generator)
    scale = torch.randn(8, generator=generator)
    gate = torch.randn(12, generator=generator)

    original = functional.linear(
        x * norm_weight * (1 + scale) + shift, weight, bias
    )
    folded_weight, folded_bias = fold_normalized_input_affine(
        weight, bias, norm_weight, shift, scale
    )
    folded = functional.linear(x, folded_weight, folded_bias)
    torch.testing.assert_close(folded, original, rtol=1e-5, atol=1e-5)

    original_gated = original * gate
    gated_weight, gated_bias = fold_output_gate(weight, bias, gate)
    folded_gated = functional.linear(
        x * norm_weight * (1 + scale) + shift,
        gated_weight,
        gated_bias,
    )
    torch.testing.assert_close(folded_gated, original_gated)


def test_specialization_inventory_removes_constant_conditioning_path() -> None:
    spec = load_model_spec(ROOT / "configs/models/mdlm_owt_169m.yaml")
    inventory = specialization_inventory(spec)

    assert inventory["constant_conditioning_parameters_removed"] == 7_380_736
    assert inventory["constant_values_evaluated_offline"] == 56_832
    assert inventory["separate_normalization_values_folded"] == 19_200
    assert inventory["new_qkv_bias_values"] == 27_648
    assert inventory["runtime_model_values"] == 162_254_930
    assert inventory["fp16_export_values_with_rotary"] == 162_259_026
    assert inventory["fp16_bytes_per_evaluation_after_folding"] == 247_420_068
    assert (
        inventory["analytical_int8_plus_fp16_rotary_bytes_per_evaluation"]
        == 123_714_130
    )
