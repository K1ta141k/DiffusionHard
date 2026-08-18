import torch

from diffusion_accel.fixed_attention import (
    fixed_attention_projection_q10,
    fixed_attention_q12,
    fixed_qkv_projection_q12,
    fixed_rotary_q12,
)


def test_fixed_rotary_q12_tracks_float_reference() -> None:
    generator = torch.Generator().manual_seed(29)
    qkv = torch.randn(4, 3, 12, 64, generator=generator) * 2.0
    angles = torch.arange(4, dtype=torch.float32)[:, None] * torch.linspace(
        0.01, 0.9, 32
    )[None, :]
    cosine = angles.cos()
    sine = angles.sin()

    query, key, query_q, key_q, details = fixed_rotary_q12(
        qkv, cosine, sine
    )

    assert query_q.dtype == torch.int64
    assert key_q.dtype == torch.int64
    assert details["metrics"]["input_saturation_fraction"] == 0.0
    assert details["metrics"]["combined_numerator_bits"] == 35
    assert details["tensors"]["cosine_q15"].shape == (4, 32)
    torch.testing.assert_close(
        query,
        torch.cat(
            (
                qkv[:, 0, :, :32] * cosine[:, None]
                - qkv[:, 0, :, 32:] * sine[:, None],
                qkv[:, 0, :, 32:] * cosine[:, None]
                + qkv[:, 0, :, :32] * sine[:, None],
            ),
            dim=-1,
        ),
        rtol=0.002,
        atol=0.001,
    )


def test_fixed_attention_q12_tracks_float_reference() -> None:
    generator = torch.Generator().manual_seed(31)
    qkv = torch.randn(64, 3, 12, 64, generator=generator) * 1.25
    angles = torch.arange(64, dtype=torch.float32)[:, None] * torch.linspace(
        0.001, 0.2, 32
    )[None, :]
    attention, attention_q, details = fixed_attention_q12(
        qkv, angles.cos(), angles.sin()
    )

    assert attention_q.shape == (64, 768)
    assert details["metrics"]["score_saturation_fraction"] == 0.0
    assert details["metrics"]["value_saturation_fraction"] == 0.0
    assert details["metrics"]["exponential_lut_entries"] == 1025
    assert details["metrics"]["maximum_probability_row_sum_error_q16"] < 64
    assert details["metrics"]["attention_accuracy"]["relative_rms_error"] < 0.003


def test_fixed_attention_projection_uses_int8_weights_and_q10_output() -> None:
    generator = torch.Generator().manual_seed(37)
    attention_q12 = torch.randint(
        -12000, 12000, (4, 768), generator=generator, dtype=torch.int64
    )
    weight = torch.randn(768, 768, generator=generator) * 0.08

    projection, projection_q10, details = fixed_attention_projection_q10(
        attention_q12, weight
    )

    assert projection.shape == (4, 768)
    assert projection_q10.dtype == torch.int64
    assert details["tensors"]["weight_int8"].dtype == torch.int8
    assert details["metrics"]["required_signed_accumulator_bits"] <= 32
    assert details["metrics"]["required_unsigned_multiplier_bits"] <= 24
    assert details["metrics"]["output_saturation_fraction"] == 0.0
    assert details["metrics"]["projection_accuracy_from_fixed_attention"][
        "relative_rms_error"
    ] < 0.02


def test_fixed_qkv_projection_uses_int16_weights_and_q12_output() -> None:
    generator = torch.Generator().manual_seed(41)
    normalized_q12 = torch.randint(
        -16000, 16000, (4, 768), generator=generator, dtype=torch.int64
    )
    weight = torch.randn(2304, 768, generator=generator) * 0.04
    bias = torch.randn(2304, generator=generator) * 0.02

    qkv, qkv_q12, details = fixed_qkv_projection_q12(
        normalized_q12, weight, bias
    )

    assert qkv.shape == (4, 2304)
    assert qkv_q12.dtype == torch.int64
    assert details["tensors"]["weight_int16"].dtype == torch.int16
    assert details["metrics"]["required_signed_accumulator_bits"] <= 48
    assert details["metrics"]["required_unsigned_multiplier_bits"] <= 24
    assert details["metrics"]["output_saturation_fraction"] == 0.0
    assert details["metrics"]["projection_accuracy_from_fixed_norm"][
        "relative_rms_error"
    ] < 0.001


def test_fixed_qkv_projection_supports_int8_weight_screen() -> None:
    generator = torch.Generator().manual_seed(43)
    normalized_q12 = torch.randint(
        -16000, 16000, (2, 768), generator=generator, dtype=torch.int64
    )
    weight = torch.randn(2304, 768, generator=generator) * 0.04
    bias = torch.randn(2304, generator=generator) * 0.02

    _, qkv_q12, details = fixed_qkv_projection_q12(
        normalized_q12, weight, bias, weight_bits=8
    )

    assert qkv_q12.dtype == torch.int64
    assert details["tensors"]["weight_int8"].dtype == torch.int8
    assert details["metrics"]["weight_bits"] == 8
    assert details["metrics"]["projection_accuracy_from_fixed_norm"][
        "relative_rms_error"
    ] < 0.02


def test_fixed_qkv_projection_supports_smoothquant_equalization() -> None:
    generator = torch.Generator().manual_seed(47)
    normalized_q12 = torch.randint(
        -16000, 16000, (2, 768), generator=generator, dtype=torch.int64
    )
    weight = torch.randn(2304, 768, generator=generator) * 0.04
    bias = torch.randn(2304, generator=generator) * 0.02

    _, _, details = fixed_qkv_projection_q12(
        normalized_q12,
        weight,
        bias,
        weight_bits=8,
        smoothquant_alpha=0.5,
    )

    assert details["metrics"]["smoothquant_alpha"] == 0.5
    assert "smoothquant_input_scales" in details["tensors"]
    assert details["metrics"][
        "smoothed_activation_saturation_fraction"
    ] == 0.0
