import torch

from diffusion_accel.fixed_mlp import (
    factorize_up_requant_scales,
    hardware_gelu_q10,
    hardware_requantized_linear,
    quantize_smoothquant_interstage,
    quantize_up_activation_fixed,
    quantized_linear,
)


def test_quantized_linear_matches_exact_integer_reference() -> None:
    activation = torch.tensor([[0.25, -0.5, 0.75]])
    weight = torch.tensor([[0.5, -0.25, 0.75], [-0.5, 0.5, 0.25]])
    bias = torch.tensor([0.125, -0.25])

    output, metrics = quantized_linear(activation, weight, bias, bits=8)

    assert output.shape == (1, 2)
    assert metrics["required_signed_accumulator_bits"] <= 32
    torch.testing.assert_close(
        output,
        torch.nn.functional.linear(activation, weight, bias),
        rtol=0.02,
        atol=0.02,
    )


def test_int16_path_uses_wide_accumulation() -> None:
    activation = torch.full((2, 32), 100.0)
    weight = torch.full((3, 32), 100.0)
    bias = torch.zeros(3)

    output, metrics = quantized_linear(activation, weight, bias, bits=16)

    assert metrics["required_signed_accumulator_bits"] > 32
    torch.testing.assert_close(
        output,
        torch.full((2, 3), 320_000.0),
        rtol=1e-5,
        atol=1e-3,
    )


def test_per_token_scaling_handles_different_activation_ranges() -> None:
    activation = torch.tensor([[0.001, -0.002], [10.0, -20.0]])
    weight = torch.tensor([[0.5, -0.25]])
    bias = torch.tensor([0.0])

    tensor_output, _ = quantized_linear(
        activation, weight, bias, bits=8, activation_granularity="tensor"
    )
    token_output, metrics = quantized_linear(
        activation, weight, bias, bits=8, activation_granularity="token"
    )
    expected = torch.nn.functional.linear(activation, weight, bias)

    assert metrics["activation_granularity"] == "token"
    assert metrics["activation_scale_min"] < metrics["activation_scale_max"]
    assert torch.mean((token_output - expected).abs()) < torch.mean(
        (tensor_output - expected).abs()
    )


def test_smoothquant_transform_preserves_linear_function() -> None:
    generator = torch.Generator().manual_seed(9)
    activation = torch.randn(4, 16, generator=generator)
    weight = torch.randn(12, 16, generator=generator)
    bias = torch.randn(12, generator=generator)

    output, metrics = quantized_linear(
        activation,
        weight,
        bias,
        bits=16,
        weight_bits=16,
        activation_granularity="token",
        smoothquant_alpha=0.5,
    )

    assert metrics["smoothquant_scale_min"] > 0
    torch.testing.assert_close(
        output,
        torch.nn.functional.linear(activation, weight, bias),
        rtol=5e-4,
        atol=5e-4,
    )


def test_hardware_requantization_uses_bounded_integer_multiplier() -> None:
    activation = torch.tensor([[0.25, -0.5, 0.75]])
    weight = torch.tensor([[0.5, -0.25, 0.75], [-0.5, 0.5, 0.25]])
    bias = torch.tensor([0.125, -0.25])

    output, output_q, metrics = hardware_requantized_linear(
        activation,
        weight,
        bias,
        activation_granularity="token",
        smoothquant_alpha=0.5,
        output_width=16,
    )

    assert output_q.dtype == torch.int64
    assert metrics["required_unsigned_multiplier_bits"] <= 24
    assert metrics["output_saturation_fraction"] == 0.0
    torch.testing.assert_close(
        output,
        torch.nn.functional.linear(activation, weight, bias),
        rtol=0.02,
        atol=0.02,
    )


def test_hardware_gelu_q10_matches_frozen_points() -> None:
    input_q = torch.tensor([-9216, -1024, 0, 1024, 6144, 9216])
    _, output_q = hardware_gelu_q10(input_q)
    assert output_q.tolist() == [0, -163, 0, 861, 6144, 9216]


def test_hardware_granularity_is_reported_by_each_linear() -> None:
    activation = torch.tensor([[0.01, -0.02], [10.0, -20.0]])
    weight = torch.tensor([[0.5, -0.25]])
    bias = torch.tensor([0.0])

    _, _, token_metrics = hardware_requantized_linear(
        activation,
        weight,
        bias,
        activation_granularity="token",
        smoothquant_alpha=0.5,
        output_width=16,
    )
    _, _, tensor_metrics = hardware_requantized_linear(
        activation,
        weight,
        bias,
        activation_granularity="tensor",
        smoothquant_alpha=0.5,
        output_width=24,
    )

    assert token_metrics["activation_granularity"] == "token"
    assert tensor_metrics["activation_granularity"] == "tensor"


def test_interstage_quantizer_matches_fixed_multiplier_reference() -> None:
    gelu_q = torch.tensor([[0, 256, -174], [0, 2048, 1024]])
    down_weight = torch.tensor(
        [[0.25, -0.5, 0.125], [-0.75, 0.25, 0.5]]
    )

    output_q, tensors, metrics = quantize_smoothquant_interstage(
        gelu_q,
        down_weight,
        smoothquant_alpha=0.5,
    )

    assert output_q.dtype == torch.int8
    assert tensors["multiplier"].dtype == torch.int32
    assert tensors["multiplier"][0].item() == 0
    assert metrics["dead_channel_count"] == 1
    assert metrics["required_unsigned_multiplier_bits"] <= 24
    assert metrics["reference_mismatch_count"] == 0
    assert output_q.abs().max().item() == 127


def test_factorized_up_scales_fit_serial_hardware_widths() -> None:
    token_scales = torch.tensor([0.01, 0.02])
    output_scales = torch.tensor([0.002, 0.01, 0.05])

    multipliers, tensors, metrics = factorize_up_requant_scales(
        token_scales, output_scales
    )

    assert multipliers.shape == (2, 3)
    assert tensors["token_factor"].max().item() < (1 << 16)
    assert tensors["output_factor"].max().item() < (1 << 18)
    assert metrics["factor_shift"] == 8
    assert metrics["maximum_multiplier_relative_error"] < 0.001


def test_fixed_up_activation_quantizer_produces_bounded_int8() -> None:
    activation = torch.tensor(
        [[0.25, -0.5, 0.75], [2.0, -1.0, 0.125]]
    )
    smoothing = torch.tensor([0.5, 2.0, 0.25])

    output_q, token_factors, tensors, metrics = quantize_up_activation_fixed(
        activation, smoothing
    )

    assert output_q.dtype == torch.int8
    assert output_q.abs().max().item() == 127
    assert token_factors.dtype == torch.int32
    assert tensors["reciprocal"].max().item() < (1 << 18)
    assert metrics["input_saturation_fraction"] == 0.0
    assert metrics["int8_reference_max_abs_difference"] <= 1
