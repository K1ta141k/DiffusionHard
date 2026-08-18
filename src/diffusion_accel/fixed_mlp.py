"""Bit-accurate fixed-point validation for the specialized MDLM MLP."""

from __future__ import annotations

import math
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, Optional, Tuple


def _quantize_symmetric(
    tensor: Any, bits: int, granularity: str
) -> Tuple[Any, Any]:
    import torch

    if bits not in {8, 16}:
        raise ValueError("fixed MLP validation supports 8 or 16 bits")
    limit = (1 << (bits - 1)) - 1
    if granularity == "tensor":
        maximum = tensor.abs().max()
    elif granularity == "token":
        if tensor.ndim != 2:
            raise ValueError("token activation quantization expects a matrix")
        maximum = tensor.abs().amax(dim=1)
    else:
        raise ValueError("activation granularity must be tensor or token")
    scale = torch.where(
        maximum > 0,
        maximum / float(limit),
        torch.ones_like(maximum),
    )
    divisor = scale if scale.ndim == 0 else scale[:, None]
    quantized = torch.round(tensor / divisor).clamp(-limit, limit)
    dtype = torch.int8 if bits == 8 else torch.int16
    return quantized.to(dtype), scale


def _quantize_weight_per_output(tensor: Any, bits: int) -> Tuple[Any, Any]:
    import torch

    if bits not in {8, 16}:
        raise ValueError("fixed MLP validation supports 8 or 16 bits")
    limit = (1 << (bits - 1)) - 1
    maximum = tensor.abs().amax(dim=1)
    scales = torch.where(
        maximum > 0,
        maximum / float(limit),
        torch.ones_like(maximum),
    )
    quantized = torch.round(tensor / scales[:, None]).clamp(-limit, limit)
    dtype = torch.int8 if bits == 8 else torch.int16
    return quantized.to(dtype), scales


def _integer_matmul(left: Any, right_transposed: Any, bits: int) -> Any:
    import torch

    if bits == 8 and hasattr(torch, "_int_mm"):
        return torch._int_mm(left.contiguous(), right_transposed.contiguous())
    # Float64 represents the worst-case 16-bit MDLM dot product exactly because
    # its integer accumulator remains well below the 53-bit mantissa limit.
    return torch.matmul(left.double(), right_transposed.double()).to(torch.int64)


def quantized_linear(
    activation: Any,
    weight: Any,
    bias: Any,
    bits: int,
    activation_granularity: str = "tensor",
    weight_bits: Optional[int] = None,
    smoothquant_alpha: Optional[float] = None,
) -> Tuple[Any, Dict[str, object]]:
    """Run per-tensor activation and per-output weight integer linear math."""
    import torch

    resolved_weight_bits = bits if weight_bits is None else int(weight_bits)
    smoothing_scale = None
    if smoothquant_alpha is not None:
        if not 0.0 <= smoothquant_alpha <= 1.0:
            raise ValueError("SmoothQuant alpha must be between zero and one")
        activation_max = activation.abs().amax(dim=0).clamp_min(1e-8)
        weight_max = weight.abs().amax(dim=0).clamp_min(1e-8)
        smoothing_scale = (
            activation_max.pow(smoothquant_alpha)
            / weight_max.pow(1.0 - smoothquant_alpha)
        ).clamp_min(1e-8)
        activation = activation / smoothing_scale[None, :]
        weight = weight * smoothing_scale[None, :]
    activation_q, activation_scale = _quantize_symmetric(
        activation, bits, activation_granularity
    )
    weight_q, weight_scales = _quantize_weight_per_output(
        weight, resolved_weight_bits
    )
    if activation_scale.ndim == 0:
        combined_scales = activation_scale * weight_scales[None, :]
    else:
        combined_scales = activation_scale[:, None] * weight_scales[None, :]
    accumulator = _integer_matmul(
        activation_q,
        weight_q.t(),
        8 if bits == 8 and resolved_weight_bits == 8 else 16,
    )
    accumulator = accumulator.to(torch.int64)
    output = (
        accumulator.double() * combined_scales.double() + bias.double()[None, :]
    )
    activation_limit = (1 << (bits - 1)) - 1
    weight_limit = (1 << (resolved_weight_bits - 1)) - 1
    metrics: Dict[str, object] = {
        "activation_bits": bits,
        "weight_bits": resolved_weight_bits,
        "smoothquant_alpha": smoothquant_alpha,
        "activation_granularity": activation_granularity,
        "activation_scale_min": float(activation_scale.min().item()),
        "activation_scale_max": float(activation_scale.max().item()),
        "weight_scale_min": float(weight_scales.min().item()),
        "weight_scale_max": float(weight_scales.max().item()),
        "activation_saturation_fraction": float(
            activation_q.abs().eq(activation_limit).float().mean().item()
        ),
        "weight_saturation_fraction": float(
            weight_q.abs().eq(weight_limit).float().mean().item()
        ),
        "accumulator_min": int(accumulator.min().item()),
        "accumulator_max": int(accumulator.max().item()),
        "required_signed_accumulator_bits": max(
            1,
            int(accumulator.abs().max().item()).bit_length() + 1,
        ),
    }
    if smoothing_scale is not None:
        metrics["smoothquant_scale_min"] = float(smoothing_scale.min().item())
        metrics["smoothquant_scale_max"] = float(smoothing_scale.max().item())
    return output.float(), metrics


def hardware_requantized_linear(
    activation: Any,
    weight: Any,
    bias: Any,
    *,
    activation_granularity: str,
    smoothquant_alpha: float,
    output_width: int,
    output_fraction_bits: int = 10,
    right_shift: int = 20,
    multiplier_bits: int = 24,
) -> Tuple[Any, Any, Dict[str, object]]:
    """Simulate W8A8 MACs and the RTL integer requantization exactly."""
    import torch

    if not 0.0 <= smoothquant_alpha <= 1.0:
        raise ValueError("SmoothQuant alpha must be between zero and one")
    if output_width <= output_fraction_bits + 1:
        raise ValueError("output width must include sign and integer bits")
    if right_shift <= 0 or multiplier_bits <= 0:
        raise ValueError("right shift and multiplier bits must be positive")
    activation_max = activation.abs().amax(dim=0).clamp_min(1e-8)
    weight_max = weight.abs().amax(dim=0).clamp_min(1e-8)
    smoothing_scale = (
        activation_max.pow(smoothquant_alpha)
        / weight_max.pow(1.0 - smoothquant_alpha)
    ).clamp_min(1e-8)
    transformed_activation = activation / smoothing_scale[None, :]
    transformed_weight = weight * smoothing_scale[None, :]
    activation_q, activation_scale = _quantize_symmetric(
        transformed_activation, 8, activation_granularity
    )
    weight_q, weight_scales = _quantize_weight_per_output(
        transformed_weight, 8
    )
    if activation_scale.ndim == 0:
        combined_scales = activation_scale * weight_scales[None, :]
    else:
        combined_scales = activation_scale[:, None] * weight_scales[None, :]
    accumulators = _integer_matmul(
        activation_q, weight_q.t(), 8
    ).to(torch.int64)

    multiplier_scale = 1 << (right_shift + output_fraction_bits)
    multipliers = torch.round(
        combined_scales.double() * multiplier_scale
    ).to(torch.int64)
    multiplier_limit = (1 << multiplier_bits) - 1
    if int(multipliers.max().item()) > multiplier_limit:
        raise ValueError("requant multiplier exceeds configured unsigned width")
    products = accumulators * multipliers
    rounding_offset = 1 << (right_shift - 1)
    rounded = torch.sign(products) * (
        (products.abs() + rounding_offset) >> right_shift
    )
    bias_q = torch.round(
        bias.double() * (1 << output_fraction_bits)
    ).to(torch.int64)
    biased = rounded + bias_q[None, :]
    output_min = -(1 << (output_width - 1))
    output_max = (1 << (output_width - 1)) - 1
    output_q = biased.clamp(output_min, output_max)
    output = output_q.double() / float(1 << output_fraction_bits)
    return output.float(), output_q, {
        "activation_granularity": activation_granularity,
        "output_width": output_width,
        "output_fraction_bits": output_fraction_bits,
        "right_shift": right_shift,
        "multiplier_bits": multiplier_bits,
        "multiplier_min": int(multipliers.min().item()),
        "multiplier_max": int(multipliers.max().item()),
        "required_unsigned_multiplier_bits": max(
            1, int(multipliers.max().item()).bit_length()
        ),
        "output_saturation_fraction": float(
            ((biased < output_min) | (biased > output_max)).float().mean().item()
        ),
        "activation_scale_min": float(activation_scale.min().item()),
        "activation_scale_max": float(activation_scale.max().item()),
        "weight_scale_min": float(weight_scales.min().item()),
        "weight_scale_max": float(weight_scales.max().item()),
        "smoothquant_scale_min": float(smoothing_scale.min().item()),
        "smoothquant_scale_max": float(smoothing_scale.max().item()),
        "accumulator_min": int(accumulators.min().item()),
        "accumulator_max": int(accumulators.max().item()),
    }


def hardware_gelu_q10(input_q: Any) -> Tuple[Any, Any]:
    """Apply the RTL Q5.10 GELU lookup and bypass rules."""
    import torch

    coefficient = math.sqrt(2.0 / math.pi)
    lookup = []
    for address in range(1024):
        input_value = (-8192 + (address << 4)) / 1024.0
        gelu_value = 0.5 * input_value * (
            1.0
            + math.tanh(
                coefficient * (input_value + 0.044715 * input_value**3)
            )
        )
        lookup.append(round(gelu_value * 1024.0))
    lookup_tensor = torch.tensor(lookup, dtype=torch.int64, device=input_q.device)
    addresses = ((input_q + 8192) >> 4).clamp(0, 1023)
    output_q = lookup_tensor[addresses]
    output_q = torch.where(input_q <= -8192, torch.zeros_like(output_q), output_q)
    output_q = torch.where(input_q >= 8192, input_q, output_q)
    return (output_q.float() / 1024.0), output_q


def quantize_smoothquant_interstage(
    gelu_q: Any,
    down_weight: Any,
    *,
    smoothquant_alpha: float,
    right_shift: int = 20,
    multiplier_bits: int = 24,
) -> Tuple[Any, Dict[str, Any], Dict[str, object]]:
    """Convert Q5.10 GELU values to streamable SmoothQuant INT8 values."""
    import torch

    if gelu_q.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("interstage quantization expects two matrices")
    if gelu_q.shape[1] != down_weight.shape[1]:
        raise ValueError("GELU and down weight input dimensions must match")
    if not 0.0 <= smoothquant_alpha <= 1.0:
        raise ValueError("SmoothQuant alpha must be between zero and one")
    if right_shift <= 0 or multiplier_bits <= 0:
        raise ValueError("right shift and multiplier bits must be positive")

    gelu = gelu_q.float() / 1024.0
    activation_max = gelu.abs().amax(dim=0)
    weight_max = down_weight.float().abs().amax(dim=0).clamp_min(1e-8)
    smoothing_scale = (
        activation_max.clamp_min(1e-8).pow(smoothquant_alpha)
        / weight_max.pow(1.0 - smoothquant_alpha)
    ).clamp_min(1e-8)
    transformed_activation = gelu / smoothing_scale[None, :]
    activation_scale = transformed_activation.abs().max() / 127.0
    if float(activation_scale.item()) == 0.0:
        activation_scale = torch.ones_like(activation_scale)

    reference_q = torch.round(
        transformed_activation / activation_scale
    ).clamp(-127, 127).to(torch.int64)
    multipliers = torch.round(
        (1.0 / (1024.0 * smoothing_scale.double() * activation_scale.double()))
        * (1 << right_shift)
    ).to(torch.int64)
    dead_channels = activation_max == 0
    multipliers[dead_channels] = 0
    if int(multipliers.max().item()) >= (1 << multiplier_bits):
        raise ValueError("live interstage multiplier exceeds configured width")

    products = gelu_q.to(torch.int64) * multipliers[None, :]
    rounded = torch.sign(products) * (
        (products.abs() + (1 << (right_shift - 1))) >> right_shift
    )
    hardware_q = rounded.clamp(-127, 127).to(torch.int8)
    mismatch = hardware_q.to(torch.int64) != reference_q
    tensors = {
        "multiplier": multipliers.to(torch.int32).contiguous(),
        "smoothing_scale": smoothing_scale.float().contiguous(),
        "activation_scale": activation_scale.float().reshape(1).contiguous(),
    }
    metrics = {
        "right_shift": right_shift,
        "multiplier_bits": multiplier_bits,
        "multiplier_min": int(multipliers.min().item()),
        "multiplier_max": int(multipliers.max().item()),
        "required_unsigned_multiplier_bits": max(
            1, int(multipliers.max().item()).bit_length()
        ),
        "dead_channel_count": int(dead_channels.sum().item()),
        "reference_mismatch_count": int(mismatch.sum().item()),
        "reference_mismatch_fraction": float(mismatch.float().mean().item()),
        "saturation_count": int(hardware_q.abs().eq(127).sum().item()),
        "saturation_fraction": float(
            hardware_q.abs().eq(127).float().mean().item()
        ),
        "activation_scale": float(activation_scale.item()),
        "smoothing_scale_min": float(smoothing_scale.min().item()),
        "smoothing_scale_max": float(smoothing_scale.max().item()),
    }
    return hardware_q, tensors, metrics


def factorize_up_requant_scales(
    token_scales: Any,
    output_scales: Any,
    *,
    token_fraction_bits: int = 18,
    output_fraction_bits: int = 20,
    combined_fraction_bits: int = 30,
    token_factor_bits: int = 16,
    output_factor_bits: int = 18,
    multiplier_bits: int = 24,
) -> Tuple[Any, Dict[str, Any], Dict[str, object]]:
    """Factor token-by-output requant multipliers into two small vectors."""
    import torch

    factor_shift = (
        token_fraction_bits
        + output_fraction_bits
        - combined_fraction_bits
    )
    if factor_shift <= 0:
        raise ValueError("factor fractions must exceed combined fraction bits")
    token_factors = torch.round(
        token_scales.double() * (1 << token_fraction_bits)
    ).to(torch.int64)
    output_factors = torch.round(
        output_scales.double() * (1 << output_fraction_bits)
    ).to(torch.int64)
    if int(token_factors.max().item()) >= (1 << token_factor_bits):
        raise ValueError("token factor exceeds configured width")
    if int(output_factors.max().item()) >= (1 << output_factor_bits):
        raise ValueError("output factor exceeds configured width")
    products = token_factors[:, None] * output_factors[None, :]
    multipliers = (products + (1 << (factor_shift - 1))) >> factor_shift
    if int(multipliers.max().item()) >= (1 << multiplier_bits):
        raise ValueError("factorized multiplier exceeds configured width")
    reference = torch.round(
        token_scales.double()[:, None]
        * output_scales.double()[None, :]
        * (1 << combined_fraction_bits)
    ).to(torch.int64)
    relative_error = (
        (multipliers - reference).abs().double()
        / reference.clamp_min(1).double()
    )
    tensors = {
        "token_factor": token_factors.to(torch.int32).contiguous(),
        "output_factor": output_factors.to(torch.int32).contiguous(),
    }
    metrics = {
        "token_fraction_bits": token_fraction_bits,
        "output_fraction_bits": output_fraction_bits,
        "combined_fraction_bits": combined_fraction_bits,
        "factor_shift": factor_shift,
        "token_factor_bits": token_factor_bits,
        "output_factor_bits": output_factor_bits,
        "token_factor_min": int(token_factors.min().item()),
        "token_factor_max": int(token_factors.max().item()),
        "output_factor_min": int(output_factors.min().item()),
        "output_factor_max": int(output_factors.max().item()),
        "maximum_multiplier_relative_error": float(relative_error.max().item()),
        "mean_multiplier_relative_error": float(relative_error.mean().item()),
        "exact_multiplier_fraction": float(
            multipliers.eq(reference).float().mean().item()
        ),
    }
    return multipliers, tensors, metrics


def quantize_up_activation_fixed(
    activation: Any,
    smoothing_scale: Any,
    *,
    input_width: int = 18,
    input_fraction_bits: int = 12,
    reciprocal_fraction_bits: int = 15,
    quantizer_shift: int = 18,
) -> Tuple[Any, Any, Dict[str, Any], Dict[str, object]]:
    """Model the two-pass runtime activation quantizer exactly."""
    import torch

    input_min = -(1 << (input_width - 1))
    input_max = (1 << (input_width - 1)) - 1
    activation_unclipped = torch.round(
        activation.double() * (1 << input_fraction_bits)
    ).to(torch.int64)
    activation_q = activation_unclipped.clamp(input_min, input_max)
    reciprocal = torch.round(
        smoothing_scale.double().reciprocal()
        * (1 << reciprocal_fraction_bits)
    ).to(torch.int64)
    if int(reciprocal.max().item()).bit_length() > 18:
        raise ValueError("SmoothQuant reciprocal exceeds unsigned 18 bits")
    products = activation_q * reciprocal[None, :]
    transformed_q = torch.sign(products) * (
        (products.abs() + (1 << (reciprocal_fraction_bits - 1)))
        >> reciprocal_fraction_bits
    )
    transformed_q = transformed_q.clamp(input_min, input_max)
    maxima = transformed_q.abs().amax(dim=1)
    quantizer_numerator = 127 << quantizer_shift
    quantizer_multipliers = torch.where(
        maxima > 0,
        (quantizer_numerator + (maxima // 2)) // maxima,
        torch.zeros_like(maxima),
    )
    quantized_products = transformed_q * quantizer_multipliers[:, None]
    quantized = torch.sign(quantized_products) * (
        (quantized_products.abs() + (1 << (quantizer_shift - 1)))
        >> quantizer_shift
    )
    quantized = quantized.clamp(-127, 127).to(torch.int8)
    token_factor_numerator = maxima << (18 - input_fraction_bits)
    token_factors = torch.where(
        maxima > 0,
        (token_factor_numerator + 63) // 127,
        torch.zeros_like(maxima),
    ).to(torch.int32)

    transformed = activation / smoothing_scale[None, :]
    reference_q, reference_scale = _quantize_symmetric(
        transformed, 8, "token"
    )
    reference_token_factors = torch.round(
        reference_scale.double() * (1 << 18)
    ).to(torch.int64)
    transformed_fixed = transformed_q.double() / (1 << input_fraction_bits)
    metrics = {
        "input_format": "signed-q5.12-18-bit",
        "reciprocal_format": "unsigned-q3.15-18-bit",
        "quantizer_multiplier_bits_required": max(
            1, int(quantizer_multipliers.max().item()).bit_length()
        ),
        "reciprocal_bits_required": max(
            1, int(reciprocal.max().item()).bit_length()
        ),
        "input_saturation_fraction": float(
            (
                (activation_unclipped < input_min)
                | (activation_unclipped > input_max)
            ).float().mean().item()
        ),
        "int8_reference_mismatch_fraction": float(
            quantized.ne(reference_q).float().mean().item()
        ),
        "int8_reference_max_abs_difference": int(
            (quantized.to(torch.int16) - reference_q.to(torch.int16))
            .abs()
            .max()
            .item()
        ),
        "transformed_relative_rms_error": float(
            (
                torch.sqrt(torch.mean((transformed_fixed - transformed.double()) ** 2))
                / torch.sqrt(torch.mean(transformed.double() ** 2))
            ).item()
        ),
        "token_factor_maximum_relative_error": float(
            (
                (token_factors.to(torch.int64) - reference_token_factors)
                .abs()
                .double()
                / reference_token_factors.clamp_min(1).double()
            ).max().item()
        ),
    }
    tensors = {
        "reciprocal": reciprocal.to(torch.int32).contiguous(),
        "quantizer_multiplier": quantizer_multipliers.to(torch.int32).contiguous(),
        "transformed_q12": transformed_q.to(torch.int32).contiguous(),
    }
    return quantized, token_factors, tensors, metrics


def _error_metrics(actual: Any, expected: Any) -> Dict[str, float]:
    import torch
    import torch.nn.functional as functional

    difference = actual.float() - expected.float()
    expected_rms = torch.sqrt(torch.mean(expected.float().square()))
    actual_flat = actual.float().reshape(1, -1)
    expected_flat = expected.float().reshape(1, -1)
    cosine = float(functional.cosine_similarity(actual_flat, expected_flat).item())
    return {
        "max_abs_error": float(difference.abs().max().item()),
        "mean_abs_error": float(difference.abs().mean().item()),
        "rms_error": float(torch.sqrt(torch.mean(difference.square())).item()),
        "relative_rms_error": float(
            (torch.sqrt(torch.mean(difference.square())) / expected_rms).item()
        ),
        "cosine_similarity": max(-1.0, min(1.0, cosine)),
    }


def _load_tensors(path: Path, names: Iterable[str]) -> Dict[str, Any]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name).float() for name in names}


def export_mlp_interstage_package(
    package_dir: Path,
    out_path: Path,
    *,
    right_shift: int = 20,
    multiplier_bits: int = 24,
) -> Dict[str, object]:
    """Export fixed constants needed between the MDLM MLP up and down paths."""
    import hashlib
    import torch
    from safetensors.torch import save_file

    alphas = (0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.5, 0.5, 0.75)
    artifact: Dict[str, Any] = {}
    blocks = []
    for block, alpha in enumerate(alphas):
        golden_prefix = "folded.block_%02d" % block
        weight_prefix = "block_%02d" % block
        goldens = _load_tensors(
            package_dir / "golden_tensors.safetensors",
            [
                golden_prefix + ".norm2_unaffine",
                golden_prefix + ".mlp_up",
                golden_prefix + ".mlp_down",
                golden_prefix + ".after_attention",
                golden_prefix + ".output",
            ],
        )
        weights = _load_tensors(
            package_dir / "folded_fp16_weights.safetensors",
            [
                weight_prefix + ".mlp_up.weight",
                weight_prefix + ".mlp_up.bias",
                weight_prefix + ".mlp_down.weight",
                weight_prefix + ".mlp_down.bias",
            ],
        )
        normalized = goldens[golden_prefix + ".norm2_unaffine"][0]
        up_weight = weights[weight_prefix + ".mlp_up.weight"].float()
        up_bias = weights[weight_prefix + ".mlp_up.bias"].float()
        up_smoothing = (
            normalized.abs().amax(dim=0).clamp_min(1e-8).pow(alpha)
            / up_weight.abs().amax(dim=0).clamp_min(1e-8).pow(1.0 - alpha)
        ).clamp_min(1e-8)
        _, up_token_scales = _quantize_symmetric(
            normalized / up_smoothing[None, :], 8, "token"
        )
        (
            up_activation_q,
            fixed_up_token_factors,
            up_activation_tensors,
            up_activation_metrics,
        ) = quantize_up_activation_fixed(
            normalized,
            up_smoothing,
        )
        up_weight_q, up_output_scales = _quantize_weight_per_output(
            up_weight * up_smoothing[None, :], 8
        )
        up_accumulators = _integer_matmul(
            up_activation_q, up_weight_q.t(), 8
        ).to(torch.int64)
        _, up_factors, up_factor_metrics = (
            factorize_up_requant_scales(
                up_token_scales,
                up_output_scales,
                multiplier_bits=multiplier_bits,
            )
        )
        up_multipliers = (
            fixed_up_token_factors.to(torch.int64)[:, None]
            * up_factors["output_factor"].to(torch.int64)[None, :]
            + 128
        ) >> 8
        direct_up_multipliers = torch.round(
            up_token_scales.double()[:, None]
            * up_output_scales.double()[None, :]
            * (1 << 30)
        ).to(torch.int64)
        effective_relative_error = (
            (up_multipliers - direct_up_multipliers).abs().double()
            / direct_up_multipliers.clamp_min(1).double()
        )
        up_factor_metrics["effective_maximum_multiplier_relative_error"] = (
            float(effective_relative_error.max().item())
        )
        up_factor_metrics["effective_mean_multiplier_relative_error"] = (
            float(effective_relative_error.mean().item())
        )
        up_products = up_accumulators * up_multipliers
        up_q = torch.sign(up_products) * (
            (up_products.abs() + (1 << (right_shift - 1))) >> right_shift
        )
        up_bias_q = torch.round(up_bias.double() * 1024.0).to(torch.int64)
        up_q = (up_q + up_bias_q[None, :]).clamp(-(1 << 15), (1 << 15)-1)
        up = up_q.float() / 1024.0
        _, gelu_q = hardware_gelu_q10(up_q)
        down_weight = weights[weight_prefix + ".mlp_down.weight"].float()
        down_bias = weights[weight_prefix + ".mlp_down.bias"].float()
        activation_q, interstage, interstage_metrics = (
            quantize_smoothquant_interstage(
                gelu_q,
                down_weight,
                smoothquant_alpha=alpha,
                right_shift=right_shift,
                multiplier_bits=multiplier_bits,
            )
        )
        transformed_weight = (
            down_weight * interstage["smoothing_scale"][None, :]
        )
        weight_q, weight_scales = _quantize_weight_per_output(
            transformed_weight, 8
        )
        accumulators = _integer_matmul(activation_q, weight_q.t(), 8).to(
            torch.int64
        )
        output_multipliers = torch.round(
            interstage["activation_scale"].double()
            * weight_scales.double()
            * float(1 << (right_shift + 10))
        ).to(torch.int64)
        if int(output_multipliers.max().item()) >= (1 << multiplier_bits):
            raise ValueError("down output multiplier exceeds configured width")
        products = accumulators * output_multipliers[None, :]
        down_q = torch.sign(products) * (
            (products.abs() + (1 << (right_shift - 1))) >> right_shift
        )
        bias_q = torch.round(down_bias.double() * 1024.0).to(torch.int64)
        down_q = (down_q + bias_q[None, :]).clamp(
            -(1 << 23), (1 << 23) - 1
        )
        down = down_q.float() / 1024.0
        residual_q = torch.round(
            goldens[golden_prefix + ".after_attention"][0].double() * 1024.0
        ).clamp(-(1 << 23), (1 << 23) - 1)
        block_q = (residual_q + down_q).clamp(
            -(1 << 23), (1 << 23) - 1
        )

        artifact_prefix = "block_%02d" % block
        artifact[artifact_prefix + ".up_smoothing_reciprocal_q15"] = (
            up_activation_tensors["reciprocal"]
        )
        artifact[artifact_prefix + ".h0_up_token_factor"] = (
            fixed_up_token_factors.contiguous()
        )
        artifact[artifact_prefix + ".h0_up_quantizer_multiplier"] = (
            up_activation_tensors["quantizer_multiplier"]
        )
        artifact[artifact_prefix + ".up_output_factor"] = up_factors[
            "output_factor"
        ]
        artifact[artifact_prefix + ".up_bias_q10"] = (
            up_bias_q.to(torch.int32).contiguous()
        )
        artifact[artifact_prefix + ".interstage_multiplier"] = interstage[
            "multiplier"
        ]
        artifact[artifact_prefix + ".down_weight"] = weight_q.contiguous()
        artifact[artifact_prefix + ".down_output_multiplier"] = (
            output_multipliers.to(torch.int32).contiguous()
        )
        artifact[artifact_prefix + ".down_bias_q10"] = (
            bias_q.to(torch.int32).contiguous()
        )
        blocks.append(
            {
                "block": block,
                "smoothquant_alpha": alpha,
                "up_factorization": up_factor_metrics,
                "up_activation_quantizer": up_activation_metrics,
                "up_accuracy": _error_metrics(
                    up, goldens[golden_prefix + ".mlp_up"][0]
                ),
                "interstage": interstage_metrics,
                "down_output_multiplier_min": int(
                    output_multipliers.min().item()
                ),
                "down_output_multiplier_max": int(
                    output_multipliers.max().item()
                ),
                "down_output_required_multiplier_bits": max(
                    1, int(output_multipliers.max().item()).bit_length()
                ),
                "down_accuracy": _error_metrics(
                    down, goldens[golden_prefix + ".mlp_down"][0]
                ),
                "block_output_accuracy": _error_metrics(
                    block_q.float() / 1024.0,
                    goldens[golden_prefix + ".output"][0],
                ),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        artifact,
        str(out_path),
        metadata={
            "model": "kuleshov-group/mdlm-owt",
            "purpose": "fixed factorized MLP up and GELU-to-down constants",
            "interstage_format": "signed-int8-symmetric-127",
            "right_shift": str(right_shift),
        },
    )
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    block_errors = [
        block["block_output_accuracy"]["relative_rms_error"]
        for block in blocks
    ]
    return {
        "source_package_dir": str(package_dir),
        "artifact": {
            "path": str(out_path),
            "sha256": digest,
            "bytes": out_path.stat().st_size,
            "tensor_count": len(artifact),
        },
        "formats": {
            "up_token_factor": "unsigned-q18-16-bit-runtime",
            "up_output_factor": "unsigned-q20-18-bit-static",
            "up_normalized_input": "signed-q5.12-18-bit-runtime",
            "up_smoothing_reciprocal": "unsigned-q3.15-18-bit-static",
            "up_factor_product_shift": 8,
            "gelu_input": "signed-q5.10-16-bit",
            "down_activation": "signed-int8-symmetric-127",
            "interstage_multiplier": "unsigned-24-bit",
            "down_weight": "signed-int8-per-output-scale",
            "down_output": "signed-q13.10-24-bit",
            "right_shift": right_shift,
        },
        "blocks": blocks,
        "summary": {
            "maximum_up_factorized_multiplier_relative_error": max(
                block["up_factorization"][
                    "maximum_multiplier_relative_error"
                ]
                for block in blocks
            ),
            "maximum_effective_up_multiplier_relative_error": max(
                block["up_factorization"][
                    "effective_maximum_multiplier_relative_error"
                ]
                for block in blocks
            ),
            "maximum_fixed_up_activation_int8_mismatch_fraction": max(
                block["up_activation_quantizer"][
                    "int8_reference_mismatch_fraction"
                ]
                for block in blocks
            ),
            "maximum_fixed_up_activation_transformed_relative_rms": max(
                block["up_activation_quantizer"][
                    "transformed_relative_rms_error"
                ]
                for block in blocks
            ),
            "maximum_interstage_multiplier_bits": max(
                block["interstage"]["required_unsigned_multiplier_bits"]
                for block in blocks
            ),
            "total_dead_channels": sum(
                block["interstage"]["dead_channel_count"] for block in blocks
            ),
            "total_interstage_reference_mismatches": sum(
                block["interstage"]["reference_mismatch_count"]
                for block in blocks
            ),
            "maximum_block_output_relative_rms": max(block_errors),
            "mean_block_output_relative_rms": statistics.mean(block_errors),
        },
        "scope": (
            "constants calibrated on the single frozen H0 trace; broader "
            "calibration and generation-level quality remain open"
        ),
    }


def validate_fixed_mlp(
    package_dir: Path,
    *,
    block: int = 0,
    bits: int = 8,
    weight_bits: Optional[int] = None,
    activation_granularity: str = "tensor",
    smoothquant_alpha: Optional[float] = None,
    mac_lanes: int = 1024,
    clock_mhz: float = 250.0,
) -> Dict[str, object]:
    """Validate one real folded checkpoint MLP using integer arithmetic."""
    import torch
    import torch.nn.functional as functional

    if not 0 <= block < 12:
        raise ValueError("block must be between 0 and 11")
    if mac_lanes <= 0 or clock_mhz <= 0:
        raise ValueError("MAC lanes and clock must be positive")
    resolved_weight_bits = bits if weight_bits is None else int(weight_bits)
    prefix = "folded.block_%02d" % block
    golden_names = [
        prefix + ".norm2_unaffine",
        prefix + ".mlp_up",
        prefix + ".gelu",
        prefix + ".mlp_down",
        prefix + ".after_attention",
        prefix + ".output",
    ]
    weight_prefix = "block_%02d" % block
    weight_names = [
        weight_prefix + ".mlp_up.weight",
        weight_prefix + ".mlp_up.bias",
        weight_prefix + ".mlp_down.weight",
        weight_prefix + ".mlp_down.bias",
    ]
    goldens = _load_tensors(
        package_dir / "golden_tensors.safetensors", golden_names
    )
    weights = _load_tensors(
        package_dir / "folded_fp16_weights.safetensors", weight_names
    )

    normalized = goldens[prefix + ".norm2_unaffine"][0]
    up, up_integer = quantized_linear(
        normalized,
        weights[weight_prefix + ".mlp_up.weight"],
        weights[weight_prefix + ".mlp_up.bias"],
        bits,
        activation_granularity,
        resolved_weight_bits,
        smoothquant_alpha,
    )
    gelu = functional.gelu(up, approximate="tanh")
    down, down_integer = quantized_linear(
        gelu,
        weights[weight_prefix + ".mlp_down.weight"],
        weights[weight_prefix + ".mlp_down.bias"],
        bits,
        activation_granularity,
        resolved_weight_bits,
        smoothquant_alpha,
    )
    block_output = goldens[prefix + ".after_attention"][0] + down

    tokens = int(normalized.shape[0])
    hidden = int(normalized.shape[1])
    intermediate = int(up.shape[1])
    up_macs = tokens * hidden * intermediate
    down_macs = tokens * intermediate * hidden
    total_macs = up_macs + down_macs
    ideal_cycles = math.ceil(total_macs / mac_lanes)
    ideal_latency_ms = ideal_cycles / (clock_mhz * 1_000_000.0) * 1e3
    return {
        "package_dir": str(package_dir),
        "block": block,
        "activation_bits": bits,
        "weight_bits": resolved_weight_bits,
        "activation_granularity": activation_granularity,
        "smoothquant_alpha": smoothquant_alpha,
        "shape": {
            "tokens": tokens,
            "hidden": hidden,
            "intermediate": intermediate,
        },
        "integer": {
            "mlp_up": up_integer,
            "mlp_down": down_integer,
        },
        "accuracy": {
            "mlp_up": _error_metrics(up, goldens[prefix + ".mlp_up"][0]),
            "gelu": _error_metrics(gelu, goldens[prefix + ".gelu"][0]),
            "mlp_down": _error_metrics(
                down, goldens[prefix + ".mlp_down"][0]
            ),
            "block_output": _error_metrics(
                block_output, goldens[prefix + ".output"][0]
            ),
        },
        "hardware_model": {
            "mac_lanes": mac_lanes,
            "clock_mhz": clock_mhz,
            "mlp_up_macs": up_macs,
            "mlp_down_macs": down_macs,
            "total_macs": total_macs,
            "ideal_compute_cycles": ideal_cycles,
            "ideal_compute_latency_ms": ideal_latency_ms,
            "scope": "compute-only lower bound before tile load and pipeline overhead",
        },
    }


def validate_hardware_fixed_mlp(
    package_dir: Path,
    *,
    block: int = 0,
    activation_granularity: str = "token",
    down_activation_granularity: Optional[str] = None,
    smoothquant_alpha: float = 0.75,
) -> Dict[str, object]:
    """Validate the real MLP through RTL-equivalent requant and GELU math."""
    import torch

    if not 0 <= block < 12:
        raise ValueError("block must be between 0 and 11")
    prefix = "folded.block_%02d" % block
    weight_prefix = "block_%02d" % block
    goldens = _load_tensors(
        package_dir / "golden_tensors.safetensors",
        [
            prefix + ".norm2_unaffine",
            prefix + ".mlp_up",
            prefix + ".gelu",
            prefix + ".mlp_down",
            prefix + ".after_attention",
            prefix + ".output",
        ],
    )
    weights = _load_tensors(
        package_dir / "folded_fp16_weights.safetensors",
        [
            weight_prefix + ".mlp_up.weight",
            weight_prefix + ".mlp_up.bias",
            weight_prefix + ".mlp_down.weight",
            weight_prefix + ".mlp_down.bias",
        ],
    )
    normalized = goldens[prefix + ".norm2_unaffine"][0]
    resolved_down_granularity = (
        activation_granularity
        if down_activation_granularity is None
        else down_activation_granularity
    )
    up, up_q, up_metrics = hardware_requantized_linear(
        normalized,
        weights[weight_prefix + ".mlp_up.weight"],
        weights[weight_prefix + ".mlp_up.bias"],
        activation_granularity=activation_granularity,
        smoothquant_alpha=smoothquant_alpha,
        output_width=16,
    )
    gelu, gelu_q = hardware_gelu_q10(up_q)
    down, down_q, down_metrics = hardware_requantized_linear(
        gelu,
        weights[weight_prefix + ".mlp_down.weight"],
        weights[weight_prefix + ".mlp_down.bias"],
        activation_granularity=resolved_down_granularity,
        smoothquant_alpha=smoothquant_alpha,
        output_width=24,
    )
    residual = goldens[prefix + ".after_attention"][0]
    residual_unclipped_q = torch.round(residual.double() * 1024.0).to(
        torch.int64
    )
    q13_10_min = -(1 << 23)
    q13_10_max = (1 << 23) - 1
    residual_q = residual_unclipped_q.clamp(q13_10_min, q13_10_max)
    block_unclipped_q = residual_q + down_q
    block_output_q = block_unclipped_q.clamp(q13_10_min, q13_10_max)
    block_output = block_output_q.float() / 1024.0
    return {
        "package_dir": str(package_dir),
        "block": block,
        "activation_granularity": activation_granularity,
        "down_activation_granularity": resolved_down_granularity,
        "smoothquant_alpha": smoothquant_alpha,
        "formats": {
            "mlp_up": "signed-q5.10-16-bit",
            "gelu": "signed-q5.10-1024-entry-lut",
            "mlp_down": "signed-q13.10-24-bit",
            "residual_and_block_output": "signed-q13.10-24-bit-saturating",
        },
        "integer": {
            "mlp_up": up_metrics,
            "mlp_down": down_metrics,
            "mlp_up_q_min": int(up_q.min().item()),
            "mlp_up_q_max": int(up_q.max().item()),
            "gelu_q_min": int(gelu_q.min().item()),
            "gelu_q_max": int(gelu_q.max().item()),
            "mlp_down_q_min": int(down_q.min().item()),
            "mlp_down_q_max": int(down_q.max().item()),
            "residual_q_min": int(residual_q.min().item()),
            "residual_q_max": int(residual_q.max().item()),
            "residual_saturation_fraction": float(
                (
                    (residual_unclipped_q < q13_10_min)
                    | (residual_unclipped_q > q13_10_max)
                ).float().mean().item()
            ),
            "block_output_saturation_fraction": float(
                (
                    (block_unclipped_q < q13_10_min)
                    | (block_unclipped_q > q13_10_max)
                ).float().mean().item()
            ),
        },
        "accuracy": {
            "mlp_up": _error_metrics(up, goldens[prefix + ".mlp_up"][0]),
            "gelu": _error_metrics(gelu, goldens[prefix + ".gelu"][0]),
            "mlp_down": _error_metrics(
                down, goldens[prefix + ".mlp_down"][0]
            ),
            "block_output": _error_metrics(
                block_output, goldens[prefix + ".output"][0]
            ),
        },
        "scope": (
            "one frozen H0 input with RTL-equivalent integer requantization "
            "and GELU; generation quality and vendor timing are not established"
        ),
    }


def sweep_hardware_fixed_mlp(
    package_dir: Path,
    *,
    activation_granularity: str = "token",
    down_activation_granularity: Optional[str] = None,
) -> Dict[str, object]:
    """Run the RTL-equivalent fixed MLP reference across all 12 blocks."""
    alphas = (0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.75, 0.5, 0.5, 0.5, 0.5, 0.75)
    blocks = [
        validate_hardware_fixed_mlp(
            package_dir,
            block=block,
            activation_granularity=activation_granularity,
            down_activation_granularity=down_activation_granularity,
            smoothquant_alpha=alphas[block],
        )
        for block in range(12)
    ]
    relative_errors = [
        result["accuracy"]["block_output"]["relative_rms_error"]
        for result in blocks
    ]
    up_saturation = [
        result["integer"]["mlp_up"]["output_saturation_fraction"]
        for result in blocks
    ]
    down_saturation = [
        result["integer"]["mlp_down"]["output_saturation_fraction"]
        for result in blocks
    ]
    residual_saturation = [
        result["integer"]["residual_saturation_fraction"]
        for result in blocks
    ]
    block_output_saturation = [
        result["integer"]["block_output_saturation_fraction"]
        for result in blocks
    ]
    multiplier_bits = [
        result["integer"][layer]["required_unsigned_multiplier_bits"]
        for result in blocks
        for layer in ("mlp_up", "mlp_down")
    ]
    return {
        "package_dir": str(package_dir),
        "activation_granularity": activation_granularity,
        "down_activation_granularity": (
            activation_granularity
            if down_activation_granularity is None
            else down_activation_granularity
        ),
        "blocks": blocks,
        "summary": {
            "minimum_block_output_relative_rms": min(relative_errors),
            "median_block_output_relative_rms": statistics.median(relative_errors),
            "maximum_block_output_relative_rms": max(relative_errors),
            "mean_block_output_relative_rms": statistics.mean(relative_errors),
            "maximum_up_q5_10_saturation_fraction": max(up_saturation),
            "maximum_down_q13_10_saturation_fraction": max(down_saturation),
            "maximum_residual_q13_10_saturation_fraction": max(
                residual_saturation
            ),
            "maximum_block_output_q13_10_saturation_fraction": max(
                block_output_saturation
            ),
            "maximum_required_unsigned_multiplier_bits": max(multiplier_bits),
        },
        "scope": (
            "all checkpoint blocks on one frozen H0 input; errors are local "
            "block comparisons and not composed generation quality"
        ),
    }


def sweep_fixed_mlp(
    package_dir: Path,
    *,
    block: int = 0,
    mac_lanes: int = 1024,
    clock_mhz: float = 250.0,
) -> Dict[str, object]:
    """Run the frozen H1 precision and SmoothQuant design-point sweep."""
    designs = [
        (8, 8, "tensor", None),
        (8, 8, "token", None),
        (8, 8, "token", 0.0),
        (8, 8, "token", 0.25),
        (8, 8, "token", 0.5),
        (8, 8, "token", 0.75),
        (8, 8, "token", 1.0),
        (8, 16, "token", None),
        (16, 8, "token", None),
        (16, 16, "token", None),
    ]
    results = []
    for activation_bits, weight_bits, granularity, alpha in designs:
        result = validate_fixed_mlp(
            package_dir,
            block=block,
            bits=activation_bits,
            weight_bits=weight_bits,
            activation_granularity=granularity,
            smoothquant_alpha=alpha,
            mac_lanes=mac_lanes,
            clock_mhz=clock_mhz,
        )
        results.append(result)
    int8_results = [
        result
        for result in results
        if result["activation_bits"] == 8 and result["weight_bits"] == 8
    ]
    best_int8 = min(
        int8_results,
        key=lambda result: result["accuracy"]["block_output"][
            "relative_rms_error"
        ],
    )
    return {
        "package_dir": str(package_dir),
        "block": block,
        "design_points": results,
        "best_w8a8": {
            "activation_granularity": best_int8["activation_granularity"],
            "smoothquant_alpha": best_int8["smoothquant_alpha"],
            "accuracy": best_int8["accuracy"],
            "integer": best_int8["integer"],
        },
        "scope": (
            "single real checkpoint block and one frozen input; full-model "
            "quality is not established"
        ),
    }


def optimize_fixed_mlp_alphas(
    package_dir: Path,
    *,
    alphas: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    mac_lanes: int = 1024,
    clock_mhz: float = 250.0,
) -> Dict[str, object]:
    """Choose a baked W8A8 SmoothQuant alpha for every checkpoint block."""
    alpha_values = tuple(float(alpha) for alpha in alphas)
    if not alpha_values:
        raise ValueError("at least one SmoothQuant alpha is required")
    blocks = []
    for block in range(12):
        candidates = []
        for alpha in alpha_values:
            result = validate_fixed_mlp(
                package_dir,
                block=block,
                bits=8,
                weight_bits=8,
                activation_granularity="token",
                smoothquant_alpha=alpha,
                mac_lanes=mac_lanes,
                clock_mhz=clock_mhz,
            )
            candidates.append(
                {
                    "alpha": alpha,
                    "block_output": result["accuracy"]["block_output"],
                    "mlp_up": result["accuracy"]["mlp_up"],
                    "mlp_down": result["accuracy"]["mlp_down"],
                    "integer": result["integer"],
                }
            )
        best = min(
            candidates,
            key=lambda candidate: candidate["block_output"][
                "relative_rms_error"
            ],
        )
        blocks.append(
            {
                "block": block,
                "best_alpha": best["alpha"],
                "best_block_output": best["block_output"],
                "candidates": candidates,
            }
        )
    errors = [
        block["best_block_output"]["relative_rms_error"] for block in blocks
    ]
    return {
        "package_dir": str(package_dir),
        "precision": "W8A8",
        "activation_granularity": "token",
        "searched_alphas": list(alpha_values),
        "blocks": blocks,
        "aggregate_best_block_output_relative_rms": {
            "minimum": min(errors),
            "median": statistics.median(errors),
            "maximum": max(errors),
            "mean": statistics.mean(errors),
        },
        "scope": (
            "one frozen H0 input per block; generation-level quality remains open"
        ),
    }
