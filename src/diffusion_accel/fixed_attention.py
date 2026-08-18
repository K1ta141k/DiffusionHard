"""Bit-accurate fixed-point primitives for MDLM bidirectional attention."""

from __future__ import annotations

from pathlib import Path
import statistics
from typing import Any, Dict, Optional, Tuple

from .fixed_mlp import (
    _error_metrics,
    _load_tensors,
    _quantize_weight_per_output,
)
from .fixed_norm import fixed_layer_norm_q12


def _symmetric_round_shift(values: Any, shift: int) -> Any:
    import torch

    magnitude = (values.abs() + (1 << (shift - 1))) >> shift
    return torch.sign(values) * magnitude


def fixed_rotary_q12(
    qkv: Any,
    cosine: Any,
    sine: Any,
) -> Tuple[Any, Any, Any, Any, Dict[str, object]]:
    """Rotate Q and K using Q5.12 vectors and signed Q1.15 constants."""
    import torch

    if qkv.ndim == 2 and qkv.shape[1] == 2304:
        qkv = qkv.view(qkv.shape[0], 3, 12, 64)
    if qkv.ndim != 4 or tuple(qkv.shape[1:]) != (3, 12, 64):
        raise ValueError("qkv must have shape tokens by 3 by 12 by 64")
    if cosine.shape != (qkv.shape[0], 32) or sine.shape != cosine.shape:
        raise ValueError("rotary tables must have shape tokens by 32")

    vector_min = -(1 << 17)
    vector_max = (1 << 17) - 1
    qkv_unclipped = torch.round(qkv.double() * 4096.0).to(torch.int64)
    qkv_q12 = qkv_unclipped.clamp(vector_min, vector_max)
    cosine_q15 = torch.round(cosine.double() * 32768.0).to(torch.int64)
    sine_q15 = torch.round(sine.double() * 32768.0).to(torch.int64)
    cosine_q15 = cosine_q15.clamp(-(1 << 15), (1 << 15) - 1)
    sine_q15 = sine_q15.clamp(-(1 << 15), (1 << 15) - 1)

    query_q12 = qkv_q12[:, 0]
    key_q12 = qkv_q12[:, 1]

    def rotate(values: Any) -> Any:
        first = values[..., :32]
        second = values[..., 32:]
        cos = cosine_q15[:, None, :]
        sin = sine_q15[:, None, :]
        first_numerator = first * cos - second * sin
        second_numerator = second * cos + first * sin
        first_output = _symmetric_round_shift(first_numerator, 15)
        second_output = _symmetric_round_shift(second_numerator, 15)
        return torch.cat((first_output, second_output), dim=-1).clamp(
            vector_min, vector_max
        )

    query_rotated_q12 = rotate(query_q12)
    key_rotated_q12 = rotate(key_q12)
    query_rotated = query_rotated_q12.float() / 4096.0
    key_rotated = key_rotated_q12.float() / 4096.0

    cos_float = cosine.float()[:, None, :]
    sin_float = sine.float()[:, None, :]

    def rotate_float(values: Any) -> Any:
        first = values[..., :32]
        second = values[..., 32:]
        return torch.cat(
            (
                first * cos_float - second * sin_float,
                second * cos_float + first * sin_float,
            ),
            dim=-1,
        )

    query_reference = rotate_float(qkv[:, 0].float())
    key_reference = rotate_float(qkv[:, 1].float())
    saturation = (
        (qkv_unclipped[:, :2] < vector_min)
        | (qkv_unclipped[:, :2] > vector_max)
    )
    metrics: Dict[str, object] = {
        "input_and_output_format": "signed-q5.12-18-bit",
        "rotary_constant_format": "signed-q1.15-16-bit",
        "product_format": "signed-q6.27",
        "combined_numerator_bits": 35,
        "input_saturation_fraction": float(saturation.float().mean().item()),
        "query_accuracy": _error_metrics(query_rotated, query_reference),
        "key_accuracy": _error_metrics(key_rotated, key_reference),
        "query_output_q12_min": int(query_rotated_q12.min().item()),
        "query_output_q12_max": int(query_rotated_q12.max().item()),
        "key_output_q12_min": int(key_rotated_q12.min().item()),
        "key_output_q12_max": int(key_rotated_q12.max().item()),
    }
    tensors = {
        "qkv_q12": qkv_q12,
        "cosine_q15": cosine_q15,
        "sine_q15": sine_q15,
    }
    return (
        query_rotated,
        key_rotated,
        query_rotated_q12,
        key_rotated_q12,
        {"metrics": metrics, "tensors": tensors},
    )


def sweep_fixed_rotary(package_dir: Path) -> Dict[str, object]:
    """Validate fixed rotary Q/K transforms for all twelve H0 blocks."""
    qkv_names = ["folded.block_%02d.qkv" % block for block in range(12)]
    qkv_tensors = _load_tensors(
        package_dir / "golden_tensors.safetensors", qkv_names
    )
    tables = _load_tensors(
        package_dir / "folded_fp16_weights.safetensors",
        ["rotary.cos", "rotary.sin"],
    )
    cosine = tables["rotary.cos"].float()
    sine = tables["rotary.sin"].float()
    blocks = []
    for block, name in enumerate(qkv_names):
        _, _, _, _, details = fixed_rotary_q12(
            qkv_tensors[name][0], cosine, sine
        )
        blocks.append({"block": block, "metrics": details["metrics"]})
    query_errors = [
        block["metrics"]["query_accuracy"]["relative_rms_error"]
        for block in blocks
    ]
    key_errors = [
        block["metrics"]["key_accuracy"]["relative_rms_error"]
        for block in blocks
    ]
    return {
        "package_dir": str(package_dir),
        "block_count": len(blocks),
        "blocks": blocks,
        "summary": {
            "mean_query_relative_rms": statistics.mean(query_errors),
            "maximum_query_relative_rms": max(query_errors),
            "mean_key_relative_rms": statistics.mean(key_errors),
            "maximum_key_relative_rms": max(key_errors),
            "maximum_input_saturation_fraction": max(
                block["metrics"]["input_saturation_fraction"]
                for block in blocks
            ),
            "minimum_output_q12": min(
                min(
                    block["metrics"]["query_output_q12_min"],
                    block["metrics"]["key_output_q12_min"],
                )
                for block in blocks
            ),
            "maximum_output_q12": max(
                max(
                    block["metrics"]["query_output_q12_max"],
                    block["metrics"]["key_output_q12_max"],
                )
                for block in blocks
            ),
        },
        "scope": "all twelve H0 Q/K rotary boundaries on one frozen input",
    }


def fixed_attention_q12(
    qkv: Any,
    cosine: Any,
    sine: Any,
    *,
    exponential_step_bits: int = 6,
    exponential_cutoff: float = 16.0,
) -> Tuple[Any, Any, Dict[str, object]]:
    """Run fixed QK, LUT softmax, and probability-times-V attention."""
    import math
    import torch

    if exponential_step_bits < 1 or exponential_step_bits > 9:
        raise ValueError("exponential_step_bits must be between 1 and 9")
    if exponential_cutoff <= 0:
        raise ValueError("exponential_cutoff must be positive")
    if qkv.ndim == 2 and qkv.shape[1] == 2304:
        qkv = qkv.view(qkv.shape[0], 3, 12, 64)
    (
        query_rotated,
        key_rotated,
        query_q12,
        key_q12,
        rotary_details,
    ) = fixed_rotary_q12(qkv, cosine, sine)
    value_unclipped = torch.round(qkv[:, 2].double() * 4096.0).to(torch.int64)
    value_q12 = value_unclipped.clamp(-(1 << 17), (1 << 17) - 1)

    dot_products_q24 = torch.einsum(
        "thd,shd->hts", query_q12, key_q12
    )
    score_shift = 17
    scores_q10 = _symmetric_round_shift(dot_products_q24, score_shift)
    score_min = -(1 << 17)
    score_max = (1 << 17) - 1
    score_saturation = (scores_q10 < score_min) | (scores_q10 > score_max)
    scores_q10 = scores_q10.clamp(score_min, score_max)
    row_maxima = scores_q10.amax(dim=-1, keepdim=True)
    cutoff_q10 = round(exponential_cutoff * 1024.0)
    deltas_q10 = (scores_q10 - row_maxima).clamp(min=-cutoff_q10, max=0)

    lut_entries = int(exponential_cutoff * (1 << exponential_step_bits)) + 1
    exponential_lut_q16 = torch.tensor(
        [
            min(
                65535,
                round(math.exp(-index / (1 << exponential_step_bits)) * 65536),
            )
            for index in range(lut_entries)
        ],
        dtype=torch.int64,
    )
    address_shift = 10 - exponential_step_bits
    addresses = (
        (-deltas_q10 + (1 << (address_shift - 1))) >> address_shift
    ).clamp(max=lut_entries - 1)
    exponentials_q16 = exponential_lut_q16[addresses]
    exponential_sums_q16 = exponentials_q16.sum(dim=-1, keepdim=True)
    reciprocal_q14 = (
        (1 << 30) + exponential_sums_q16 // 2
    ) // exponential_sums_q16
    probabilities_q16 = (
        (exponentials_q16 * reciprocal_q14 + (1 << 13)) >> 14
    ).clamp(max=65535)

    weighted_sums_q28 = torch.einsum(
        "hts,shd->thd", probabilities_q16, value_q12
    )
    attention_q12_heads = _symmetric_round_shift(weighted_sums_q28, 16)
    attention_q12_heads = attention_q12_heads.clamp(
        -(1 << 17), (1 << 17) - 1
    )
    attention_q12 = attention_q12_heads.reshape(qkv.shape[0], 768)
    attention = attention_q12.float() / 4096.0

    float_scores = torch.einsum(
        "thd,shd->hts", query_rotated, key_rotated
    ) / 8.0
    float_probabilities = torch.softmax(float_scores, dim=-1)
    float_attention = torch.einsum(
        "hts,shd->thd", float_probabilities, qkv[:, 2].float()
    ).reshape(qkv.shape[0], 768)
    probabilities = probabilities_q16.float() / 65536.0
    metrics: Dict[str, object] = {
        "score_format": "signed-q7.10-18-bit",
        "exponential_format": "unsigned-q0.16-16-bit-saturating",
        "probability_format": "unsigned-q0.16-16-bit-saturating",
        "value_and_output_format": "signed-q5.12-18-bit",
        "exponential_step_bits": exponential_step_bits,
        "exponential_lut_entries": lut_entries,
        "exponential_cutoff": exponential_cutoff,
        "reciprocal_format": "unsigned-q14-derived-from-2^30-over-sum",
        "score_saturation_fraction": float(score_saturation.float().mean().item()),
        "value_saturation_fraction": float(
            (
                (value_unclipped < -(1 << 17))
                | (value_unclipped > (1 << 17) - 1)
            ).float().mean().item()
        ),
        "score_q10_min": int(scores_q10.min().item()),
        "score_q10_max": int(scores_q10.max().item()),
        "minimum_delta_q10": int(deltas_q10.min().item()),
        "maximum_probability_row_sum_error_q16": int(
            (probabilities_q16.sum(dim=-1) - 65536).abs().max().item()
        ),
        "probability_accuracy": _error_metrics(
            probabilities, float_probabilities
        ),
        "attention_accuracy": _error_metrics(attention, float_attention),
        "score_argmax_agreement": float(
            probabilities_q16.argmax(dim=-1)
            .eq(float_probabilities.argmax(dim=-1))
            .float()
            .mean()
            .item()
        ),
        "rotary": rotary_details["metrics"],
    }
    tensors = {
        "scores_q10": scores_q10,
        "probabilities_q16": probabilities_q16,
        "value_q12": value_q12,
        "exponential_lut_q16": exponential_lut_q16,
    }
    return attention, attention_q12, {"metrics": metrics, "tensors": tensors}


def fixed_attention_projection_q10(
    attention_q12: Any,
    weight: Any,
    *,
    requant_shift: int = 24,
) -> Tuple[Any, Any, Dict[str, object]]:
    """Project Q5.12 attention with per-output INT8 folded weights."""
    import torch

    if attention_q12.ndim != 2 or attention_q12.shape[1] != 768:
        raise ValueError("attention_q12 must have shape tokens by 768")
    if tuple(weight.shape) != (768, 768):
        raise ValueError("attention projection weight must have shape 768 by 768")
    if requant_shift < 1:
        raise ValueError("requant_shift must be positive")

    weight_q, weight_scales = _quantize_weight_per_output(weight.float(), 8)
    accumulators = attention_q12.to(torch.int64) @ weight_q.to(torch.int64).t()
    # attention_q12 carries 12 fractional bits.  The desired output has 10,
    # so each per-output floating weight scale is divided by four.
    multipliers = torch.round(
        weight_scales.double() * (1 << requant_shift) / 4.0
    ).to(torch.int64)
    products = accumulators * multipliers[None, :]
    projection_unclipped_q10 = _symmetric_round_shift(products, requant_shift)
    output_min = -(1 << 23)
    output_max = (1 << 23) - 1
    projection_q10 = projection_unclipped_q10.clamp(output_min, output_max)
    projection = projection_q10.float() / 1024.0
    reference = (attention_q12.float() / 4096.0) @ weight.float().t()
    metrics: Dict[str, object] = {
        "activation_format": "signed-q5.12-18-bit",
        "weight_format": "signed-int8-per-output-scale",
        "output_format": "signed-q13.10-24-bit-saturating",
        "requant_shift": requant_shift,
        "requant_multiplier_min": int(multipliers.min().item()),
        "requant_multiplier_max": int(multipliers.max().item()),
        "required_unsigned_multiplier_bits": max(
            1, int(multipliers.max().item()).bit_length()
        ),
        "accumulator_min": int(accumulators.min().item()),
        "accumulator_max": int(accumulators.max().item()),
        "required_signed_accumulator_bits": max(
            1, int(accumulators.abs().max().item()).bit_length() + 1
        ),
        "output_saturation_fraction": float(
            (
                (projection_unclipped_q10 < output_min)
                | (projection_unclipped_q10 > output_max)
            ).float().mean().item()
        ),
        "projection_accuracy_from_fixed_attention": _error_metrics(
            projection, reference
        ),
    }
    tensors = {
        "weight_int8": weight_q,
        "requant_multiplier_q24": multipliers,
        "accumulators": accumulators,
    }
    return projection, projection_q10, {
        "metrics": metrics,
        "tensors": tensors,
    }


def fixed_qkv_projection_q12(
    normalized_q12: Any,
    weight: Any,
    bias: Any,
    *,
    requant_shift: int = 28,
    weight_bits: int = 16,
    smoothquant_alpha: Optional[float] = None,
    smoothquant_scales: Optional[Any] = None,
) -> Tuple[Any, Any, Dict[str, object]]:
    """Project fixed LayerNorm output with per-output signed integer weights."""
    import torch

    if normalized_q12.ndim != 2 or normalized_q12.shape[1] != 768:
        raise ValueError("normalized_q12 must have shape tokens by 768")
    if tuple(weight.shape) != (2304, 768):
        raise ValueError("QKV weight must have shape 2304 by 768")
    if tuple(bias.shape) != (2304,):
        raise ValueError("QKV bias must have shape 2304")
    if requant_shift < 1:
        raise ValueError("requant_shift must be positive")
    if weight_bits not in (8, 16):
        raise ValueError("weight_bits must be 8 or 16")
    if smoothquant_alpha is not None and not 0.0 <= smoothquant_alpha <= 1.0:
        raise ValueError("smoothquant_alpha must be between zero and one")
    if smoothquant_alpha is not None and smoothquant_scales is not None:
        raise ValueError(
            "smoothquant_alpha and smoothquant_scales are mutually exclusive"
        )

    projection_input_q12 = normalized_q12.to(torch.int64)
    projection_weight = weight.float()
    smoothing_scales = None
    smoothing_saturation_fraction = 0.0
    if smoothquant_alpha is not None:
        activation_maxima = (
            normalized_q12.double().abs().amax(dim=0) / 4096.0
        ).clamp_min(1e-8)
        weight_maxima = weight.double().abs().amax(dim=0).clamp_min(1e-8)
        smoothing_scales = (
            activation_maxima.pow(smoothquant_alpha)
            / weight_maxima.pow(1.0 - smoothquant_alpha)
        )
    elif smoothquant_scales is not None:
        smoothing_scales = torch.as_tensor(
            smoothquant_scales, dtype=torch.float64
        )
        if tuple(smoothing_scales.shape) != (768,):
            raise ValueError("smoothquant_scales must have shape 768")
        if not bool(torch.isfinite(smoothing_scales).all()) or bool(
            smoothing_scales.le(0).any()
        ):
            raise ValueError("smoothquant_scales must be finite and positive")

    if smoothing_scales is not None:
        scaled_unclipped = torch.round(
            normalized_q12.double() / smoothing_scales[None, :]
        ).to(torch.int64)
        smoothing_saturation = (scaled_unclipped < -(1 << 17)) | (
            scaled_unclipped > (1 << 17) - 1
        )
        smoothing_saturation_fraction = float(
            smoothing_saturation.float().mean().item()
        )
        projection_input_q12 = scaled_unclipped.clamp(
            -(1 << 17), (1 << 17) - 1
        )
        projection_weight = (
            weight.double() * smoothing_scales[None, :]
        ).float()

    weight_q, weight_scales = _quantize_weight_per_output(
        projection_weight, weight_bits
    )
    accumulators = projection_input_q12 @ weight_q.to(torch.int64).t()
    multipliers = torch.round(
        weight_scales.double() * (1 << requant_shift)
    ).to(torch.int64)
    bias_q12 = torch.round(bias.double() * 4096.0).to(torch.int64)
    output_unclipped_q12 = (
        _symmetric_round_shift(accumulators * multipliers[None, :], requant_shift)
        + bias_q12[None, :]
    )
    output_min = -(1 << 17)
    output_max = (1 << 17) - 1
    output_q12 = output_unclipped_q12.clamp(output_min, output_max)
    output = output_q12.float() / 4096.0
    reference = (normalized_q12.float() / 4096.0) @ weight.float().t()
    reference = reference + bias.float()[None, :]
    metrics: Dict[str, object] = {
        "activation_format": "signed-q5.12-18-bit",
        "weight_format": "signed-int%d-per-output-scale" % weight_bits,
        "weight_bits": weight_bits,
        "smoothquant_alpha": smoothquant_alpha,
        "smoothed_activation_saturation_fraction": (
            smoothing_saturation_fraction
        ),
        "bias_format": "signed-q5.12-18-bit",
        "output_format": "signed-q5.12-18-bit-saturating",
        "requant_shift": requant_shift,
        "requant_multiplier_min": int(multipliers.min().item()),
        "requant_multiplier_max": int(multipliers.max().item()),
        "required_unsigned_multiplier_bits": max(
            1, int(multipliers.max().item()).bit_length()
        ),
        "required_signed_accumulator_bits": max(
            1, int(accumulators.abs().max().item()).bit_length() + 1
        ),
        "output_saturation_fraction": float(
            (
                (output_unclipped_q12 < output_min)
                | (output_unclipped_q12 > output_max)
            ).float().mean().item()
        ),
        "projection_accuracy_from_fixed_norm": _error_metrics(
            output, reference
        ),
    }
    tensors = {
        "weight_int%d" % weight_bits: weight_q,
        "requant_multiplier_q28": multipliers,
        "bias_q12": bias_q12,
        "accumulators": accumulators,
    }
    if smoothing_scales is not None:
        tensors["smoothquant_input_scales"] = smoothing_scales
        tensors["smoothed_activation_q12"] = projection_input_q12
    return output, output_q12, {"metrics": metrics, "tensors": tensors}


def sweep_qkv_weight_precision(
    package_dir: Path,
    *,
    weight_bits: Tuple[int, ...] = (8, 16),
    smoothquant_alphas: Tuple[Optional[float], ...] = (None,),
) -> Dict[str, object]:
    """Screen QKV weight widths at every captured H0 attention boundary."""
    if not weight_bits or any(bits not in (8, 16) for bits in weight_bits):
        raise ValueError("weight_bits must contain only 8 or 16")
    if not smoothquant_alphas or any(
        alpha is not None and not 0.0 <= alpha <= 1.0
        for alpha in smoothquant_alphas
    ):
        raise ValueError(
            "smoothquant_alphas must contain none or values from zero to one"
        )

    names = ["folded.embedding"]
    for block in range(12):
        names.extend(
            [
                "folded.block_%02d.qkv" % block,
                "folded.block_%02d.attention" % block,
                "folded.block_%02d.output" % block,
            ]
        )
    goldens = _load_tensors(
        package_dir / "golden_tensors.safetensors", names
    )
    weight_names = []
    for block in range(12):
        weight_names.extend(
            [
                "block_%02d.qkv.weight" % block,
                "block_%02d.qkv.bias" % block,
            ]
        )
    tables = _load_tensors(
        package_dir / "folded_fp16_weights.safetensors",
        ["rotary.cos", "rotary.sin", *weight_names],
    )
    cosine = tables["rotary.cos"].float()
    sine = tables["rotary.sin"].float()
    designs = []
    for bits in weight_bits:
        selected_alphas = smoothquant_alphas if bits == 8 else (None,)
        for alpha in selected_alphas:
            blocks = _evaluate_qkv_precision_design(
                bits=bits,
                alpha=alpha,
                goldens=goldens,
                tables=tables,
                cosine=cosine,
                sine=sine,
            )
            qkv_errors = [
                block["qkv_accuracy_vs_h0"]["relative_rms_error"]
                for block in blocks
            ]
            attention_errors = [
                block["attention_accuracy_vs_h0"]["relative_rms_error"]
                for block in blocks
            ]
            designs.append(
                {
                    "weight_bits": bits,
                    "smoothquant_alpha": alpha,
                    "blocks": blocks,
                    "summary": {
                        "mean_qkv_relative_rms_vs_h0": statistics.mean(
                            qkv_errors
                        ),
                        "maximum_qkv_relative_rms_vs_h0": max(qkv_errors),
                        "mean_attention_relative_rms_vs_h0": (
                            statistics.mean(attention_errors)
                        ),
                        "maximum_attention_relative_rms_vs_h0": max(
                            attention_errors
                        ),
                        "maximum_qkv_output_saturation_fraction": max(
                            block["qkv_projection"][
                                "output_saturation_fraction"
                            ]
                            for block in blocks
                        ),
                        "maximum_smoothed_activation_saturation_fraction": (
                            max(
                                block["qkv_projection"][
                                    "smoothed_activation_saturation_fraction"
                                ]
                                for block in blocks
                            )
                        ),
                        "minimum_score_argmax_agreement": min(
                            block["attention"]["score_argmax_agreement"]
                            for block in blocks
                        ),
                    },
                }
            )
    return {
        "package_dir": str(package_dir),
        "block_count": 12,
        "designs": designs,
        "scope": (
            "all twelve H0 QKV projection and attention boundaries on one "
            "frozen input"
        ),
        "limitation": (
            "boundary fidelity screen only; fresh-canvas final-logit and "
            "generation validation remains required before changing RTL"
        ),
    }


def _evaluate_qkv_precision_design(
    *,
    bits: int,
    alpha: Optional[float],
    goldens: Dict[str, Any],
    tables: Dict[str, Any],
    cosine: Any,
    sine: Any,
) -> list[Dict[str, object]]:
    """Evaluate one QKV weight and equalization design."""
    blocks = []
    for block in range(12):
        residual = (
            goldens["folded.embedding"][0]
            if block == 0
            else goldens["folded.block_%02d.output" % (block - 1)][0]
        )
        _, normalized_q12, _ = fixed_layer_norm_q12(residual)
        fixed_qkv, _, qkv_details = fixed_qkv_projection_q12(
            normalized_q12,
            tables["block_%02d.qkv.weight" % block],
            tables["block_%02d.qkv.bias" % block],
            weight_bits=bits,
            smoothquant_alpha=alpha,
        )
        attention, _, attention_details = fixed_attention_q12(
            fixed_qkv, cosine, sine
        )
        blocks.append(
            {
                "block": block,
                "qkv_accuracy_vs_h0": _error_metrics(
                    fixed_qkv,
                    goldens["folded.block_%02d.qkv" % block][0],
                ),
                "attention_accuracy_vs_h0": _error_metrics(
                    attention,
                    goldens["folded.block_%02d.attention" % block][0],
                ),
                "qkv_projection": qkv_details["metrics"],
                "attention": attention_details["metrics"],
            }
        )
    return blocks


def sweep_fixed_attention(package_dir: Path) -> Dict[str, object]:
    """Validate attention, output projection, and residual for all H0 blocks."""
    import torch

    names = ["folded.embedding"]
    for block in range(12):
        names.extend(
            [
                "folded.block_%02d.qkv" % block,
                "folded.block_%02d.attention" % block,
                "folded.block_%02d.attention_projection" % block,
                "folded.block_%02d.after_attention" % block,
                "folded.block_%02d.output" % block,
            ]
        )
    goldens = _load_tensors(
        package_dir / "golden_tensors.safetensors", names
    )
    weight_names = [
        "block_%02d.attention_out.weight" % block for block in range(12)
    ]
    for block in range(12):
        weight_names.extend(
            [
                "block_%02d.qkv.weight" % block,
                "block_%02d.qkv.bias" % block,
            ]
        )
    tables = _load_tensors(
        package_dir / "folded_fp16_weights.safetensors",
        ["rotary.cos", "rotary.sin", *weight_names],
    )
    cosine = tables["rotary.cos"].float()
    sine = tables["rotary.sin"].float()
    blocks = []
    for block in range(12):
        residual = (
            goldens["folded.embedding"][0]
            if block == 0
            else goldens["folded.block_%02d.output" % (block - 1)][0]
        )
        _, normalized_q12, _ = fixed_layer_norm_q12(residual)
        fixed_qkv, fixed_qkv_q12, qkv_details = fixed_qkv_projection_q12(
            normalized_q12,
            tables["block_%02d.qkv.weight" % block],
            tables["block_%02d.qkv.bias" % block],
        )
        qkv_details["metrics"]["accuracy_vs_h0_golden"] = _error_metrics(
            fixed_qkv, goldens["folded.block_%02d.qkv" % block][0]
        )
        cascade_attention, cascade_attention_q12, cascade_attention_details = (
            fixed_attention_q12(fixed_qkv, cosine, sine)
        )
        qkv_details["metrics"]["attention_accuracy_vs_h0_golden"] = (
            _error_metrics(
                cascade_attention,
                goldens["folded.block_%02d.attention" % block][0],
            )
        )
        attention, attention_q12, details = fixed_attention_q12(
            goldens["folded.block_%02d.qkv" % block][0], cosine, sine
        )
        metrics = details["metrics"]
        metrics["qkv_projection"] = qkv_details["metrics"]
        metrics["accuracy_vs_h0_golden"] = _error_metrics(
            attention, goldens["folded.block_%02d.attention" % block][0]
        )
        projection, projection_q10, projection_details = (
            fixed_attention_projection_q10(
                cascade_attention_q12,
                tables["block_%02d.attention_out.weight" % block],
            )
        )
        metrics["projection"] = projection_details["metrics"]
        metrics["projection"]["accuracy_vs_h0_golden"] = _error_metrics(
            projection,
            goldens["folded.block_%02d.attention_projection" % block][0],
        )
        residual_q10 = torch.round(residual.double() * 1024.0).to(torch.int64)
        residual_q10 = residual_q10.clamp(-(1 << 23), (1 << 23) - 1)
        after_attention_unclipped_q10 = residual_q10 + projection_q10
        after_attention_q10 = after_attention_unclipped_q10.clamp(
            -(1 << 23), (1 << 23) - 1
        )
        after_attention = after_attention_q10.float() / 1024.0
        metrics["residual"] = {
            "format": "signed-q13.10-24-bit-saturating",
            "output_saturation_fraction": float(
                (
                    (after_attention_unclipped_q10 < -(1 << 23))
                    | (after_attention_unclipped_q10 > (1 << 23) - 1)
                ).float().mean().item()
            ),
            "accuracy_vs_h0_golden": _error_metrics(
                after_attention,
                goldens["folded.block_%02d.after_attention" % block][0],
            ),
        }
        blocks.append({"block": block, "metrics": metrics})
    errors = [
        block["metrics"]["accuracy_vs_h0_golden"]["relative_rms_error"]
        for block in blocks
    ]
    projection_errors = [
        block["metrics"]["projection"]["accuracy_vs_h0_golden"][
            "relative_rms_error"
        ]
        for block in blocks
    ]
    residual_errors = [
        block["metrics"]["residual"]["accuracy_vs_h0_golden"][
            "relative_rms_error"
        ]
        for block in blocks
    ]
    qkv_errors = [
        block["metrics"]["qkv_projection"]["accuracy_vs_h0_golden"][
            "relative_rms_error"
        ]
        for block in blocks
    ]
    cascade_attention_errors = [
        block["metrics"]["qkv_projection"][
            "attention_accuracy_vs_h0_golden"
        ]["relative_rms_error"]
        for block in blocks
    ]
    return {
        "package_dir": str(package_dir),
        "block_count": len(blocks),
        "blocks": blocks,
        "summary": {
            "mean_attention_relative_rms_vs_h0": statistics.mean(errors),
            "maximum_attention_relative_rms_vs_h0": max(errors),
            "maximum_score_saturation_fraction": max(
                block["metrics"]["score_saturation_fraction"]
                for block in blocks
            ),
            "maximum_value_saturation_fraction": max(
                block["metrics"]["value_saturation_fraction"]
                for block in blocks
            ),
            "minimum_score_argmax_agreement": min(
                block["metrics"]["score_argmax_agreement"]
                for block in blocks
            ),
            "maximum_probability_row_sum_error_q16": max(
                block["metrics"]["maximum_probability_row_sum_error_q16"]
                for block in blocks
            ),
            "mean_projection_relative_rms_vs_h0": statistics.mean(
                projection_errors
            ),
            "maximum_projection_relative_rms_vs_h0": max(projection_errors),
            "mean_after_attention_relative_rms_vs_h0": statistics.mean(
                residual_errors
            ),
            "maximum_after_attention_relative_rms_vs_h0": max(residual_errors),
            "maximum_projection_output_saturation_fraction": max(
                block["metrics"]["projection"][
                    "output_saturation_fraction"
                ]
                for block in blocks
            ),
            "maximum_residual_output_saturation_fraction": max(
                block["metrics"]["residual"]["output_saturation_fraction"]
                for block in blocks
            ),
            "mean_qkv_projection_relative_rms_vs_h0": statistics.mean(
                qkv_errors
            ),
            "maximum_qkv_projection_relative_rms_vs_h0": max(qkv_errors),
            "mean_attention_from_fixed_qkv_relative_rms_vs_h0": (
                statistics.mean(cascade_attention_errors)
            ),
            "maximum_attention_from_fixed_qkv_relative_rms_vs_h0": max(
                cascade_attention_errors
            ),
        },
        "scope": (
            "all twelve complete H0 attention, folded output projection, and "
            "residual boundaries on one frozen input"
        ),
    }
