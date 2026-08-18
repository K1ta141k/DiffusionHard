"""Bit-accurate unaffine LayerNorm reference for the fixed MDLM graph."""

from __future__ import annotations

import math
from pathlib import Path
import statistics
from typing import Any, Dict, Tuple

from .fixed_mlp import _error_metrics, _load_tensors


def fixed_layer_norm_q12(
    activation: Any,
    *,
    input_fraction_bits: int = 10,
    output_fraction_bits: int = 12,
    inverse_fraction_bits: int = 18,
    sqrt_extra_fraction_bits: int = 6,
    epsilon: float = 1e-5,
) -> Tuple[Any, Any, Dict[str, object]]:
    """Normalize 768-channel Q13.10 inputs into signed 18-bit Q5.12."""
    import torch

    if activation.ndim != 2 or activation.shape[1] != 768:
        raise ValueError("fixed LayerNorm expects a token by 768 matrix")
    input_min = -(1 << 23)
    input_max = (1 << 23) - 1
    unclipped = torch.round(
        activation.double() * (1 << input_fraction_bits)
    ).to(torch.int64)
    activation_q = unclipped.clamp(input_min, input_max)
    channel_count = activation_q.shape[1]
    sums = activation_q.sum(dim=1)
    means_q10 = torch.sign(sums) * (
        (sums.abs() + channel_count // 2) // channel_count
    )
    sum_squares = (activation_q * activation_q).sum(dim=1)
    mean_squares_q20 = (
        sum_squares + channel_count // 2
    ) // channel_count
    variance_q20 = (mean_squares_q20 - means_q10 * means_q10).clamp_min(0)
    epsilon_q20 = round(epsilon * (1 << (2 * input_fraction_bits)))
    radicands = variance_q20 + epsilon_q20
    roots = torch.tensor(
        [
            math.isqrt(int(value) << (2 * sqrt_extra_fraction_bits))
            for value in radicands
        ],
        dtype=torch.int64,
    )
    inverse_numerator = 1 << (
        inverse_fraction_bits
        + input_fraction_bits
        + sqrt_extra_fraction_bits
    )
    inverse_q18 = (inverse_numerator + roots // 2) // roots
    deviations_q10 = activation_q - means_q10[:, None]
    products = deviations_q10 * inverse_q18[:, None]
    output_shift = (
        inverse_fraction_bits + input_fraction_bits - output_fraction_bits
    )
    output_q = torch.sign(products) * (
        (products.abs() + (1 << (output_shift - 1))) >> output_shift
    )
    output_q = output_q.clamp(-(1 << 17), (1 << 17) - 1)
    output = output_q.float() / float(1 << output_fraction_bits)
    reference = torch.nn.functional.layer_norm(activation.float(), [768])
    metrics: Dict[str, object] = {
        "input_format": "signed-q13.10-24-bit",
        "output_format": "signed-q5.12-18-bit",
        "mean_format": "signed-q13.10",
        "variance_format": "unsigned-q12.20-32-bit",
        "inverse_std_format": "unsigned-q18",
        "sqrt_extra_fraction_bits": sqrt_extra_fraction_bits,
        "epsilon_q20": epsilon_q20,
        "input_saturation_fraction": float(
            ((unclipped < input_min) | (unclipped > input_max))
            .float()
            .mean()
            .item()
        ),
        "variance_bits_required": max(
            1, int(variance_q20.max().item()).bit_length()
        ),
        "sqrt_radicand_bits_required": max(
            1,
            (
                int(radicands.max().item())
                << (2 * sqrt_extra_fraction_bits)
            ).bit_length(),
        ),
        "inverse_std_bits_required": max(
            1, int(inverse_q18.max().item()).bit_length()
        ),
        "mean_q10_min": int(means_q10.min().item()),
        "mean_q10_max": int(means_q10.max().item()),
        "variance_q20_min": int(variance_q20.min().item()),
        "variance_q20_max": int(variance_q20.max().item()),
        "inverse_q18_min": int(inverse_q18.min().item()),
        "inverse_q18_max": int(inverse_q18.max().item()),
        "accuracy": _error_metrics(output, reference),
    }
    return output, output_q, metrics


def sweep_fixed_layer_norm(package_dir: Path) -> Dict[str, object]:
    """Validate every norm1, norm2, and final LayerNorm H0 boundary."""
    names = ["folded.embedding", "folded.final.norm_unaffine"]
    for block in range(12):
        prefix = "folded.block_%02d" % block
        names.extend(
            [
                prefix + ".after_attention",
                prefix + ".output",
                prefix + ".norm1_unaffine",
                prefix + ".norm2_unaffine",
            ]
        )
    tensors = _load_tensors(
        package_dir / "golden_tensors.safetensors", names
    )
    boundaries = []
    for block in range(12):
        prefix = "folded.block_%02d" % block
        norm1_input = (
            tensors["folded.embedding"][0]
            if block == 0
            else tensors["folded.block_%02d.output" % (block - 1)][0]
        )
        for name, activation, expected in (
            (
                "block_%02d.norm1" % block,
                norm1_input,
                tensors[prefix + ".norm1_unaffine"][0],
            ),
            (
                "block_%02d.norm2" % block,
                tensors[prefix + ".after_attention"][0],
                tensors[prefix + ".norm2_unaffine"][0],
            ),
        ):
            output, _, metrics = fixed_layer_norm_q12(activation)
            metrics["accuracy_vs_h0_golden"] = _error_metrics(output, expected)
            boundaries.append({"name": name, "metrics": metrics})
    final_input = tensors["folded.block_11.output"][0]
    final_output, _, final_metrics = fixed_layer_norm_q12(final_input)
    final_metrics["accuracy_vs_h0_golden"] = _error_metrics(
        final_output, tensors["folded.final.norm_unaffine"][0]
    )
    boundaries.append({"name": "final.norm", "metrics": final_metrics})
    relative_errors = [
        boundary["metrics"]["accuracy_vs_h0_golden"]["relative_rms_error"]
        for boundary in boundaries
    ]
    return {
        "package_dir": str(package_dir),
        "boundary_count": len(boundaries),
        "boundaries": boundaries,
        "summary": {
            "minimum_relative_rms": min(relative_errors),
            "mean_relative_rms": statistics.mean(relative_errors),
            "median_relative_rms": statistics.median(relative_errors),
            "maximum_relative_rms": max(relative_errors),
            "maximum_input_saturation_fraction": max(
                boundary["metrics"]["input_saturation_fraction"]
                for boundary in boundaries
            ),
            "maximum_variance_bits_required": max(
                boundary["metrics"]["variance_bits_required"]
                for boundary in boundaries
            ),
            "maximum_sqrt_radicand_bits_required": max(
                boundary["metrics"]["sqrt_radicand_bits_required"]
                for boundary in boundaries
            ),
            "maximum_inverse_std_bits_required": max(
                boundary["metrics"]["inverse_std_bits_required"]
                for boundary in boundaries
            ),
        },
        "scope": "all 25 H0 unaffine LayerNorm boundaries on one frozen input",
    }
