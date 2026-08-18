"""Captured-trace screen for paired-token INT8 attention."""

from __future__ import annotations

import math
from pathlib import Path
import statistics
from typing import Any, Dict, Optional, Sequence

from .fixed_attention import (
    _load_tensors,
    _symmetric_round_shift,
    fixed_attention_projection_q10,
    fixed_attention_q12,
    fixed_rotary_q12,
)
from .fixed_mlp import _error_metrics


def _requantize_symmetric_int8(values: Any) -> tuple[Any, int, int]:
    """Quantize an integer tensor with a captured maximum and Q24 reciprocal."""
    maximum = max(1, int(values.abs().max().item()))
    multiplier_q24 = round(127 * (1 << 24) / maximum)
    quantized = _symmetric_round_shift(values * multiplier_q24, 24)
    return quantized.clamp(-127, 127), maximum, multiplier_q24


def dynamic_qk_fixed_pv_attention_q12(
    qkv: Any,
    cosine: Any,
    sine: Any,
    *,
    qk_multiplier_fraction_bits: int = 17,
) -> tuple[Any, Dict[str, Any]]:
    """Run the retained dynamic-INT8-QK and fixed18-PV attention path.

    This is the bit-accurate software contract for the packed M8 RTL. Query
    and key vectors receive independent dynamic scales. Probabilities and
    values retain the existing Q16 by Q12 product.
    """
    import torch

    if qk_multiplier_fraction_bits != 17:
        raise ValueError("the retained packed RTL uses Q17 multipliers")
    _, _, query_q12, key_q12, rotary_details = fixed_rotary_q12(
        qkv, cosine, sine
    )
    value_q12 = rotary_details["tensors"]["qkv_q12"][:, 2].to(torch.int64)
    exponential_lut_q16 = torch.tensor(
        [
            min(65535, round(math.exp(-index / 64.0) * 65536))
            for index in range(1025)
        ],
        dtype=torch.int64,
        device=query_q12.device,
    )
    attention_heads = torch.empty_like(value_q12)
    score_heads = torch.empty(
        (12, 64, 64), dtype=torch.int64, device=query_q12.device
    )
    probability_heads = torch.empty_like(score_heads)
    query_maxima_heads = torch.empty(
        (12, 64), dtype=torch.int64, device=query_q12.device
    )
    key_maxima_heads = torch.empty_like(query_maxima_heads)
    for head in range(12):
        query = query_q12[:, head].to(torch.int64)
        key = key_q12[:, head].to(torch.int64)
        query_maxima = query.abs().amax(dim=1).clamp(min=1)
        key_maxima = key.abs().amax(dim=1).clamp(min=1)
        query_multipliers = torch.round(
            127 * (1 << 17) / query_maxima.double()
        ).to(torch.int64)
        key_multipliers = torch.round(
            127 * (1 << 17) / key_maxima.double()
        ).to(torch.int64)
        query_int8 = _symmetric_round_shift(
            query * query_multipliers[:, None], 17
        ).clamp(-127, 127)
        key_int8 = _symmetric_round_shift(
            key * key_multipliers[:, None], 17
        ).clamp(-127, 127)
        dots = query_int8 @ key_int8.t()
        score_multipliers = torch.round(
            query_maxima[:, None].double()
            * key_maxima[None, :].double()
            * (1 << 28)
            / (127 * 127 * (1 << 17))
        ).to(torch.int64)
        scores_q10 = _symmetric_round_shift(
            dots * score_multipliers, 28
        ).clamp(-(1 << 17), (1 << 17) - 1)
        deltas_q10 = (
            scores_q10 - scores_q10.amax(dim=-1, keepdim=True)
        ).clamp(min=-16384, max=0)
        addresses = ((-deltas_q10 + 8) >> 4).clamp(max=1024)
        exponentials_q16 = exponential_lut_q16[addresses]
        exponential_sums_q16 = exponentials_q16.sum(dim=-1, keepdim=True)
        reciprocal_q14 = (
            (1 << 30) + exponential_sums_q16 // 2
        ) // exponential_sums_q16
        probabilities_q16 = (
            (exponentials_q16 * reciprocal_q14 + (1 << 13)) >> 14
        ).clamp(max=65535)
        attention_heads[:, head] = _symmetric_round_shift(
            probabilities_q16 @ value_q12[:, head], 16
        ).clamp(-(1 << 17), (1 << 17) - 1)
        score_heads[head] = scores_q10
        probability_heads[head] = probabilities_q16
        query_maxima_heads[head] = query_maxima
        key_maxima_heads[head] = key_maxima
    return attention_heads.reshape(64, 768), {
        "tensors": {
            "query_q12": query_q12,
            "key_q12": key_q12,
            "value_q12": value_q12,
            "query_maxima_q12": query_maxima_heads,
            "key_maxima_q12": key_maxima_heads,
            "scores_q10": score_heads,
            "probabilities_q16": probability_heads,
            "attention_q12": attention_heads,
        }
    }


def _evaluate_with_fixed_maxima(
    qkv: Any,
    cosine: Any,
    sine: Any,
    calibration: Sequence[Dict[str, int]],
    exponential_lut_q16: Any,
    *,
    quantize_qk: bool = True,
    quantize_pv: bool = True,
    qk_bits: int = 8,
    qk_scale_mode: str = "calibrated-head",
    qk_multiplier_fraction_bits: int = 24,
    pv_probability_levels: int = 127,
    pv_probability_sum_correction: bool = False,
    pv_value_bits: int = 8,
) -> tuple[Any, Any, list[Dict[str, object]]]:
    """Evaluate packed INT8 attention using previously calibrated maxima."""
    import torch

    if qk_bits not in {8, 9}:
        raise ValueError("qk_bits must be 8 or 9")
    if qk_scale_mode not in {"calibrated-head", "dynamic-vector"}:
        raise ValueError(
            "qk_scale_mode must be calibrated-head or dynamic-vector"
        )
    if qk_multiplier_fraction_bits not in {17, 24}:
        raise ValueError("qk_multiplier_fraction_bits must be 17 or 24")
    if pv_probability_levels not in {127, 255}:
        raise ValueError("pv_probability_levels must be 127 or 255")
    if pv_value_bits not in {8, 9}:
        raise ValueError("pv_value_bits must be 8 or 9")
    pv_value_levels = (1 << (pv_value_bits - 1)) - 1
    qk_levels = (1 << (qk_bits - 1)) - 1

    _, baseline_attention_q12, baseline_details = fixed_attention_q12(
        qkv, cosine, sine
    )
    _, _, query_q12, key_q12, _ = fixed_rotary_q12(qkv, cosine, sine)
    value_q12 = (
        torch.round(qkv.view(64, 3, 12, 64)[:, 2].double() * 4096.0)
        .to(torch.int64)
        .clamp(-(1 << 17), (1 << 17) - 1)
    )
    candidate_heads_q12 = torch.empty_like(value_q12)
    head_metrics = []
    for head in range(12):
        maxima = calibration[head]

        def quantize(
            values: Any,
            maximum: int,
            levels: int,
            fraction_bits: int = 24,
        ) -> tuple[Any, int, float]:
            multiplier = round(
                levels * (1 << fraction_bits) / max(1, maximum)
            )
            quantized = _symmetric_round_shift(
                values * multiplier, fraction_bits
            )
            saturation = float(values.abs().gt(maximum).float().mean().item())
            return quantized.clamp(-levels, levels), multiplier, saturation

        if qk_scale_mode == "dynamic-vector":
            query_maxima = query_q12[:, head].abs().amax(dim=-1).clamp(min=1)
            key_maxima = key_q12[:, head].abs().amax(dim=-1).clamp(min=1)
            query_multipliers = torch.round(
                qk_levels
                * (1 << qk_multiplier_fraction_bits)
                / query_maxima.double()
            ).to(torch.int64)
            key_multipliers = torch.round(
                qk_levels
                * (1 << qk_multiplier_fraction_bits)
                / key_maxima.double()
            ).to(torch.int64)
            query_int8 = _symmetric_round_shift(
                query_q12[:, head] * query_multipliers[:, None],
                qk_multiplier_fraction_bits,
            ).clamp(-qk_levels, qk_levels)
            key_int8 = _symmetric_round_shift(
                key_q12[:, head] * key_multipliers[:, None],
                qk_multiplier_fraction_bits,
            ).clamp(-qk_levels, qk_levels)
            query_saturation = 0.0
            key_saturation = 0.0
        else:
            query_int8, _, query_saturation = quantize(
                query_q12[:, head],
                maxima["query_q12"],
                qk_levels,
                qk_multiplier_fraction_bits,
            )
            key_int8, _, key_saturation = quantize(
                key_q12[:, head],
                maxima["key_q12"],
                qk_levels,
                qk_multiplier_fraction_bits,
            )
        value_quantized, _, value_saturation = quantize(
            value_q12[:, head], maxima["value_q12"], pv_value_levels
        )
        baseline_scores = baseline_details["tensors"]["scores_q10"][head]
        if quantize_qk:
            dot_products_int16 = query_int8 @ key_int8.t()
            if qk_scale_mode == "dynamic-vector":
                score_multiplier_q28 = torch.round(
                    query_maxima[:, None].double()
                    * key_maxima[None, :].double()
                    * (1 << 28)
                    / (qk_levels * qk_levels * (1 << 17))
                ).to(torch.int64)
            else:
                score_multiplier_q28 = round(
                    maxima["query_q12"]
                    * maxima["key_q12"]
                    * (1 << 28)
                    / (qk_levels * qk_levels * (1 << 17))
                )
            scores_q10 = _symmetric_round_shift(
                dot_products_int16 * score_multiplier_q28, 28
            ).clamp(-(1 << 17), (1 << 17) - 1)
            deltas_q10 = (
                scores_q10 - scores_q10.amax(dim=-1, keepdim=True)
            ).clamp(min=-16384, max=0)
            addresses = ((-deltas_q10 + 8) >> 4).clamp(max=1024)
            exponentials_q16 = exponential_lut_q16[addresses]
            exponential_sums_q16 = exponentials_q16.sum(dim=-1, keepdim=True)
            reciprocal_q14 = (
                (1 << 30) + exponential_sums_q16 // 2
            ) // exponential_sums_q16
            probabilities_q16 = (
                (exponentials_q16 * reciprocal_q14 + (1 << 13)) >> 14
            ).clamp(max=65535)
        else:
            scores_q10 = baseline_scores
            probabilities_q16 = baseline_details["tensors"][
                "probabilities_q16"
            ][head]
        if quantize_pv:
            probabilities_quantized = (
                (
                    probabilities_q16 * pv_probability_levels
                    + (1 << 15)
                )
                >> 16
            ).clamp(0, pv_probability_levels)
            if pv_probability_sum_correction:
                row_residual = (
                    pv_probability_levels
                    - probabilities_quantized.sum(dim=-1)
                )
                maximum_indices = probabilities_q16.argmax(dim=-1)
                row_indices = torch.arange(
                    probabilities_quantized.shape[0],
                    device=probabilities_quantized.device,
                )
                probabilities_quantized[row_indices, maximum_indices] += (
                    row_residual
                )
            weighted_sums = probabilities_quantized @ value_quantized
            output_multiplier_q24 = round(
                maxima["value_q12"]
                * (1 << 24)
                / (pv_value_levels * pv_probability_levels)
            )
            candidate_heads_q12[:, head] = _symmetric_round_shift(
                weighted_sums * output_multiplier_q24, 24
            ).clamp(-(1 << 17), (1 << 17) - 1)
        else:
            weighted_sums_q28 = probabilities_q16 @ value_q12[:, head]
            candidate_heads_q12[:, head] = _symmetric_round_shift(
                weighted_sums_q28, 16
            ).clamp(-(1 << 17), (1 << 17) - 1)
        head_metrics.append(
            {
                "head": head,
                "query_saturation_fraction": query_saturation,
                "key_saturation_fraction": key_saturation,
                "value_saturation_fraction": value_saturation,
                "score_argmax_agreement_vs_fixed18": float(
                    scores_q10.argmax(dim=-1)
                    .eq(baseline_scores.argmax(dim=-1))
                    .float()
                    .mean()
                    .item()
                ),
            }
        )
    return candidate_heads_q12.reshape(64, 768), baseline_attention_q12, head_metrics


def screen_packed_int8_attention(package_dir: Path) -> Dict[str, object]:
    """Measure an integer-only paired-token attention candidate on H0 traces.

    Q, K, and V use one symmetric calibration maximum per block and head.
    Probabilities use unsigned magnitude levels 0 through 127 so they remain
    representable by the signed INT8 activation input of the packed MAC.
    """
    import torch

    qkv_names = ["folded.block_%02d.qkv" % block for block in range(12)]
    projection_names = [
        "block_%02d.attention_out.weight" % block for block in range(12)
    ]
    goldens = _load_tensors(
        package_dir / "golden_tensors.safetensors", qkv_names
    )
    tables = _load_tensors(
        package_dir / "folded_fp16_weights.safetensors",
        ["rotary.cos", "rotary.sin", *projection_names],
    )
    cosine = tables["rotary.cos"].float()
    sine = tables["rotary.sin"].float()
    exponential_lut_q16 = torch.tensor(
        [
            min(65535, round(math.exp(-index / 64.0) * 65536))
            for index in range(1025)
        ],
        dtype=torch.int64,
    )

    blocks = []
    for block, qkv_name in enumerate(qkv_names):
        qkv = goldens[qkv_name][0]
        _, baseline_attention_q12, baseline_details = fixed_attention_q12(
            qkv, cosine, sine
        )
        _, _, query_q12, key_q12, _ = fixed_rotary_q12(qkv, cosine, sine)
        value_q12 = (
            torch.round(qkv.view(64, 3, 12, 64)[:, 2].double() * 4096.0)
            .to(torch.int64)
            .clamp(-(1 << 17), (1 << 17) - 1)
        )
        candidate_heads_q12 = torch.empty_like(value_q12)
        head_metrics = []
        for head in range(12):
            query_int8, query_maximum, query_multiplier = (
                _requantize_symmetric_int8(query_q12[:, head])
            )
            key_int8, key_maximum, key_multiplier = (
                _requantize_symmetric_int8(key_q12[:, head])
            )
            value_int8, value_maximum, value_multiplier = (
                _requantize_symmetric_int8(value_q12[:, head])
            )

            dot_products_int16 = query_int8 @ key_int8.t()
            score_multiplier_q28 = round(
                query_maximum
                * key_maximum
                * (1 << 28)
                / (127 * 127 * (1 << 17))
            )
            scores_q10 = _symmetric_round_shift(
                dot_products_int16 * score_multiplier_q28, 28
            ).clamp(-(1 << 17), (1 << 17) - 1)
            deltas_q10 = (
                scores_q10 - scores_q10.amax(dim=-1, keepdim=True)
            ).clamp(min=-16384, max=0)
            addresses = ((-deltas_q10 + 8) >> 4).clamp(max=1024)
            exponentials_q16 = exponential_lut_q16[addresses]
            exponential_sums_q16 = exponentials_q16.sum(dim=-1, keepdim=True)
            reciprocal_q14 = (
                (1 << 30) + exponential_sums_q16 // 2
            ) // exponential_sums_q16
            probabilities_q16 = (
                (exponentials_q16 * reciprocal_q14 + (1 << 13)) >> 14
            ).clamp(max=65535)
            probabilities_int7 = (
                (probabilities_q16 * 127 + (1 << 15)) >> 16
            ).clamp(0, 127)

            weighted_sums = probabilities_int7 @ value_int8
            output_multiplier_q24 = round(
                value_maximum * (1 << 24) / (127 * 127)
            )
            candidate_heads_q12[:, head] = _symmetric_round_shift(
                weighted_sums * output_multiplier_q24, 24
            ).clamp(-(1 << 17), (1 << 17) - 1)
            baseline_scores = baseline_details["tensors"]["scores_q10"][head]
            head_metrics.append(
                {
                    "head": head,
                    "query_calibration_maximum_q12": query_maximum,
                    "key_calibration_maximum_q12": key_maximum,
                    "value_calibration_maximum_q12": value_maximum,
                    "query_requant_multiplier_q24": query_multiplier,
                    "key_requant_multiplier_q24": key_multiplier,
                    "value_requant_multiplier_q24": value_multiplier,
                    "score_multiplier_q28": score_multiplier_q28,
                    "output_multiplier_q24": output_multiplier_q24,
                    "score_argmax_agreement_vs_fixed18": float(
                        scores_q10.argmax(dim=-1)
                        .eq(baseline_scores.argmax(dim=-1))
                        .float()
                        .mean()
                        .item()
                    ),
                    "zero_probability_fraction_int7": float(
                        probabilities_int7.eq(0).float().mean().item()
                    ),
                }
            )

        candidate_attention_q12 = candidate_heads_q12.reshape(64, 768)
        attention_metrics = _error_metrics(
            candidate_attention_q12.float() / 4096.0,
            baseline_attention_q12.float() / 4096.0,
        )
        _, baseline_projection_q10, _ = fixed_attention_projection_q10(
            baseline_attention_q12,
            tables[projection_names[block]],
        )
        _, candidate_projection_q10, _ = fixed_attention_projection_q10(
            candidate_attention_q12,
            tables[projection_names[block]],
        )
        projection_metrics = _error_metrics(
            candidate_projection_q10.float() / 1024.0,
            baseline_projection_q10.float() / 1024.0,
        )
        blocks.append(
            {
                "block": block,
                "attention_accuracy_vs_fixed18": attention_metrics,
                "projection_accuracy_vs_fixed18": projection_metrics,
                "maximum_attention_absolute_error_q12": int(
                    (candidate_attention_q12 - baseline_attention_q12)
                    .abs()
                    .max()
                    .item()
                ),
                "minimum_score_argmax_agreement_vs_fixed18": min(
                    head["score_argmax_agreement_vs_fixed18"]
                    for head in head_metrics
                ),
                "heads": head_metrics,
            }
        )

    attention_errors = [
        block["attention_accuracy_vs_fixed18"]["relative_rms_error"]
        for block in blocks
    ]
    attention_cosines = [
        block["attention_accuracy_vs_fixed18"]["cosine_similarity"]
        for block in blocks
    ]
    projection_errors = [
        block["projection_accuracy_vs_fixed18"]["relative_rms_error"]
        for block in blocks
    ]
    return {
        "artifact": "paired-token INT8 attention captured-trace screen",
        "package_dir": str(package_dir),
        "blocks": blocks,
        "summary": {
            "mean_attention_relative_rms_vs_fixed18": statistics.mean(
                attention_errors
            ),
            "maximum_attention_relative_rms_vs_fixed18": max(attention_errors),
            "mean_attention_cosine_similarity_vs_fixed18": statistics.mean(
                attention_cosines
            ),
            "minimum_attention_cosine_similarity_vs_fixed18": min(
                attention_cosines
            ),
            "maximum_attention_absolute_error_q12": max(
                block["maximum_attention_absolute_error_q12"]
                for block in blocks
            ),
            "mean_projection_relative_rms_vs_fixed18": statistics.mean(
                projection_errors
            ),
            "maximum_projection_relative_rms_vs_fixed18": max(
                projection_errors
            ),
            "minimum_head_score_argmax_agreement_vs_fixed18": min(
                block["minimum_score_argmax_agreement_vs_fixed18"]
                for block in blocks
            ),
        },
        "formats": {
            "query_key_value": "signed-int8-symmetric-per-block-per-head",
            "probability": "unsigned-magnitude-7-bit-in-signed-int8-lane",
            "score": "signed-q7.10-18-bit",
            "attention_output": "signed-q5.12-18-bit",
        },
        "scope": (
            "all twelve H0 blocks on the same captured input used to calibrate "
            "each per-head maximum; this is not held-out quality evidence"
        ),
    }


def screen_packed_int8_attention_heldout(
    package_dir: Path,
    *,
    seeds: Optional[Sequence[int]] = None,
    device: str = "auto",
) -> Dict[str, object]:
    """Apply captured per-head calibration to fresh deterministic canvases."""
    import torch

    from .hardware_package import folded_mdlm_forward
    from .mdlm import (
        DEFAULT_MODEL_ID,
        DEFAULT_REVISION,
        _load_mdlm_model,
        _resolve_device,
    )

    selected_seeds = [101, 202, 303, 404, 505] if seeds is None else list(seeds)
    if not selected_seeds:
        raise ValueError("at least one held-out seed is required")
    qkv_names = ["folded.block_%02d.qkv" % block for block in range(12)]
    projection_names = [
        "block_%02d.attention_out.weight" % block for block in range(12)
    ]
    goldens = _load_tensors(
        package_dir / "golden_tensors.safetensors", qkv_names
    )
    tables = _load_tensors(
        package_dir / "folded_fp16_weights.safetensors",
        ["rotary.cos", "rotary.sin", *projection_names],
    )
    cosine = tables["rotary.cos"].float()
    sine = tables["rotary.sin"].float()
    exponential_lut_q16 = torch.tensor(
        [
            min(65535, round(math.exp(-index / 64.0) * 65536))
            for index in range(1025)
        ],
        dtype=torch.int64,
    )
    calibrations = []
    for qkv_name in qkv_names:
        qkv = goldens[qkv_name][0]
        _, _, query_q12, key_q12, _ = fixed_rotary_q12(qkv, cosine, sine)
        value_q12 = (
            torch.round(qkv.view(64, 3, 12, 64)[:, 2].double() * 4096.0)
            .to(torch.int64)
            .clamp(-(1 << 17), (1 << 17) - 1)
        )
        calibrations.append(
            [
                {
                    "query_q12": max(1, int(query_q12[:, head].abs().max())),
                    "key_q12": max(1, int(key_q12[:, head].abs().max())),
                    "value_q12": max(1, int(value_q12[:, head].abs().max())),
                }
                for head in range(12)
            ]
        )

    resolved_device = _resolve_device(device)
    model = _load_mdlm_model(
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_REVISION,
        device=resolved_device,
        local_files_only=True,
    )
    mask_token_id = int(model.config.vocab_size) - 1
    samples = []
    for seed in selected_seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        input_ids = torch.randint(
            0,
            mask_token_id,
            (1, 64),
            generator=generator,
            dtype=torch.long,
        )
        input_ids[:, 1::2] = mask_token_id
        captured: Dict[str, Any] = {}
        with torch.inference_mode():
            folded_mdlm_forward(
                model, input_ids.to(resolved_device), resolved_device, captured
            )
        block_results = []
        for block, qkv_name in enumerate(qkv_names):
            candidate_q12, baseline_q12, head_metrics = _evaluate_with_fixed_maxima(
                captured[qkv_name][0],
                cosine,
                sine,
                calibrations[block],
                exponential_lut_q16,
            )
            _, candidate_projection_q10, _ = fixed_attention_projection_q10(
                candidate_q12, tables[projection_names[block]]
            )
            _, baseline_projection_q10, _ = fixed_attention_projection_q10(
                baseline_q12, tables[projection_names[block]]
            )
            block_results.append(
                {
                    "block": block,
                    "attention_accuracy_vs_fixed18": _error_metrics(
                        candidate_q12.float() / 4096.0,
                        baseline_q12.float() / 4096.0,
                    ),
                    "projection_accuracy_vs_fixed18": _error_metrics(
                        candidate_projection_q10.float() / 1024.0,
                        baseline_projection_q10.float() / 1024.0,
                    ),
                    "maximum_operand_saturation_fraction": max(
                        max(
                            head["query_saturation_fraction"],
                            head["key_saturation_fraction"],
                            head["value_saturation_fraction"],
                        )
                        for head in head_metrics
                    ),
                    "minimum_score_argmax_agreement_vs_fixed18": min(
                        head["score_argmax_agreement_vs_fixed18"]
                        for head in head_metrics
                    ),
                }
            )
        samples.append({"seed": seed, "blocks": block_results})

    all_blocks = [block for sample in samples for block in sample["blocks"]]
    return {
        "artifact": "held-out paired-token INT8 attention boundary screen",
        "device": resolved_device,
        "calibration": "original deterministic hardware-package H0 trace",
        "samples": samples,
        "summary": {
            "sample_count": len(samples),
            "block_boundaries": len(all_blocks),
            "mean_attention_relative_rms_vs_fixed18": statistics.mean(
                block["attention_accuracy_vs_fixed18"]["relative_rms_error"]
                for block in all_blocks
            ),
            "maximum_attention_relative_rms_vs_fixed18": max(
                block["attention_accuracy_vs_fixed18"]["relative_rms_error"]
                for block in all_blocks
            ),
            "mean_projection_relative_rms_vs_fixed18": statistics.mean(
                block["projection_accuracy_vs_fixed18"]["relative_rms_error"]
                for block in all_blocks
            ),
            "maximum_projection_relative_rms_vs_fixed18": max(
                block["projection_accuracy_vs_fixed18"]["relative_rms_error"]
                for block in all_blocks
            ),
            "maximum_operand_saturation_fraction": max(
                block["maximum_operand_saturation_fraction"]
                for block in all_blocks
            ),
            "minimum_head_score_argmax_agreement_vs_fixed18": min(
                block["minimum_score_argmax_agreement_vs_fixed18"]
                for block in all_blocks
            ),
        },
        "scope": (
            "fresh deterministic random half-masked canvases, fixed calibration, "
            "independent attention boundaries without quantization-error propagation"
        ),
    }


def screen_packed_int8_attention_logits(
    package_dir: Path,
    *,
    seeds: Optional[Sequence[int]] = None,
    device: str = "auto",
    quantize_qk: bool = True,
    quantize_pv: bool = True,
    qk_bits: int = 8,
    qk_scale_mode: str = "calibrated-head",
    qk_multiplier_fraction_bits: int = 24,
    pv_probability_levels: int = 127,
    pv_probability_sum_correction: bool = False,
    pv_value_bits: int = 8,
) -> Dict[str, object]:
    """Propagate fixed-calibration INT8 attention through all twelve blocks."""
    import torch
    import torch.nn.functional as functional

    from .hardware_package import (
        _constant_condition,
        _fold_block,
        _fold_output_layer,
        folded_mdlm_forward,
    )
    from .mdlm import (
        DEFAULT_MODEL_ID,
        DEFAULT_REVISION,
        _load_mdlm_model,
        _resolve_device,
    )

    selected_seeds = [101, 202, 303, 404, 505] if seeds is None else list(seeds)
    if not selected_seeds:
        raise ValueError("at least one logit-screen seed is required")
    if not quantize_qk and not quantize_pv:
        raise ValueError("at least one attention matrix phase must be quantized")
    qkv_names = ["folded.block_%02d.qkv" % block for block in range(12)]
    goldens = _load_tensors(
        package_dir / "golden_tensors.safetensors", qkv_names
    )
    tables = _load_tensors(
        package_dir / "folded_fp16_weights.safetensors",
        ["rotary.cos", "rotary.sin"],
    )
    cosine = tables["rotary.cos"].float()
    sine = tables["rotary.sin"].float()
    exponential_lut_q16 = torch.tensor(
        [
            min(65535, round(math.exp(-index / 64.0) * 65536))
            for index in range(1025)
        ],
        dtype=torch.int64,
    )
    calibrations = []
    for qkv_name in qkv_names:
        qkv = goldens[qkv_name][0]
        _, _, query_q12, key_q12, _ = fixed_rotary_q12(qkv, cosine, sine)
        value_q12 = (
            torch.round(qkv.view(64, 3, 12, 64)[:, 2].double() * 4096.0)
            .to(torch.int64)
            .clamp(-(1 << 17), (1 << 17) - 1)
        )
        calibrations.append(
            [
                {
                    "query_q12": max(1, int(query_q12[:, head].abs().max())),
                    "key_q12": max(1, int(key_q12[:, head].abs().max())),
                    "value_q12": max(1, int(value_q12[:, head].abs().max())),
                }
                for head in range(12)
            ]
        )

    resolved_device = _resolve_device(device)
    model = _load_mdlm_model(
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_REVISION,
        device=resolved_device,
        local_files_only=True,
    )
    condition = _constant_condition(model, resolved_device)
    backbone = model.backbone
    hidden_dimension = int(model.config.hidden_dim)
    mask_token_id = int(model.config.vocab_size) - 1

    def candidate_forward(input_ids: Any) -> Any:
        x = backbone.vocab_embed(input_ids)
        for block_index, block in enumerate(backbone.blocks):
            folded = _fold_block(block, condition)
            normalized = functional.layer_norm(x.float(), [hidden_dimension])
            qkv = functional.linear(
                normalized, folded["qkv.weight"], folded["qkv.bias"]
            )
            candidate_q12, _, _ = _evaluate_with_fixed_maxima(
                qkv.detach().cpu()[0],
                cosine,
                sine,
                calibrations[block_index],
                exponential_lut_q16,
                quantize_qk=quantize_qk,
                quantize_pv=quantize_pv,
                qk_bits=qk_bits,
                qk_scale_mode=qk_scale_mode,
                qk_multiplier_fraction_bits=qk_multiplier_fraction_bits,
                pv_probability_levels=pv_probability_levels,
                pv_probability_sum_correction=pv_probability_sum_correction,
                pv_value_bits=pv_value_bits,
            )
            attention = (
                candidate_q12.to(device=resolved_device, dtype=x.dtype)[None]
                / 4096.0
            )
            x = x + functional.linear(
                attention, folded["attention_out.weight"], None
            )
            normalized = functional.layer_norm(x.float(), [hidden_dimension])
            mlp = functional.linear(
                normalized, folded["mlp_up.weight"], folded["mlp_up.bias"]
            )
            mlp = functional.gelu(mlp, approximate="tanh")
            mlp = functional.linear(
                mlp, folded["mlp_down.weight"], folded["mlp_down.bias"]
            )
            x = x + mlp
        output = _fold_output_layer(backbone.output_layer, condition)
        normalized = functional.layer_norm(x.float(), [hidden_dimension])
        return functional.linear(normalized, output["weight"], output["bias"])

    samples = []
    with torch.inference_mode():
        for seed in selected_seeds:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            input_ids = torch.randint(
                0,
                mask_token_id,
                (1, 64),
                generator=generator,
                dtype=torch.long,
            )
            input_ids[:, 1::2] = mask_token_id
            input_ids = input_ids.to(resolved_device)
            baseline = folded_mdlm_forward(
                model, input_ids, resolved_device
            )[0, 1::2].float()
            candidate = candidate_forward(input_ids)[0, 1::2].float()
            baseline[:, mask_token_id] = -torch.inf
            candidate[:, mask_token_id] = -torch.inf
            finite_baseline = torch.where(
                torch.isfinite(baseline), baseline, torch.zeros_like(baseline)
            )
            finite_candidate = torch.where(
                torch.isfinite(candidate), candidate, torch.zeros_like(candidate)
            )
            difference = finite_candidate - finite_baseline
            baseline_log_probabilities = torch.log_softmax(baseline, dim=-1)
            candidate_log_probabilities = torch.log_softmax(candidate, dim=-1)
            baseline_probabilities = baseline_log_probabilities.exp()
            candidate_probabilities = candidate_log_probabilities.exp()
            total_variation = 0.5 * (
                baseline_probabilities - candidate_probabilities
            ).abs().sum(dim=-1)
            log_probability_delta = torch.where(
                baseline_probabilities > 0,
                baseline_log_probabilities - candidate_log_probabilities,
                torch.zeros_like(baseline_log_probabilities),
            )
            kl_divergence = (
                baseline_probabilities * log_probability_delta
            ).sum(dim=-1)
            top1_matches = baseline.argmax(dim=-1).eq(
                candidate.argmax(dim=-1)
            )
            samples.append(
                {
                    "seed": seed,
                    "masked_positions": 32,
                    "top1_matches": int(top1_matches.sum().item()),
                    "top1_agreement": float(top1_matches.float().mean().item()),
                    "logit_cosine_similarity": float(
                        functional.cosine_similarity(
                            finite_baseline.flatten(),
                            finite_candidate.flatten(),
                            dim=0,
                        ).item()
                    ),
                    "normalized_logit_rmse": float(
                        (
                            difference.square().mean().sqrt()
                            / finite_baseline.square().mean().sqrt()
                        ).item()
                    ),
                    "mean_total_variation": float(total_variation.mean().item()),
                    "maximum_total_variation": float(total_variation.max().item()),
                    "mean_kl_divergence_nats": float(kl_divergence.mean().item()),
                    "maximum_kl_divergence_nats": float(kl_divergence.max().item()),
                }
            )
    total_masked_positions = sum(
        sample["masked_positions"] for sample in samples
    )
    total_top1_matches = sum(sample["top1_matches"] for sample in samples)
    return {
        "artifact": "error-propagating packed INT8 attention logit screen",
        "quantized_phases": {
            "query_key": quantize_qk,
            "probability_value": quantize_pv,
            "query_key_bits": qk_bits if quantize_qk else 18,
            "query_key_scale_mode": qk_scale_mode if quantize_qk else "fixed18",
            "query_key_multiplier_fraction_bits": (
                qk_multiplier_fraction_bits if quantize_qk else 0
            ),
            "probability_levels": pv_probability_levels if quantize_pv else 65535,
            "probability_sum_correction": (
                pv_probability_sum_correction if quantize_pv else False
            ),
            "value_bits": pv_value_bits if quantize_pv else 18,
        },
        "device": resolved_device,
        "samples": samples,
        "summary": {
            "sample_count": len(samples),
            "masked_positions": total_masked_positions,
            "top1_matches": total_top1_matches,
            "mean_top1_agreement": (
                total_top1_matches / total_masked_positions
            ),
            "minimum_top1_agreement": min(
                sample["top1_agreement"] for sample in samples
            ),
            "mean_logit_cosine_similarity": statistics.mean(
                sample["logit_cosine_similarity"] for sample in samples
            ),
            "maximum_normalized_logit_rmse": max(
                sample["normalized_logit_rmse"] for sample in samples
            ),
            "mean_total_variation": statistics.mean(
                sample["mean_total_variation"] for sample in samples
            ),
            "maximum_total_variation": max(
                sample["maximum_total_variation"] for sample in samples
            ),
        },
        "scope": (
            f"{len(samples)} fresh deterministic half-masked canvases with "
            "selected INT8 attention error propagated through all twelve "
            "otherwise floating-point folded blocks"
        ),
    }
