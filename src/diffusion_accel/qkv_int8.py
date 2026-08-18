"""Error-propagating quality screens for packed INT8 QKV projection."""

from __future__ import annotations

from pathlib import Path
import statistics
from typing import Any, Dict, Optional, Sequence

from .attention_int8 import dynamic_qk_fixed_pv_attention_q12
from .fixed_attention import fixed_qkv_projection_q12, _load_tensors
from .fixed_norm import fixed_layer_norm_q12


def screen_qkv_int8_logits(
    package_dir: Path,
    *,
    seeds: Optional[Sequence[int]] = None,
    device: str = "auto",
    smoothquant_alpha: float = 0.25,
) -> Dict[str, object]:
    """Compare calibrated INT8 QKV against INT16 through all model blocks."""
    import torch
    import torch.nn.functional as functional

    from .hardware_package import (
        _constant_condition,
        _fold_block,
        _fold_output_layer,
    )
    from .mdlm import (
        DEFAULT_MODEL_ID,
        DEFAULT_REVISION,
        _load_mdlm_model,
        _resolve_device,
    )

    if not 0.0 <= smoothquant_alpha <= 1.0:
        raise ValueError("smoothquant_alpha must be between zero and one")
    selected_seeds = [101, 202, 303, 404, 505] if seeds is None else list(seeds)
    if not selected_seeds:
        raise ValueError("at least one logit-screen seed is required")

    golden_names = ["folded.embedding"]
    qkv_parameter_names = []
    for block in range(12):
        golden_names.append("folded.block_%02d.output" % block)
        qkv_parameter_names.extend(
            [
                "block_%02d.qkv.weight" % block,
                "block_%02d.qkv.bias" % block,
            ]
        )
    goldens = _load_tensors(
        package_dir / "golden_tensors.safetensors", golden_names
    )
    tables = _load_tensors(
        package_dir / "folded_fp16_weights.safetensors",
        ["rotary.cos", "rotary.sin", *qkv_parameter_names],
    )
    cosine = tables["rotary.cos"].float()
    sine = tables["rotary.sin"].float()

    calibration_scales = []
    calibration_metrics = []
    for block in range(12):
        residual = (
            goldens["folded.embedding"][0]
            if block == 0
            else goldens["folded.block_%02d.output" % (block - 1)][0]
        )
        _, normalized_q12, _ = fixed_layer_norm_q12(residual)
        _, _, details = fixed_qkv_projection_q12(
            normalized_q12,
            tables["block_%02d.qkv.weight" % block],
            tables["block_%02d.qkv.bias" % block],
            weight_bits=8,
            smoothquant_alpha=smoothquant_alpha,
        )
        calibration_scales.append(
            details["tensors"]["smoothquant_input_scales"]
        )
        calibration_metrics.append(
            {
                "block": block,
                "minimum_scale": float(calibration_scales[-1].min().item()),
                "maximum_scale": float(calibration_scales[-1].max().item()),
                "activation_saturation_fraction": details["metrics"][
                    "smoothed_activation_saturation_fraction"
                ],
            }
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
    folded_blocks = [
        _fold_block(block, condition) for block in backbone.blocks
    ]

    def fixed_qkv_forward(input_ids: Any, *, weight_bits: int) -> Any:
        x = backbone.vocab_embed(input_ids)
        for block_index, folded in enumerate(folded_blocks):
            normalized = functional.layer_norm(x.float(), [hidden_dimension])
            normalized_q12 = (
                torch.round(normalized.detach().cpu().double() * 4096.0)
                .to(torch.int64)
                .clamp(-(1 << 17), (1 << 17) - 1)
            )
            _, qkv_q12, _ = fixed_qkv_projection_q12(
                normalized_q12[0],
                tables["block_%02d.qkv.weight" % block_index],
                tables["block_%02d.qkv.bias" % block_index],
                weight_bits=weight_bits,
                smoothquant_scales=(
                    calibration_scales[block_index]
                    if weight_bits == 8
                    else None
                ),
            )
            attention_q12, _ = dynamic_qk_fixed_pv_attention_q12(
                qkv_q12.float() / 4096.0, cosine, sine
            )
            attention = (
                attention_q12.to(device=resolved_device, dtype=x.dtype)[None]
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
            baseline = fixed_qkv_forward(input_ids, weight_bits=16)[
                0, 1::2
            ].float()
            candidate = fixed_qkv_forward(input_ids, weight_bits=8)[
                0, 1::2
            ].float()
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
                }
            )

    total_positions = sum(sample["masked_positions"] for sample in samples)
    total_matches = sum(sample["top1_matches"] for sample in samples)
    return {
        "artifact": "error-propagating SmoothQuant INT8 QKV logit screen",
        "model": DEFAULT_MODEL_ID,
        "revision": DEFAULT_REVISION,
        "device": resolved_device,
        "reference": "INT16 QKV with retained dynamic-INT8-QK fixed18-PV path",
        "candidate": {
            "qkv_weight_bits": 8,
            "smoothquant_alpha": smoothquant_alpha,
            "calibration": "deterministic hardware-package H0 trace",
        },
        "calibration_blocks": calibration_metrics,
        "samples": samples,
        "summary": {
            "sample_count": len(samples),
            "masked_positions": total_positions,
            "top1_matches": total_matches,
            "mean_top1_agreement": total_matches / total_positions,
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
            "QKV projection error propagated through all twelve blocks"
        ),
        "limitation": (
            "one forward evaluation per canvas, not an eight-evaluation "
            "generation-quality result"
        ),
    }
