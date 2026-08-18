import pytest

from diffusion_accel.mdlm import (
    MDLMSpec,
    MDLMStepMeasurement,
    lower_mdlm_measurements,
)
from diffusion_accel.optimizations import (
    fused_streaming_candidate_head,
    masked_output_head,
    quantize_weights,
)
from diffusion_accel.trace import synthetic_trace


def _real_shaped_trace():
    spec = MDLMSpec(
        model_id="test/mdlm",
        revision="abc123",
        hidden_size=64,
        layers=2,
        heads=4,
        vocab_size=128,
        parameter_count=100_000,
        model_weight_bytes=400_000,
    )
    return lower_mdlm_measurements(
        spec,
        [
            MDLMStepMeasurement(0, 16, 4, 0.01, 1.0),
            MDLMStepMeasurement(1, 8, 8, 0.008, 0.5),
        ],
        canvas_tokens=16,
    )


def test_masked_output_head_reduces_late_step_projection_and_logits() -> None:
    baseline = _real_shaped_trace()
    optimized = masked_output_head(baseline)

    baseline_projection = baseline.steps[1].operations[-2]
    optimized_projection = optimized.steps[1].operations[-2]
    assert optimized_projection.name == "masked_vocabulary_projection"
    assert optimized_projection.flops == baseline_projection.flops // 2
    assert optimized_projection.writes[0].size_bytes == (
        baseline_projection.writes[0].size_bytes // 2
    )
    assert optimized.steps[1].operations[-1].reads[0].size_bytes == (
        baseline.steps[1].operations[-1].reads[0].size_bytes // 2
    )
    assert optimized.metadata["optimizations"] == ["masked-output-head"]


def test_masked_output_head_requires_real_full_projection() -> None:
    with pytest.raises(ValueError, match="full_vocabulary_projection"):
        masked_output_head(synthetic_trace(steps=1))


def test_fused_streaming_candidate_head_removes_logit_roundtrip_only() -> None:
    baseline = masked_output_head(_real_shaped_trace())
    fused = fused_streaming_candidate_head(baseline)

    baseline_projection = baseline.steps[0].operations[-2]
    fused_projection = fused.steps[0].operations[-2]
    baseline_sampler = baseline.steps[0].operations[-1]
    fused_sampler = fused.steps[0].operations[-1]
    assert fused_projection.name == "fused_streaming_candidate_projection"
    assert fused_projection.flops == baseline_projection.flops
    assert fused_projection.writes[0].category == "metadata"
    assert fused_projection.writes[0].size_bytes == 20
    assert fused_sampler.name == "streaming_candidate_select"
    assert fused_sampler.flops == baseline_sampler.flops
    assert fused_sampler.reads[0].name == fused_projection.writes[0].name
    assert fused.metadata["categorical_compute_preserved"] is True


def test_quantize_weights_changes_only_weight_traffic() -> None:
    baseline = _real_shaped_trace()
    quantized = quantize_weights(baseline, target_bits=8)
    baseline_block = baseline.steps[0].operations[1]
    quantized_block = quantized.steps[0].operations[1]

    assert quantized_block.reads[0].category == "weight"
    assert quantized_block.reads[0].size_bytes == baseline_block.reads[0].size_bytes // 4
    assert quantized_block.reads[1] == baseline_block.reads[1]
    assert quantized_block.flops == baseline_block.flops
    assert quantized.metadata["weight_bits"] == 8
    assert quantized.metadata["quantization_accuracy_validated"] is False


def test_quantize_weights_rejects_non_reduction() -> None:
    with pytest.raises(ValueError, match="smaller"):
        quantize_weights(
            quantize_weights(_real_shaped_trace(), target_bits=8),
            target_bits=8,
        )


def test_mixed_int8_preserves_output_projection_weight_bytes() -> None:
    baseline = _real_shaped_trace()
    quantized = quantize_weights(
        baseline,
        target_bits=8,
        preserve_output_head=True,
    )
    baseline_projection = baseline.steps[0].operations[-2]
    quantized_projection = quantized.steps[0].operations[-2]
    assert quantized_projection.reads[1] == baseline_projection.reads[1]
    assert quantized.metadata["preserve_output_head_fp32"] is True
