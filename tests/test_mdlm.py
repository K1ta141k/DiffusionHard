import pytest

from diffusion_accel.mdlm import (
    MDLMSpec,
    MDLMStepMeasurement,
    lower_mdlm_measurements,
    _mdlm_log_probabilities,
    _fake_quantize_weights_int8,
    _run_ddpm_cache_sampler,
    _run_ddpm_candidate_cache_sampler,
    validate_mdlm_output_head_int8_generation,
    _PREFIX_ISOLATION_TOKENS,
    _BLOCK_ISOLATION_ENDS,
    block_isolation,
    prefix_isolation,
)


def _spec() -> MDLMSpec:
    return MDLMSpec(
        model_id="test/mdlm",
        revision="abc123",
        hidden_size=64,
        layers=2,
        heads=4,
        vocab_size=128,
        parameter_count=100_000,
        model_weight_bytes=400_000,
    )


def test_lower_mdlm_measurements_captures_real_baseline_shape() -> None:
    trace = lower_mdlm_measurements(
        _spec(),
        [
            MDLMStepMeasurement(0, 16, 4, 0.01, 1.0),
            MDLMStepMeasurement(1, 12, 12, 0.008, 0.5),
        ],
        canvas_tokens=16,
        device="mps",
    )

    assert trace.workload_name == "mdlm-owt-real"
    assert trace.metadata["device"] == "mps"
    assert trace.steps[0].metadata["measured_step_latency_ms"] == 10.0
    projection = next(
        operation
        for operation in trace.steps[0].operations
        if operation.name == "full_vocabulary_projection"
    )
    assert projection.parallelism == 16
    assert projection.metadata["active_tokens"] == 16
    assert projection.metadata["computed_tokens"] == 16
    assert len(trace.steps[0].operations) == 5


def test_lower_mdlm_measurements_rejects_impossible_commit_count() -> None:
    with pytest.raises(ValueError, match="changed_tokens"):
        lower_mdlm_measurements(
            _spec(),
            [MDLMStepMeasurement(0, 4, 5, 0.01, 1.0)],
            canvas_tokens=4,
        )


def test_subs_parameterization_excludes_mask_token() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[1.0, 2.0, 9.0]]])
    log_probabilities = _mdlm_log_probabilities(logits, mask_token_id=2)
    assert torch.isneginf(log_probabilities[0, 0, 2])
    assert torch.allclose(log_probabilities.exp().sum(dim=-1), torch.ones(1, 1))


def test_output_head_only_fake_quantization_leaves_backbone_unchanged() -> None:
    torch = pytest.importorskip("torch")

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_layer = torch.nn.Linear(4, 3, bias=False)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = Backbone()
            self.other = torch.nn.Linear(4, 4, bias=False)

    model = Model()
    with torch.no_grad():
        model.backbone.output_layer.weight.copy_(
            torch.linspace(-0.93, 0.87, 12).reshape(3, 4)
        )
        model.other.weight.copy_(torch.linspace(-0.71, 0.82, 16).reshape(4, 4))
    output_before = model.backbone.output_layer.weight.detach().clone()
    other_before = model.other.weight.detach().clone()

    count = _fake_quantize_weights_int8(model, only_output_head=True)

    assert count == 12
    assert not model.backbone.output_layer.weight.equal(output_before)
    assert model.other.weight.equal(other_before)


def test_generation_quantization_gate_requires_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        validate_mdlm_output_head_int8_generation(seeds=[])


def test_ddpm_cache_tracks_model_evaluations_separately() -> None:
    torch = pytest.importorskip("torch")

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, input_ids, timesteps):
            del timesteps
            self.calls += 1
            logits = torch.zeros((*input_ids.shape, 4))
            logits[..., 0] = 2.0
            logits[..., 1] = 1.0
            return logits

    torch.manual_seed(4)
    model = FakeModel()
    output, measurements, metadata = _run_ddpm_cache_sampler(
        model,
        torch.full((1, 4), 3, dtype=torch.long),
        mask_token_id=3,
        steps=8,
        device="cpu",
    )
    assert 3 not in output.tolist()[0]
    assert model.calls == len(measurements)
    assert metadata["model_evaluations"] == len(measurements)
    assert metadata["sampling_transitions"] <= 8
    assert metadata["probability_cache_hits"] >= 0


def test_candidate_cache_tracks_compact_distribution_equivalent_state() -> None:
    torch = pytest.importorskip("torch")

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, input_ids, timesteps):
            del timesteps
            self.calls += 1
            logits = torch.zeros((*input_ids.shape, 4))
            logits[..., 0] = 2.0
            logits[..., 1] = 1.0
            return logits

    torch.manual_seed(4)
    model = FakeModel()
    output, measurements, metadata = _run_ddpm_candidate_cache_sampler(
        model,
        torch.full((1, 4), 3, dtype=torch.long),
        mask_token_id=3,
        steps=8,
        device="cpu",
    )

    assert 3 not in output.tolist()[0]
    assert model.calls == len(measurements)
    assert metadata["model_evaluations"] == len(measurements)
    assert metadata["candidate_cache_hits"] >= 0
    assert metadata["hardware_candidate_cache_bytes"] == 6
    assert metadata["candidate_id_bytes"] == 1
    assert metadata["cache_correctness_class"] == "distribution-equivalent"


def test_prefix_isolation_context_restores_default() -> None:
    assert _PREFIX_ISOLATION_TOKENS.get() == 0
    with prefix_isolation(7):
        assert _PREFIX_ISOLATION_TOKENS.get() == 7
    assert _PREFIX_ISOLATION_TOKENS.get() == 0


def test_block_isolation_context_restores_default() -> None:
    assert _BLOCK_ISOLATION_ENDS.get() == ()
    with block_isolation([8, 12]):
        assert _BLOCK_ISOLATION_ENDS.get() == (8, 12)
    assert _BLOCK_ISOLATION_ENDS.get() == ()


def test_block_isolation_rejects_unsorted_boundaries() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        with block_isolation([8, 8]):
            pass
