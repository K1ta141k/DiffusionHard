import pytest

from diffusion_accel.apple_mdlm import (
    _reveal_steps_from_uniforms,
    _run_event_driven_candidate_sampler,
    _sample_categorical_from_logits,
)


def test_reveal_steps_include_terminal_residual_mass() -> None:
    torch = pytest.importorskip("torch")
    uniforms = torch.tensor([0.0, 0.24, 0.25, 0.74, 0.749, 0.9])
    reveal_steps = _reveal_steps_from_uniforms(
        uniforms,
        steps=3,
        sampling_epsilon=0.25,
    )
    assert reveal_steps.tolist() == [0, 0, 1, 2, 2, 3]


def test_logits_exponential_race_matches_probability_reference() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(9)
    logits = torch.randn((7, 31), generator=generator)
    uniforms = torch.rand(logits.shape, generator=generator)
    mask_token_id = 30

    reference_logits = logits.clone()
    reference_logits[:, mask_token_id] = -torch.inf
    probabilities = torch.softmax(reference_logits, dim=-1)
    exponential = 1e-10 - torch.log(uniforms + 1e-10)
    reference = (probabilities / exponential).argmax(dim=-1)
    optimized = _sample_categorical_from_logits(
        logits,
        mask_token_id=mask_token_id,
        uniforms=uniforms,
    )
    assert optimized.equal(reference)


def test_event_sampler_projects_each_revealed_position_once() -> None:
    torch = pytest.importorskip("torch")
    calls = []

    def logits_function(input_ids, positions):
        calls.append(positions.tolist())
        logits = torch.full((positions.numel(), 4), -torch.inf)
        logits[:, 0] = 0.0
        return logits

    output, metadata = _run_event_driven_candidate_sampler(
        logits_function,
        torch.full((1, 4), 3, dtype=torch.long),
        mask_token_id=3,
        steps=2,
        device="cpu",
        reveal_seed=0,
        sampling_epsilon=0.0,
        reveal_uniforms=torch.tensor([0.1, 0.4, 0.6, 0.9]),
    )

    assert output.tolist() == [[0, 0, 0, 0]]
    assert calls == [[0, 1], [2, 3]]
    assert metadata["model_evaluations"] == 2
    assert metadata["selected_output_rows"] == 4
    assert metadata["full_output_rows_equivalent"] == 8
    assert metadata["output_row_reduction"] == 0.5
    assert metadata["all_tokens_committed"] is True

