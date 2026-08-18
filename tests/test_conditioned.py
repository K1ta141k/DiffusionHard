import pytest

from diffusion_accel.conditioned import (
    _run_conditioned_ddpm_cache,
    _run_conditioned_ddpm_candidate_cache,
    _sequence_agreement,
)


def test_conditioned_sampler_commits_all_masks() -> None:
    torch = pytest.importorskip("torch")

    calls = 0

    def logits_function(input_ids, sigma):
        nonlocal calls
        del sigma
        calls += 1
        logits = torch.zeros((*input_ids.shape, 4))
        logits[..., 1] = 5.0
        return logits

    torch.manual_seed(2)
    output, metadata = _run_conditioned_ddpm_cache(
        logits_function,
        torch.full((1, 8), 3, dtype=torch.long),
        mask_token_id=3,
        steps=16,
        device="cpu",
    )
    assert 3 not in output[0].tolist()
    assert metadata["all_tokens_committed"] is True
    assert metadata["model_evaluations"] == calls
    assert metadata["probability_cache_hits"] >= 0
    assert metadata["terminal_kv_valid"] is False


def test_sequence_agreement_reports_fraction() -> None:
    torch = pytest.importorskip("torch")
    left = torch.tensor([[1, 2, 3, 4]])
    right = torch.tensor([[1, 9, 3, 9]])
    assert _sequence_agreement(left, right) == 0.5


def test_candidate_cache_sampler_commits_masks_with_compact_state() -> None:
    torch = pytest.importorskip("torch")

    calls = 0

    def logits_function(input_ids, sigma):
        nonlocal calls
        del sigma
        calls += 1
        logits = torch.zeros((*input_ids.shape, 4))
        logits[..., 1] = 5.0
        return logits

    torch.manual_seed(2)
    output, metadata = _run_conditioned_ddpm_candidate_cache(
        logits_function,
        torch.full((1, 8), 3, dtype=torch.long),
        mask_token_id=3,
        steps=16,
        device="cpu",
    )

    assert 3 not in output[0].tolist()
    assert metadata["all_tokens_committed"] is True
    assert metadata["model_evaluations"] == calls
    assert metadata["candidate_cache_hits"] >= 0
    assert metadata["hardware_candidate_cache_bytes"] == 10
    assert metadata["candidate_id_bytes"] == 1
