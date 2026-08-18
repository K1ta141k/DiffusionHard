import pytest

from diffusion_accel.rng import (
    PHILOX4X32_10_KNOWN_ANSWERS,
    approximate_factorized_gumbel_q_from_words,
    approximate_gumbel_q_from_words,
    philox4x32_10,
    validate_rng_and_gumbel,
)


def test_philox_matches_random123_known_answers() -> None:
    for counter, key, expected in PHILOX4X32_10_KNOWN_ANSWERS:
        assert philox4x32_10(counter, key) == expected


def test_gumbel_approximation_is_monotonic() -> None:
    torch = pytest.importorskip("torch")
    words = torch.tensor(
        [0, 1, 100, 1 << 16, 1 << 24, (1 << 32) - 2, (1 << 32) - 1],
        dtype=torch.int64,
    )
    scores = approximate_gumbel_q_from_words(words)
    assert bool((scores[1:] >= scores[:-1]).all().item())
    factorized = approximate_factorized_gumbel_q_from_words(words)
    assert bool((factorized[1:] >= factorized[:-1]).all().item())


def test_rng_and_gumbel_small_validation_passes() -> None:
    pytest.importorskip("torch")
    result = validate_rng_and_gumbel(
        distribution_trials=20_000,
        distribution_vocabulary_size=8,
        full_vocabulary_size=257,
        full_vocabulary_batches=2,
        full_vocabulary_batch_size=8,
        mantissa_bits=[4, 8],
        maximum_distribution_tv=0.03,
        minimum_full_vocabulary_agreement=0.9,
    )
    assert result["philox4x32_10"]["passed"] is True
    assert result["distribution_validation"]["passed"] is True
    assert result["full_vocabulary_stress"]["selected_passed"] is True
    assert result["passed"] is True
