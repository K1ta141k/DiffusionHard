import pytest

from diffusion_accel.candidate_cache import (
    analyze_candidate_reveal_kernel,
    validate_candidate_cache_equivalence,
)


def test_candidate_factorization_is_exact_and_storage_is_compact() -> None:
    pytest.importorskip("torch")
    result = validate_candidate_cache_equivalence(
        positions=8,
        vocabulary_size=128,
        analytic_vocabulary_size=17,
        transitions=4,
        monte_carlo_trials=50_000,
        monte_carlo_vocabulary_size=4,
        maximum_total_variation=0.02,
        seed=3,
    )

    assert result["analytic_factorization"]["passed"] is True
    assert result["analytic_factorization"][
        "maximum_absolute_probability_error"
    ] < 1e-12
    assert result["monte_carlo"]["passed"] is True
    assert result["storage"]["candidate_cache_bytes"] == 10
    assert result["storage"]["candidate_id_bits"] == 7
    assert result["storage"]["full_probability_cache_bytes"] == 2048
    assert result["passed"] is True


def test_candidate_validation_rejects_non_byte_aligned_probability_storage() -> None:
    with pytest.raises(ValueError, match="byte aligned"):
        validate_candidate_cache_equivalence(probability_bits=7)


def test_candidate_kernel_cycle_model_counts_stream_and_invalidation() -> None:
    result = analyze_candidate_reveal_kernel(
        positions=64,
        vocabulary_size=50_258,
        cache_hit_transitions=23,
        clock_mhz=300.0,
        initiation_interval=1,
        measured_model_forward_ms=776.8,
    )

    assert result["cycles"]["total_cycles_per_hit"] == 66
    assert result["cycles"]["total_cycles_all_hits"] == 1518
    assert result["cycles"]["total_latency_us"] == 5.06
    assert result["traffic"]["stream_bytes_per_hit"] == 680
