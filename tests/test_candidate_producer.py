import pytest

from diffusion_accel.candidate_producer import (
    analyze_output_head_lane_sweep,
    streaming_exponential_race_candidates,
    validate_streaming_candidate_producer,
)


def test_output_head_sweep_tracks_bandwidth_crossover() -> None:
    result = analyze_output_head_lane_sweep(
        [8, 4, 2],
        vocabulary_size=32,
        hidden_size=16,
        weight_bits=[16],
        mac_lanes=[8, 16],
        clock_mhz=100,
        ddr_bandwidth_gbps=1,
        ddr_efficiency=1,
    )

    assert result["schedule"]["active_token_positions"] == 14
    points = result["design_points"]
    assert len(points) == 2
    assert points[1]["compute_only_total_ms"] < points[0]["compute_only_total_ms"]
    assert points[1]["memory_only_total_ms"] == points[0]["memory_only_total_ms"]


def test_streaming_candidate_matches_probability_sampler_across_chunks() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn((3, 19), generator=generator, dtype=torch.float64)
    uniforms = torch.rand((3, 19), generator=generator, dtype=torch.float64)
    masked = logits.clone()
    masked[:, 18] = -torch.inf
    exponential = 1e-10 - torch.log(uniforms + 1e-10)
    reference = (torch.softmax(masked, dim=-1) / exponential).argmax(dim=-1)

    for chunk_size in (1, 4, 19, 32):
        actual = streaming_exponential_race_candidates(
            logits,
            uniforms,
            mask_token_id=18,
            chunk_size=chunk_size,
        )
        assert actual.equal(reference)


def test_streaming_candidate_output_head_model_is_conservative() -> None:
    pytest.importorskip("torch")
    result = validate_streaming_candidate_producer(
        positions=4,
        vocabulary_size=32,
        hidden_size=16,
        trials=3,
        chunk_size=7,
        mac_lanes=8,
    )

    assert result["passed"] is True
    assert result["pathwise_validation"]["agreement"] == 1.0
    assert result["state"]["candidate_cache_bytes"] == 6
    assert result["output_projection"]["ddr_bytes_removed"] > 0


def test_streaming_candidate_rejects_invalid_uniforms() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 4), dtype=torch.float64)
    uniforms = torch.ones((1, 4), dtype=torch.float64)
    with pytest.raises(ValueError, match="uniforms"):
        streaming_exponential_race_candidates(
            logits,
            uniforms,
            mask_token_id=3,
            chunk_size=2,
        )
