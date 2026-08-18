import pytest

from diffusion_accel.rng_hardware import analyze_rng_hardware


def test_four_iterative_cores_clear_output_head_lower_bound() -> None:
    result = analyze_rng_hardware(
        iterative_core_usage={
            "dsp_primitives": 16,
            "lut_primitives": 712,
            "flip_flop_primitives": 326,
        },
        four_core_farm_usage={
            "dsp_primitives": 64,
            "lut_primitives": 3_146,
            "flip_flop_primitives": 1_584,
        },
        dual_gumbel_usage={
            "dsp_primitives": 2,
            "lut_primitives": 988,
            "flip_flop_primitives": 82,
            "bram_primitives": 2,
        },
        integrated_stream_usage={
            "dsp_primitives": 66,
            "lut_primitives": 4_136,
            "flip_flop_primitives": 1_880,
            "distributed_ram_primitives": 14,
            "bram_primitives": 2,
        },
    )

    assert result["workload"]["score_count"] == 64 * 50_258
    assert result["workload"]["required_rng_scores_per_cycle"] == pytest.approx(
        4 / 3
    )
    assert result["unrolled_stream"]["output_head_bottleneck"] is True
    assert result["iterative_farm"]["minimum_non_bottleneck_cores"] == 4
    points = result["iterative_farm"]["design_points"]
    assert points[2]["output_head_bottleneck"] is True
    assert points[3]["output_head_bottleneck"] is False
    assert points[3]["projected_core_only_resources"]["dsp_primitives"] == 64
    combined = result["iterative_farm"][
        "mapped_farm_plus_dual_gumbel_resources"
    ]
    assert combined["dsp_primitives"] == 66
    assert combined["lut_primitives"] == 4_134
    assert combined["bram_primitives"] == 2
    integrated = result["iterative_farm"]["mapped_integrated_stream_resources"]
    assert integrated["lut_primitives"] == 4_136
    assert integrated["distributed_ram_primitives"] == 14


def test_rng_hardware_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="positions must be positive"):
        analyze_rng_hardware(positions=0)
