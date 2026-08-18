from pathlib import Path

from diffusion_accel.realtime_target import analyze_realtime_target


ROOT = Path(__file__).resolve().parents[1]


def test_realtime_target_exposes_honest_latency_contract() -> None:
    report = analyze_realtime_target(
        ROOT / "configs/targets/mdlm_owt_169m_realtime.yaml", ROOT
    )

    assert report["full_forward_macs"] == 7_988_920_320
    assert report["baseline"]["model_evaluations"] == 41
    assert report["latency_contract"]["target_generation_ms"] == 500.0

    commit, stretch = report["design_points"]
    assert commit["weight_bytes"] == 123_714_130
    assert commit["meets_target"] is True
    assert commit["meets_stretch"] is False
    assert 460.0 < commit["estimated_generation_lower_bound_ms"] < 480.0
    assert stretch["meets_target"] is True
    assert stretch["meets_stretch"] is True
    assert 190.0 < stretch["estimated_generation_lower_bound_ms"] < 200.0
