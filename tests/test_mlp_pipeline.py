from diffusion_accel.mlp_pipeline import analyze_mlp_pingpong


def test_mdlm_pingpong_hides_weight_loads() -> None:
    report = analyze_mlp_pingpong()
    up, down = report["layers"]

    assert up["compute_cycles_per_output_tile"] == 384
    assert down["compute_cycles_per_output_tile"] == 1536
    assert up["stream_adapter_cycles_per_output_tile"] == 72
    assert down["stream_adapter_cycles_per_output_tile"] == 288
    assert up["pingpong_weight_buffer_bytes"] == 9_216
    assert down["pingpong_weight_buffer_bytes"] == 36_864
    assert up["overlap_spill_cycles_per_output_tile"] == 0
    assert down["overlap_spill_cycles_per_output_tile"] == 0
    assert report["summary"]["all_weight_loads_hidden_after_initial_tile"] is True
    assert report["summary"]["saved_latency_ms_for_generation"] > 30.0


def test_pingpong_model_rejects_invalid_configuration() -> None:
    try:
        analyze_mlp_pingpong(output_lanes=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid output lane count was accepted")
