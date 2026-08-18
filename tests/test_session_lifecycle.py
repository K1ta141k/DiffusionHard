from diffusion_accel.session_cache import SessionCacheConfig
from diffusion_accel.session_lifecycle import (
    SessionRequest,
    analyze_and_trace_session_lifecycle,
    analyze_session_lifecycle,
)
from diffusion_accel.simulator import (
    AllHBMPolicy,
    ComputeConfig,
    HardwareConfig,
    MemoryConfig,
    SramConfig,
    simulate,
)


def _config() -> SessionCacheConfig:
    return SessionCacheConfig(
        layers=2,
        hidden_size=64,
        attention_heads=4,
        kv_heads=4,
        kv_bits=16,
        parameter_count=1_000,
        weight_bits=8,
        ddr_bytes=1_000_000,
        runtime_reserve_bytes=1_000,
    )


def _hardware() -> HardwareConfig:
    return HardwareConfig(
        name="traffic-test",
        compute=ComputeConfig(peak_tops=1.0, utilization=1.0),
        hbm=MemoryConfig(bandwidth_gb_s=1.0, efficiency=1.0),
        sram=SramConfig(
            capacity_bytes=1_000_000,
            bandwidth_gb_s=10.0,
            efficiency=1.0,
        ),
    )


def test_lifecycle_hit_then_ttl_expiry_preserves_transcript() -> None:
    requests = [
        SessionRequest("a", 0.0, 10, 10, 2),
        SessionRequest("a", 5.0, 5, 5, 2),
        SessionRequest("a", 16.0, 5, 5, 2),
    ]
    result = analyze_session_lifecycle(
        requests,
        _config(),
        ttl_s=10.0,
        capacity_bytes=100 * _config().kv_bytes_per_token,
    )

    aggregate = result["aggregate"]
    assert aggregate["followup_requests"] == 2
    assert aggregate["cache_hits"] == 1
    assert aggregate["cache_expirations"] == 1
    assert aggregate["baseline_conditioning_tokens"] == 70
    assert aggregate["cache_path_prefill_tokens"] == 50
    assert aggregate["answer_finalization_tokens"] == 20
    assert aggregate["cache_path_conditioning_work_tokens"] == 70
    assert aggregate["baseline_model_token_positions"] == 180
    assert aggregate["request_local_cache_model_token_positions"] == 110
    assert aggregate["cache_path_model_token_positions"] == 110
    assert aggregate["cross_request_model_token_positions_saved"] == 0
    assert aggregate["terminal_finalization_tokens_without_observed_followup"] == 5
    assert aggregate[
        "observed_cache_model_token_positions_excluding_terminal_finalization"
    ] == 105
    assert aggregate["observed_cross_request_model_token_positions_saved"] == 5
    assert result["requests"][2]["cache_miss_reason"] == "expired"
    assert result["requests"][2]["history_tokens_before_request"] == 30


def test_lru_hit_updates_recency_and_evicts_other_session() -> None:
    token_bytes = _config().kv_bytes_per_token
    result = analyze_session_lifecycle(
        [
            SessionRequest("a", 0.0, 10, 10, 1),
            SessionRequest("b", 1.0, 10, 10, 1),
            SessionRequest("a", 2.0, 1, 1, 1),
            SessionRequest("b", 3.0, 1, 1, 1),
        ],
        _config(),
        capacity_bytes=40 * token_bytes,
    )

    assert result["requests"][2]["cache_hit"] is True
    assert result["requests"][2]["evicted_session_ids"] == ["b"]
    assert result["requests"][3]["cache_miss_reason"] == "lru_evicted"
    assert result["aggregate"]["cache_evictions"] == 2


def test_oversized_session_is_recomputed_without_finalization() -> None:
    token_bytes = _config().kv_bytes_per_token
    result = analyze_session_lifecycle(
        [
            SessionRequest("a", 0.0, 10, 10, 1),
            SessionRequest("a", 1.0, 2, 2, 1),
        ],
        _config(),
        capacity_bytes=15 * token_bytes,
    )

    assert result["aggregate"]["cache_hits"] == 0
    assert result["aggregate"]["oversized_cache_rejections"] == 2
    assert result["aggregate"]["answer_finalization_tokens"] == 0
    assert result["requests"][1]["cache_miss_reason"] == "oversized"


def test_lowered_trace_reproduces_modeled_all_hbm_kv_traffic() -> None:
    result, trace = analyze_and_trace_session_lifecycle(
        [
            SessionRequest("a", 0.0, 10, 10, 2),
            SessionRequest("a", 1.0, 5, 5, 3),
        ],
        _config(),
        capacity_bytes=100 * _config().kv_bytes_per_token,
    )
    simulation = simulate(trace, _hardware(), AllHBMPolicy())

    assert len(trace.steps) == 2
    assert trace.steps[-1].metadata["request_end"] is True
    assert simulation.hbm_bytes == result["aggregate"]["modeled_kv_traffic_bytes"]


def test_terminal_kv_trace_still_accounts_for_persistent_answer_write() -> None:
    result, trace = analyze_and_trace_session_lifecycle(
        [SessionRequest("a", 0.0, 10, 10, 2)],
        _config(),
        capacity_bytes=100 * _config().kv_bytes_per_token,
        terminal_kv_available=True,
    )
    simulation = simulate(trace, _hardware(), AllHBMPolicy())

    assert result["aggregate"]["answer_finalization_tokens"] == 0
    assert simulation.hbm_bytes == result["aggregate"]["modeled_kv_traffic_bytes"]
