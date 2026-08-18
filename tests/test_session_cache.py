import pytest

from diffusion_accel.session_cache import (
    ConversationTurn,
    SessionCacheConfig,
    analyze_session_cache,
)


def _small_config() -> SessionCacheConfig:
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


def test_full_mha_kv_size_counts_keys_values_and_layers() -> None:
    config = _small_config()
    assert config.kv_bytes_per_token == 2 * 2 * 64 * 2


def test_two_turn_cache_analysis_includes_diffusion_finalization() -> None:
    result = analyze_session_cache(
        [ConversationTurn(128, 256), ConversationTurn(32, 128)],
        _small_config(),
    )
    totals = result["totals_for_observed_turns"]
    assert totals["full_attention_recompute_prefill_tokens"] == 544
    assert totals["prefix_cache_prefill_tokens"] == 160
    assert totals["prefix_cache_answer_finalization_tokens"] == 256
    assert totals["prefix_cache_conditioning_work_tokens"] == 416
    assert totals["conditioning_work_tokens_saved"] == 128


def test_terminal_kv_removes_finalization_work() -> None:
    result = analyze_session_cache(
        [ConversationTurn(128, 256), ConversationTurn(32, 128)],
        _small_config(),
        terminal_kv_available=True,
    )
    totals = result["totals_for_observed_turns"]
    assert totals["prefix_cache_answer_finalization_tokens"] == 0
    assert totals["conditioning_work_tokens_saved"] == 384


def test_invalid_attention_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SessionCacheConfig(hidden_size=65)
