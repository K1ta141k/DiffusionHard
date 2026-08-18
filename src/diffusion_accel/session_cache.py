"""Analytical multi-turn K/V cache experiment.

This module deliberately separates correctness from capacity.  A full-attention
diffusion model must recompute old token state after a follow-up is appended.
A prefix-isolated model may retain completed-turn K/V, but may need one final
forward pass to materialize exact K/V for a just-denoised answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence


@dataclass(frozen=True)
class ConversationTurn:
    user_tokens: int
    answer_tokens: int

    def __post_init__(self) -> None:
        if self.user_tokens <= 0:
            raise ValueError("user_tokens must be positive")
        if self.answer_tokens <= 0:
            raise ValueError("answer_tokens must be positive")


@dataclass(frozen=True)
class SessionCacheConfig:
    layers: int = 32
    hidden_size: int = 4096
    attention_heads: int = 32
    kv_heads: int = 32
    kv_bits: int = 16
    parameter_count: int = 8_000_000_000
    weight_bits: int = 4
    ddr_bytes: int = 4 * 1024**3
    runtime_reserve_bytes: int = 512 * 1024**2

    def __post_init__(self) -> None:
        integer_fields = asdict(self)
        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError("%s must be positive" % name)
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.kv_heads > self.attention_heads:
            raise ValueError("kv_heads cannot exceed attention_heads")
        if self.kv_bits % 8:
            raise ValueError("kv_bits must be byte aligned")
        if self.weight_bits % 2:
            raise ValueError("weight_bits must be divisible by two")

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.attention_heads

    @property
    def kv_bytes_per_token(self) -> int:
        # Two tensors (K and V) at every layer.
        return (
            2
            * self.layers
            * self.kv_heads
            * self.head_dimension
            * (self.kv_bits // 8)
        )

    @property
    def raw_weight_bytes(self) -> int:
        return self.parameter_count * self.weight_bits // 8

    @property
    def session_cache_budget_bytes(self) -> int:
        return max(
            0,
            self.ddr_bytes - self.raw_weight_bytes - self.runtime_reserve_bytes,
        )


def _turn_record(
    *,
    turn_index: int,
    turn: ConversationTurn,
    history_before: int,
    config: SessionCacheConfig,
    terminal_kv_available: bool,
) -> Dict[str, object]:
    history_after = history_before + turn.user_tokens + turn.answer_tokens
    cache_bytes = history_after * config.kv_bytes_per_token
    finalization_tokens = 0 if terminal_kv_available else turn.answer_tokens
    # If all session K/V would otherwise be dropped, eager finalization is worth
    # its compute when a follow-up arrives often enough to repay that pass.
    break_even = finalization_tokens / history_after
    budget = config.session_cache_budget_bytes
    return {
        "turn": turn_index,
        "user_tokens": turn.user_tokens,
        "answer_tokens": turn.answer_tokens,
        "history_tokens_before_request": history_before,
        "history_tokens_after_answer": history_after,
        "full_attention_exact_prefill_tokens": history_before + turn.user_tokens,
        "prefix_cache_prefill_tokens": turn.user_tokens,
        "answer_finalization_tokens_if_retained": finalization_tokens,
        "session_kv_bytes_after_answer": cache_bytes,
        "session_kv_mib_after_answer": cache_bytes / 1024**2,
        "cache_fits_available_ddr": cache_bytes <= budget,
        "break_even_followup_probability_ignoring_memory_pressure": break_even,
    }


def analyze_session_cache(
    turns: Sequence[ConversationTurn],
    config: SessionCacheConfig,
    *,
    terminal_kv_available: bool = False,
) -> Dict[str, object]:
    """Compare exact recomputation with exact prefix-isolated session caching."""
    if not turns:
        raise ValueError("at least one conversation turn is required")

    records = []
    history = 0
    recompute_prefill = 0
    prefix_prefill = 0
    observed_finalization = 0
    for index, turn in enumerate(turns, start=1):
        record = _turn_record(
            turn_index=index,
            turn=turn,
            history_before=history,
            config=config,
            terminal_kv_available=terminal_kv_available,
        )
        records.append(record)
        recompute_prefill += history + turn.user_tokens
        prefix_prefill += turn.user_tokens
        # Only answers preceding an observed follow-up had to be finalized to
        # produce the already-observed result. The last answer is a retention
        # decision for a possible future request.
        if index < len(turns):
            observed_finalization += int(
                record["answer_finalization_tokens_if_retained"]
            )
        history += turn.user_tokens + turn.answer_tokens

    raw_footprint = config.raw_weight_bytes + config.runtime_reserve_bytes
    model_fits = raw_footprint <= config.ddr_bytes
    final_cache_bytes = history * config.kv_bytes_per_token
    cache_budget = config.session_cache_budget_bytes
    sessions_at_final_size = (
        cache_budget // final_cache_bytes if final_cache_bytes else 0
    )
    prefix_observed_work = prefix_prefill + observed_finalization

    return {
        "configuration": {
            **asdict(config),
            "head_dimension": config.head_dimension,
            "kv_bytes_per_token": config.kv_bytes_per_token,
            "kv_kib_per_token": config.kv_bytes_per_token / 1024,
            "raw_weight_bytes": config.raw_weight_bytes,
            "raw_weight_gib": config.raw_weight_bytes / 1024**3,
            "session_cache_budget_bytes": cache_budget,
            "session_cache_budget_mib": cache_budget / 1024**2,
            "raw_model_plus_reserve_fits_ddr": model_fits,
        },
        "correctness": {
            "full_attention": (
                "retained KV is invalid after appending a follow-up; recompute"
            ),
            "prefix_isolated": (
                "completed-prefix KV is exact and may be retained"
            ),
            "terminal_kv_available": terminal_kv_available,
        },
        "turns": records,
        "totals_for_observed_turns": {
            "full_attention_recompute_prefill_tokens": recompute_prefill,
            "prefix_cache_prefill_tokens": prefix_prefill,
            "prefix_cache_answer_finalization_tokens": observed_finalization,
            "prefix_cache_conditioning_work_tokens": prefix_observed_work,
            "conditioning_work_tokens_saved": (
                recompute_prefill - prefix_observed_work
            ),
            "conditioning_work_reduction_fraction": (
                (recompute_prefill - prefix_observed_work) / recompute_prefill
            ),
        },
        "capacity": {
            "final_history_tokens": history,
            "final_session_kv_bytes": final_cache_bytes,
            "final_session_kv_mib": final_cache_bytes / 1024**2,
            "final_cache_fits_available_ddr": final_cache_bytes <= cache_budget,
            "whole_sessions_at_final_size": sessions_at_final_size,
        },
    }
