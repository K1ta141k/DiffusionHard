"""Bounded multi-session K/V lifecycle and traffic trace generation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .ir import DiffusionStep, Operation, TensorAccess, WorkloadTrace
from .session_cache import SessionCacheConfig
from .trace import SCHEMA_VERSION


@dataclass(frozen=True)
class SessionRequest:
    """One timestamped generation request in a conversation session."""

    session_id: str
    arrival_s: float
    user_tokens: int
    answer_tokens: int
    model_evaluations: int

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.arrival_s < 0:
            raise ValueError("arrival_s must be non-negative")
        if min(self.user_tokens, self.answer_tokens, self.model_evaluations) <= 0:
            raise ValueError("token counts and model_evaluations must be positive")


@dataclass(frozen=True)
class _CacheEntry:
    tokens: int
    size_bytes: int
    last_access_s: float


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def analyze_session_lifecycle(
    requests: Sequence[SessionRequest],
    config: SessionCacheConfig,
    *,
    ttl_s: float = 300.0,
    capacity_bytes: Optional[int] = None,
    terminal_kv_available: bool = False,
) -> Dict[str, object]:
    """Replay requests through an exact prefix-safe TTL/LRU K/V cache.

    Transcript token counts survive K/V eviction. A cache miss therefore
    recomputes the completed transcript, while a hit materializes only the new
    user block. The cache is external-DDR state; active answer K/V remains a
    request-local working set.
    """
    if not requests:
        raise ValueError("at least one session request is required")
    if ttl_s <= 0:
        raise ValueError("ttl_s must be positive")
    previous_arrival = -1.0
    for request in requests:
        if request.arrival_s < previous_arrival:
            raise ValueError("requests must be ordered by nondecreasing arrival_s")
        previous_arrival = request.arrival_s

    capacity = (
        config.session_cache_budget_bytes
        if capacity_bytes is None
        else int(capacity_bytes)
    )
    if capacity < 0:
        raise ValueError("capacity_bytes must be non-negative")

    entries: OrderedDict[str, _CacheEntry] = OrderedDict()
    transcript_tokens: Dict[str, int] = {}
    removal_reason: Dict[str, str] = {}
    occupancy = 0
    peak_occupancy = 0
    records: List[Dict[str, object]] = []
    cache_hits = 0
    followup_requests = 0
    evictions = 0
    expirations = 0
    oversized_rejections = 0

    baseline_conditioning_tokens = 0
    cached_conditioning_tokens = 0
    finalization_tokens = 0
    baseline_model_token_positions = 0
    request_local_cache_model_token_positions = 0
    cache_path_model_token_positions = 0
    context_kv_read_bytes = 0
    context_kv_write_bytes = 0
    active_kv_write_bytes = 0

    bytes_per_token = config.kv_bytes_per_token
    for request_index, request in enumerate(requests):
        expired_session_ids = []
        for session_id, entry in list(entries.items()):
            if request.arrival_s >= entry.last_access_s + ttl_s:
                expired_session_ids.append(session_id)
                occupancy -= entry.size_bytes
                del entries[session_id]
                removal_reason[session_id] = "expired"
                expirations += 1

        history_before = transcript_tokens.get(request.session_id, 0)
        if history_before:
            followup_requests += 1
        entry = entries.pop(request.session_id, None)
        cache_hit = entry is not None
        if cache_hit:
            miss_reason = None
        elif history_before == 0:
            miss_reason = "cold"
        else:
            miss_reason = removal_reason.get(request.session_id, "not_resident")
        if cache_hit:
            if entry.tokens != history_before:
                raise RuntimeError("cached token count does not match transcript")
            occupancy -= entry.size_bytes
            cache_hits += 1

        baseline_tokens = history_before + request.user_tokens
        conditioned_tokens = request.user_tokens if cache_hit else baseline_tokens
        history_after = baseline_tokens + request.answer_tokens
        final_cache_bytes = history_after * bytes_per_token
        retained = capacity > 0 and final_cache_bytes <= capacity
        answer_finalization_tokens = (
            request.answer_tokens
            if retained and not terminal_kv_available
            else 0
        )

        followup_materialization_read = (
            history_before * bytes_per_token if cache_hit else 0
        )
        denoising_read = (
            baseline_tokens * bytes_per_token * request.model_evaluations
        )
        finalization_read = (
            baseline_tokens * bytes_per_token
            if answer_finalization_tokens
            else 0
        )
        request_context_reads = (
            followup_materialization_read + denoising_read + finalization_read
        )
        conditioned_write_tokens = request.user_tokens if cache_hit else baseline_tokens
        request_context_writes = (
            conditioned_write_tokens
            + (request.answer_tokens if retained else 0)
        ) * bytes_per_token
        request_active_writes = (
            request.answer_tokens
            * bytes_per_token
            * request.model_evaluations
        )
        request_baseline_model_positions = (
            baseline_tokens + request.answer_tokens
        ) * request.model_evaluations
        request_local_model_positions = (
            baseline_tokens
            + request.answer_tokens * request.model_evaluations
        )
        request_cache_model_positions = (
            conditioned_tokens
            + request.answer_tokens * request.model_evaluations
            + answer_finalization_tokens
        )

        evicted_session_ids = []
        if retained:
            while entries and occupancy + final_cache_bytes > capacity:
                evicted_id, evicted_entry = entries.popitem(last=False)
                occupancy -= evicted_entry.size_bytes
                evicted_session_ids.append(evicted_id)
                removal_reason[evicted_id] = "lru_evicted"
                evictions += 1
            entries[request.session_id] = _CacheEntry(
                tokens=history_after,
                size_bytes=final_cache_bytes,
                last_access_s=request.arrival_s,
            )
            occupancy += final_cache_bytes
            removal_reason.pop(request.session_id, None)
        else:
            removal_reason[request.session_id] = "oversized"
            oversized_rejections += 1

        transcript_tokens[request.session_id] = history_after
        peak_occupancy = max(peak_occupancy, occupancy)
        baseline_conditioning_tokens += baseline_tokens
        cached_conditioning_tokens += conditioned_tokens
        finalization_tokens += answer_finalization_tokens
        baseline_model_token_positions += request_baseline_model_positions
        request_local_cache_model_token_positions += request_local_model_positions
        cache_path_model_token_positions += request_cache_model_positions
        context_kv_read_bytes += request_context_reads
        context_kv_write_bytes += request_context_writes
        active_kv_write_bytes += request_active_writes

        records.append(
            {
                "request": request_index,
                "session_id": request.session_id,
                "arrival_s": request.arrival_s,
                "user_tokens": request.user_tokens,
                "answer_tokens": request.answer_tokens,
                "model_evaluations": request.model_evaluations,
                "history_tokens_before_request": history_before,
                "history_tokens_after_answer": history_after,
                "cache_hit": cache_hit,
                "cache_miss_reason": miss_reason,
                "retained_after_request": retained,
                "expired_session_ids": expired_session_ids,
                "evicted_session_ids": evicted_session_ids,
                "baseline_conditioning_tokens": baseline_tokens,
                "cache_path_conditioning_tokens": conditioned_tokens,
                "answer_finalization_tokens": answer_finalization_tokens,
                "baseline_model_token_positions": (
                    request_baseline_model_positions
                ),
                "request_local_cache_model_token_positions": (
                    request_local_model_positions
                ),
                "cache_path_model_token_positions": request_cache_model_positions,
                "followup_materialization_context_read_bytes": (
                    followup_materialization_read
                ),
                "denoising_context_read_bytes": denoising_read,
                "finalization_context_read_bytes": finalization_read,
                "context_kv_read_bytes": request_context_reads,
                "context_kv_write_bytes": request_context_writes,
                "active_kv_write_bytes": request_active_writes,
                "session_cache_bytes_after_request": occupancy,
                "resident_session_ids_after_request": list(entries),
            }
        )

    cache_conditioning_work = cached_conditioning_tokens + finalization_tokens
    saved_conditioning_work = (
        baseline_conditioning_tokens - cache_conditioning_work
    )
    saved_model_positions = (
        baseline_model_token_positions - cache_path_model_token_positions
    )
    cross_request_saved_positions = (
        request_local_cache_model_token_positions
        - cache_path_model_token_positions
    )
    final_request_by_session = {
        str(record["session_id"]): int(record["request"])
        for record in records
    }
    terminal_finalization_tokens = sum(
        int(record["answer_finalization_tokens"])
        for record in records
        if int(record["request"])
        == final_request_by_session[str(record["session_id"])]
    )
    observed_cache_model_positions = (
        cache_path_model_token_positions - terminal_finalization_tokens
    )
    observed_cross_request_saved_positions = (
        request_local_cache_model_token_positions - observed_cache_model_positions
    )
    traffic_bytes = (
        context_kv_read_bytes + context_kv_write_bytes + active_kv_write_bytes
    )
    return {
        "configuration": {
            "ttl_s": ttl_s,
            "capacity_bytes": capacity,
            "capacity_mib": capacity / 1024**2,
            "kv_bytes_per_token": bytes_per_token,
            "terminal_kv_available": terminal_kv_available,
            "attention_contract": "completed-block-causal",
        },
        "requests": records,
        "aggregate": {
            "request_count": len(requests),
            "distinct_sessions": len(transcript_tokens),
            "followup_requests": followup_requests,
            "cache_hits": cache_hits,
            "followup_cache_hit_fraction": _fraction(
                cache_hits, followup_requests
            ),
            "cache_evictions": evictions,
            "cache_expirations": expirations,
            "oversized_cache_rejections": oversized_rejections,
            "peak_session_cache_bytes": peak_occupancy,
            "peak_session_cache_mib": peak_occupancy / 1024**2,
            "resident_sessions_at_end": len(entries),
            "baseline_conditioning_tokens": baseline_conditioning_tokens,
            "cache_path_prefill_tokens": cached_conditioning_tokens,
            "answer_finalization_tokens": finalization_tokens,
            "cache_path_conditioning_work_tokens": cache_conditioning_work,
            "conditioning_work_tokens_saved": saved_conditioning_work,
            "conditioning_work_reduction_fraction": _fraction(
                saved_conditioning_work, baseline_conditioning_tokens
            ),
            "baseline_model_token_positions": baseline_model_token_positions,
            "request_local_cache_model_token_positions": (
                request_local_cache_model_token_positions
            ),
            "cache_path_model_token_positions": cache_path_model_token_positions,
            "model_token_positions_saved": saved_model_positions,
            "model_token_position_reduction_fraction": _fraction(
                saved_model_positions, baseline_model_token_positions
            ),
            "cross_request_model_token_positions_saved": (
                cross_request_saved_positions
            ),
            "cross_request_model_token_position_reduction_fraction": _fraction(
                cross_request_saved_positions,
                request_local_cache_model_token_positions,
            ),
            "terminal_finalization_tokens_without_observed_followup": (
                terminal_finalization_tokens
            ),
            "observed_cache_model_token_positions_excluding_terminal_finalization": (
                observed_cache_model_positions
            ),
            "observed_cross_request_model_token_positions_saved": (
                observed_cross_request_saved_positions
            ),
            "observed_cross_request_model_token_position_reduction_fraction": (
                _fraction(
                    observed_cross_request_saved_positions,
                    request_local_cache_model_token_positions,
                )
            ),
            "context_kv_read_bytes": context_kv_read_bytes,
            "context_kv_write_bytes": context_kv_write_bytes,
            "active_kv_write_bytes": active_kv_write_bytes,
            "modeled_kv_traffic_bytes": traffic_bytes,
        },
    }


def lower_session_lifecycle_trace(
    lifecycle: Dict[str, object],
    config: SessionCacheConfig,
) -> WorkloadTrace:
    """Lower a lifecycle replay into a K/V-only hardware traffic trace."""
    steps: List[DiffusionStep] = []
    records = lifecycle["requests"]
    if not isinstance(records, list):
        raise ValueError("lifecycle requests must be a list")
    bytes_per_token = config.kv_bytes_per_token

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("lifecycle request records must be mappings")
        request_index = int(record["request"])
        session_id = str(record["session_id"])
        answer_tokens = int(record["answer_tokens"])
        model_evaluations = int(record["model_evaluations"])
        baseline_tokens = int(record["baseline_conditioning_tokens"])
        cache_hit = bool(record["cache_hit"])
        retained = bool(record["retained_after_request"])
        operations: List[Operation] = []

        conditioned_tokens = int(record["cache_path_conditioning_tokens"])
        if conditioned_tokens:
            operations.append(
                Operation(
                    name="request_%03d_context_materialization" % request_index,
                    flops=0,
                    writes=[
                        TensorAccess(
                            name="request_%03d_conditioned_kv" % request_index,
                            size_bytes=conditioned_tokens * bytes_per_token,
                            access="write",
                            lifetime="session" if retained else "request",
                            category="session_kv",
                        )
                    ],
                    metadata={
                        "phase": "context-materialization",
                        "cache_hit": cache_hit,
                        "tokens": conditioned_tokens,
                    },
                )
            )

        context_bytes = baseline_tokens * bytes_per_token
        active_bytes = answer_tokens * bytes_per_token
        context_layer_bytes = context_bytes // config.layers
        active_layer_bytes = active_bytes // config.layers
        for evaluation in range(model_evaluations):
            for layer in range(config.layers):
                operations.append(
                    Operation(
                        name="request_%03d_denoise_%03d_layer_%02d" % (
                            request_index,
                            evaluation,
                            layer,
                        ),
                        flops=0,
                        reads=[
                            TensorAccess(
                                name=(
                                    "request_%03d_context_kv_read_%03d_layer_%02d"
                                    % (request_index, evaluation, layer)
                                ),
                                size_bytes=context_layer_bytes,
                                access="read",
                                lifetime="operation",
                                category="session_kv",
                            )
                        ],
                        writes=[
                            TensorAccess(
                                name=(
                                    "request_%03d_active_kv_%03d_layer_%02d"
                                    % (request_index, evaluation, layer)
                                ),
                                size_bytes=active_layer_bytes,
                                access="write",
                                lifetime="operation",
                                category="canvas_kv",
                            )
                        ],
                        parallelism=answer_tokens,
                        metadata={
                            "phase": "answer-denoising",
                            "model_evaluation": evaluation,
                            "layer": layer,
                        },
                    )
                )

        finalization_tokens = int(record["answer_finalization_tokens"])
        if finalization_tokens:
            operations.append(
                Operation(
                    name="request_%03d_answer_finalization" % request_index,
                    flops=0,
                    reads=[
                        TensorAccess(
                            name="request_%03d_finalization_context" % request_index,
                            size_bytes=context_bytes,
                            access="read",
                            lifetime="operation",
                            category="session_kv",
                        )
                    ],
                    writes=[
                        TensorAccess(
                            name="request_%03d_final_answer_kv" % request_index,
                            size_bytes=finalization_tokens * bytes_per_token,
                            access="write",
                            lifetime="session",
                            category="session_kv",
                        )
                    ],
                    parallelism=finalization_tokens,
                    metadata={"phase": "answer-finalization"},
                )
            )
        elif retained and answer_tokens:
            operations.append(
                Operation(
                    name="request_%03d_retain_terminal_answer_kv" % request_index,
                    flops=0,
                    writes=[
                        TensorAccess(
                            name="request_%03d_terminal_answer_kv" % request_index,
                            size_bytes=answer_tokens * bytes_per_token,
                            access="write",
                            lifetime="session",
                            category="session_kv",
                        )
                    ],
                    parallelism=answer_tokens,
                    metadata={"phase": "terminal-answer-retention"},
                )
            )

        if cache_hit:
            prior_history = int(record["history_tokens_before_request"])
            operations.insert(
                0,
                Operation(
                    name="request_%03d_followup_materialization" % request_index,
                    flops=0,
                    reads=[
                        TensorAccess(
                            name="request_%03d_prior_session_kv" % request_index,
                            size_bytes=prior_history * bytes_per_token,
                            access="read",
                            lifetime="operation",
                            category="session_kv",
                        )
                    ],
                    metadata={"phase": "followup-materialization"},
                ),
            )

        steps.append(
            DiffusionStep(
                canvas_id=request_index,
                step_id=request_index,
                active_tokens=answer_tokens,
                changed_tokens=answer_tokens,
                operations=operations,
                metadata={
                    "request_end": True,
                    "session_id": session_id,
                    "arrival_s": float(record["arrival_s"]),
                    "cache_hit": cache_hit,
                    "retained_after_request": retained,
                    "evicted_session_ids": record["evicted_session_ids"],
                    "expired_session_ids": record["expired_session_ids"],
                },
            )
        )

    metadata = {
        "provenance": "derived-session-kv-traffic",
        "scope": "kv-traffic-only-no-weight-or-end-to-end-compute-model",
        "layers": config.layers,
        "hidden_size": config.hidden_size,
        "kv_heads": config.kv_heads,
        "kv_bits": config.kv_bits,
        "kv_bytes_per_token": bytes_per_token,
        "lifecycle_configuration": lifecycle["configuration"],
        "lifecycle_aggregate": lifecycle["aggregate"],
    }
    return WorkloadTrace(
        schema_version=SCHEMA_VERSION,
        workload_name="session-kv-lifecycle",
        steps=steps,
        metadata=metadata,
    )


def analyze_and_trace_session_lifecycle(
    requests: Sequence[SessionRequest],
    config: SessionCacheConfig,
    *,
    ttl_s: float = 300.0,
    capacity_bytes: Optional[int] = None,
    terminal_kv_available: bool = False,
) -> Tuple[Dict[str, object], WorkloadTrace]:
    lifecycle = analyze_session_lifecycle(
        requests,
        config,
        ttl_s=ttl_s,
        capacity_bytes=capacity_bytes,
        terminal_kv_available=terminal_kv_available,
    )
    return lifecycle, lower_session_lifecycle_trace(lifecycle, config)
