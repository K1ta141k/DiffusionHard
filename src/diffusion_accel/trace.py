"""Versioned JSONL trace I/O and synthetic workload generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .ir import DiffusionStep, Operation, TensorAccess, WorkloadTrace

SCHEMA_VERSION = "0.1"


def _access_from_dict(data: Dict[str, Any]) -> TensorAccess:
    return TensorAccess(**data)


def _operation_from_dict(data: Dict[str, Any]) -> Operation:
    return Operation(
        name=data["name"],
        flops=data["flops"],
        reads=[_access_from_dict(item) for item in data.get("reads", [])],
        writes=[_access_from_dict(item) for item in data.get("writes", [])],
        parallelism=data.get("parallelism", 1),
        metadata=data.get("metadata", {}),
    )


def _step_from_dict(data: Dict[str, Any]) -> DiffusionStep:
    return DiffusionStep(
        canvas_id=data["canvas_id"],
        step_id=data["step_id"],
        active_tokens=data["active_tokens"],
        changed_tokens=data["changed_tokens"],
        operations=[_operation_from_dict(item) for item in data["operations"]],
        metadata=data.get("metadata", {}),
    )


def write_trace(trace: WorkloadTrace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "record_type": "header",
        "schema_version": trace.schema_version,
        "workload_name": trace.workload_name,
        "metadata": trace.metadata,
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for step in trace.steps:
            record = {"record_type": "step", **WorkloadTrace(
                schema_version=trace.schema_version,
                workload_name=trace.workload_name,
                steps=[step],
            ).to_dict()["steps"][0]}
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_trace(path: Path) -> WorkloadTrace:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records or records[0].get("record_type") != "header":
        raise ValueError("trace must begin with a header record")
    header = records[0]
    if header["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "unsupported schema version %s" % header["schema_version"]
        )
    steps = []
    for record in records[1:]:
        if record.get("record_type") != "step":
            raise ValueError("unexpected trace record type")
        step_data = dict(record)
        step_data.pop("record_type")
        steps.append(_step_from_dict(step_data))
    return WorkloadTrace(
        schema_version=header["schema_version"],
        workload_name=header["workload_name"],
        steps=steps,
        metadata=header.get("metadata", {}),
    )


def synthetic_trace(
    *,
    steps: int = 10,
    layers: int = 12,
    canvas_tokens: int = 128,
    hidden_size: int = 768,
    vocab_size: int = 32768,
    model_weight_bytes: int = 250_000_000,
    bytes_per_element: int = 2,
) -> WorkloadTrace:
    """Create a deterministic masked-diffusion trace for simulator testing."""
    for name, value in {
        "steps": steps,
        "layers": layers,
        "canvas_tokens": canvas_tokens,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "model_weight_bytes": model_weight_bytes,
        "bytes_per_element": bytes_per_element,
    }.items():
        if value <= 0:
            raise ValueError("%s must be positive" % name)

    hidden_bytes = canvas_tokens * hidden_size * bytes_per_element
    kv_bytes = 2 * canvas_tokens * hidden_size * bytes_per_element
    logits_bytes = canvas_tokens * vocab_size * bytes_per_element
    lm_head_weight_bytes = hidden_size * vocab_size * bytes_per_element
    transformer_weight_bytes = max(0, model_weight_bytes - lm_head_weight_bytes)
    per_layer_weights = transformer_weight_bytes // layers
    trace_steps: List[DiffusionStep] = []

    for step_id in range(steps):
        active_tokens = max(1, canvas_tokens - (canvas_tokens * step_id // steps))
        next_active = max(0, canvas_tokens - (canvas_tokens * (step_id + 1) // steps))
        changed_tokens = active_tokens - next_active
        operations: List[Operation] = []

        for layer in range(layers):
            source = "canvas_hidden_a" if layer % 2 == 0 else "canvas_hidden_b"
            target = "canvas_hidden_b" if layer % 2 == 0 else "canvas_hidden_a"
            operations.append(
                Operation(
                    name="transformer_layer_%d" % layer,
                    flops=4 * canvas_tokens * hidden_size * hidden_size,
                    reads=[
                        TensorAccess(
                            name="layer_%d_weights" % layer,
                            size_bytes=per_layer_weights,
                            access="read",
                            lifetime="request",
                            category="weight",
                        ),
                        TensorAccess(
                            name=source,
                            size_bytes=hidden_bytes,
                            access="read",
                            lifetime="canvas",
                            category="canvas_state",
                        ),
                    ],
                    writes=[
                        TensorAccess(
                            name=target,
                            size_bytes=hidden_bytes,
                            access="write",
                            lifetime="canvas",
                            category="canvas_state",
                        ),
                        TensorAccess(
                            name="step_%d_layer_%d_kv" % (step_id, layer),
                            size_bytes=kv_bytes,
                            access="write",
                            lifetime="operation",
                            category="canvas_kv",
                        ),
                    ],
                    parallelism=canvas_tokens,
                    metadata={"layer": layer},
                )
            )

        final_hidden = "canvas_hidden_a" if layers % 2 == 0 else "canvas_hidden_b"
        operations.append(
            Operation(
                name="vocabulary_projection",
                flops=2 * active_tokens * hidden_size * vocab_size,
                reads=[
                    TensorAccess(
                        name=final_hidden,
                        size_bytes=hidden_bytes,
                        access="read",
                        lifetime="canvas",
                        category="canvas_state",
                    ),
                    TensorAccess(
                        name="lm_head_weights",
                        size_bytes=lm_head_weight_bytes,
                        access="read",
                        lifetime="request",
                        category="weight",
                    ),
                ],
                writes=[
                    TensorAccess(
                        name="step_%d_logits" % step_id,
                        size_bytes=logits_bytes,
                        access="write",
                        lifetime="step",
                        category="logit",
                    ),
                ],
                parallelism=active_tokens,
                metadata={"active_tokens": active_tokens},
            )
        )
        operations.append(
            Operation(
                name="confidence_selection",
                flops=4 * active_tokens * vocab_size,
                reads=[
                    TensorAccess(
                        name="step_%d_logits" % step_id,
                        size_bytes=logits_bytes,
                        access="read",
                        lifetime="step",
                        category="logit",
                    ),
                ],
                writes=[
                    TensorAccess(
                        name="commit_bitmap",
                        size_bytes=(canvas_tokens + 7) // 8,
                        access="write",
                        lifetime="canvas",
                        category="metadata",
                    ),
                ],
                parallelism=active_tokens,
                metadata={"active_tokens": active_tokens},
            )
        )
        trace_steps.append(
            DiffusionStep(
                canvas_id=0,
                step_id=step_id,
                active_tokens=active_tokens,
                changed_tokens=changed_tokens,
                operations=operations,
            )
        )

    return WorkloadTrace(
        schema_version=SCHEMA_VERSION,
        workload_name="synthetic-mdlm",
        steps=trace_steps,
        metadata={
            "layers": layers,
            "canvas_tokens": canvas_tokens,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "model_weight_bytes": model_weight_bytes,
            "lm_head_weight_bytes": lm_head_weight_bytes,
            "bytes_per_element": bytes_per_element,
            "provenance": "assumed-and-derived",
        },
    )
