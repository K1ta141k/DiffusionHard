"""Trace-level architectural what-if transformations."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import List

from .ir import DiffusionStep, Operation, TensorAccess, WorkloadTrace


def masked_output_head(trace: WorkloadTrace) -> WorkloadTrace:
    """Project vocabulary logits only for positions still masked at each step."""
    optimized_steps: List[DiffusionStep] = []
    transformed = 0

    for step in trace.steps:
        operations: List[Operation] = []
        logits_name = None
        optimized_logits_bytes = None
        for operation in step.operations:
            if operation.name != "full_vocabulary_projection":
                operations.append(operation)
                continue

            computed_tokens = int(
                operation.metadata.get("computed_tokens", operation.parallelism)
            )
            if computed_tokens <= 0:
                raise ValueError("projection computed_tokens must be positive")
            active_tokens = step.active_tokens
            numerator = min(active_tokens, computed_tokens)
            writes: List[TensorAccess] = []
            for access in operation.writes:
                reduced_bytes = access.size_bytes * numerator // computed_tokens
                writes.append(replace(access, size_bytes=reduced_bytes))
                if access.category == "logit":
                    logits_name = access.name
                    optimized_logits_bytes = reduced_bytes
            metadata = dict(operation.metadata)
            metadata.update(
                {
                    "computed_tokens": numerator,
                    "baseline_computed_tokens": computed_tokens,
                    "optimization": "masked-output-head",
                }
            )
            operations.append(
                replace(
                    operation,
                    name="masked_vocabulary_projection",
                    flops=operation.flops * numerator // computed_tokens,
                    writes=writes,
                    parallelism=max(1, numerator),
                    metadata=metadata,
                )
            )
            transformed += 1

        if logits_name is not None and optimized_logits_bytes is not None:
            rewritten: List[Operation] = []
            for operation in operations:
                reads = [
                    replace(access, size_bytes=optimized_logits_bytes)
                    if access.name == logits_name and access.category == "logit"
                    else access
                    for access in operation.reads
                ]
                rewritten.append(replace(operation, reads=reads))
            operations = rewritten

        optimized_steps.append(replace(step, operations=operations))

    if transformed == 0:
        raise ValueError("trace has no full_vocabulary_projection operations")
    metadata = dict(trace.metadata)
    applied = list(metadata.get("optimizations", []))
    applied.append("masked-output-head")
    metadata["optimizations"] = applied
    metadata["source_workload_name"] = trace.workload_name
    return replace(
        trace,
        workload_name=trace.workload_name + "+masked-output-head",
        steps=optimized_steps,
        metadata=metadata,
    )


def fused_streaming_candidate_head(trace: WorkloadTrace) -> WorkloadTrace:
    """Stream projection logits into categorical selection without a tensor.

    Projection and categorical FLOPs are preserved. Only the materialized
    logit write and read are replaced by compact candidate state, so this
    transformation isolates the memory benefit of fusion.
    """
    vocabulary_size = int(trace.metadata.get("vocab_size", 0))
    canvas_tokens = int(trace.metadata.get("canvas_tokens", 0))
    if vocabulary_size <= 1 or canvas_tokens <= 0:
        raise ValueError("trace must provide positive vocab_size and canvas_tokens")
    candidate_id_bits = max(1, math.ceil(math.log2(vocabulary_size)))
    candidate_id_bytes = math.ceil(candidate_id_bits / 8)
    candidate_state_bytes = (
        canvas_tokens * candidate_id_bytes
        + 2 * math.ceil(canvas_tokens / 8)
    )

    transformed_steps: List[DiffusionStep] = []
    transformed = 0
    for step in trace.steps:
        logit_names = set()
        candidate_name = "step_%03d_candidate_state" % step.step_id
        operations: List[Operation] = []
        for operation in step.operations:
            if operation.name not in {
                "full_vocabulary_projection",
                "masked_vocabulary_projection",
            }:
                operations.append(operation)
                continue
            writes: List[TensorAccess] = []
            for access in operation.writes:
                if access.category == "logit":
                    logit_names.add(access.name)
                    writes.append(
                        TensorAccess(
                            name=candidate_name,
                            size_bytes=candidate_state_bytes,
                            access="write",
                            lifetime="step",
                            category="metadata",
                        )
                    )
                else:
                    writes.append(access)
            metadata = dict(operation.metadata)
            metadata.update(
                {
                    "optimization": "fused-streaming-candidate-head",
                    "candidate_state_bytes": candidate_state_bytes,
                    "categorical_compute_preserved": True,
                }
            )
            operations.append(
                replace(
                    operation,
                    name="fused_streaming_candidate_projection",
                    writes=writes,
                    metadata=metadata,
                )
            )
            transformed += 1

        rewritten: List[Operation] = []
        for operation in operations:
            replaced_read = False
            reads: List[TensorAccess] = []
            for access in operation.reads:
                if access.name in logit_names and access.category == "logit":
                    reads.append(
                        TensorAccess(
                            name=candidate_name,
                            size_bytes=candidate_state_bytes,
                            access="read",
                            lifetime="step",
                            category="metadata",
                        )
                    )
                    replaced_read = True
                else:
                    reads.append(access)
            rewritten.append(
                replace(
                    operation,
                    name=(
                        "streaming_candidate_select"
                        if replaced_read
                        else operation.name
                    ),
                    reads=reads,
                )
            )
        transformed_steps.append(replace(step, operations=rewritten))

    if transformed == 0:
        raise ValueError("trace has no vocabulary projection operations")
    metadata = dict(trace.metadata)
    applied = list(metadata.get("optimizations", []))
    applied.append("fused-streaming-candidate-head")
    metadata.update(
        {
            "optimizations": applied,
            "source_workload_name": trace.workload_name,
            "candidate_state_bytes": candidate_state_bytes,
            "categorical_compute_preserved": True,
        }
    )
    return replace(
        trace,
        workload_name=trace.workload_name + "+fused-streaming-candidate-head",
        steps=transformed_steps,
        metadata=metadata,
    )


def quantize_weights(
    trace: WorkloadTrace,
    *,
    target_bits: int,
    preserve_output_head: bool = False,
) -> WorkloadTrace:
    """Scale weight traffic for a weight-only quantization design point."""
    if target_bits not in {4, 8}:
        raise ValueError("target_bits must be 4 or 8")
    metadata = dict(trace.metadata)
    source_bits = int(
        metadata.get(
            "weight_bits",
            int(metadata.get("bytes_per_element", 4)) * 8,
        )
    )
    if target_bits >= source_bits:
        raise ValueError("target_bits must be smaller than source weight bits")

    def quantized(access: TensorAccess) -> TensorAccess:
        if access.category != "weight":
            return access
        if preserve_output_head and access.name == "output_projection_weights":
            return access
        return replace(
            access,
            size_bytes=math.ceil(access.size_bytes * target_bits / source_bits),
        )

    steps: List[DiffusionStep] = []
    for step in trace.steps:
        operations = [
            replace(
                operation,
                reads=[quantized(access) for access in operation.reads],
                writes=[quantized(access) for access in operation.writes],
            )
            for operation in step.operations
        ]
        steps.append(replace(step, operations=operations))

    applied = list(metadata.get("optimizations", []))
    optimization_name = "weight-int%d" % target_bits
    if preserve_output_head:
        optimization_name += "-fp32-output-head"
    applied.append(optimization_name)
    metadata.update(
        {
            "optimizations": applied,
            "source_weight_bits": source_bits,
            "weight_bits": target_bits,
            "quantization_accuracy_validated": False,
            "preserve_output_head_fp32": preserve_output_head,
        }
    )
    return replace(
        trace,
        workload_name=trace.workload_name + "+" + optimization_name,
        steps=steps,
        metadata=metadata,
    )
