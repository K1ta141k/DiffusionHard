"""Hardware-neutral intermediate representation for diffusion inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal

AccessKind = Literal["read", "write"]
Lifetime = Literal["operation", "step", "canvas", "request", "session"]
TensorCategory = Literal[
    "weight",
    "context_kv",
    "session_kv",
    "canvas_kv",
    "canvas_state",
    "activation",
    "logit",
    "metadata",
]


@dataclass(frozen=True)
class TensorAccess:
    name: str
    size_bytes: int
    access: AccessKind
    lifetime: Lifetime
    category: TensorCategory

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")


@dataclass(frozen=True)
class Operation:
    name: str
    flops: int
    reads: List[TensorAccess] = field(default_factory=list)
    writes: List[TensorAccess] = field(default_factory=list)
    parallelism: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.flops < 0:
            raise ValueError("flops must be non-negative")
        if self.parallelism < 1:
            raise ValueError("parallelism must be at least one")


@dataclass(frozen=True)
class DiffusionStep:
    canvas_id: int
    step_id: int
    active_tokens: int
    changed_tokens: int
    operations: List[Operation]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadTrace:
    schema_version: str
    workload_name: str
    steps: List[DiffusionStep]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
