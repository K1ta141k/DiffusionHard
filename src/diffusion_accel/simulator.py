"""Capacity-aware analytical compute and memory simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Set, Tuple

from .ir import Lifetime, Operation, TensorAccess, WorkloadTrace

MemoryLevel = Literal["hbm", "sram"]


@dataclass(frozen=True)
class ComputeConfig:
    peak_tops: float
    utilization: float


@dataclass(frozen=True)
class MemoryConfig:
    bandwidth_gb_s: float
    efficiency: float


@dataclass(frozen=True)
class SramConfig(MemoryConfig):
    capacity_bytes: int


@dataclass(frozen=True)
class HardwareConfig:
    name: str
    compute: ComputeConfig
    hbm: MemoryConfig
    sram: SramConfig


@dataclass(frozen=True)
class SimulationResult:
    hardware_name: str
    policy_name: str
    total_latency_s: float
    compute_time_s: float
    hbm_time_s: float
    sram_time_s: float
    total_flops: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    sram_read_bytes: int
    sram_write_bytes: int
    peak_sram_bytes: int

    @property
    def hbm_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes

    @property
    def sram_bytes(self) -> int:
        return self.sram_read_bytes + self.sram_write_bytes

    def to_dict(self) -> Dict[str, object]:
        return {
            "hardware_name": self.hardware_name,
            "policy_name": self.policy_name,
            "total_latency_ms": self.total_latency_s * 1e3,
            "compute_time_ms": self.compute_time_s * 1e3,
            "hbm_time_ms": self.hbm_time_s * 1e3,
            "sram_time_ms": self.sram_time_s * 1e3,
            "total_flops": self.total_flops,
            "hbm_read_bytes": self.hbm_read_bytes,
            "hbm_write_bytes": self.hbm_write_bytes,
            "hbm_bytes": self.hbm_bytes,
            "sram_read_bytes": self.sram_read_bytes,
            "sram_write_bytes": self.sram_write_bytes,
            "sram_bytes": self.sram_bytes,
            "peak_sram_bytes": self.peak_sram_bytes,
        }


class MemoryPolicy:
    name = "base"

    def wants_sram(self, access: TensorAccess) -> bool:
        return False


class AllHBMPolicy(MemoryPolicy):
    name = "all-hbm"


class CanvasSRAMPolicy(MemoryPolicy):
    name = "canvas-sram"
    _categories: Set[str] = {"canvas_state", "canvas_kv", "logit", "metadata"}

    def wants_sram(self, access: TensorAccess) -> bool:
        return access.category in self._categories


@dataclass
class _ResidentTensor:
    size_bytes: int
    lifetime: Lifetime


class _MemoryState:
    def __init__(self, capacity_bytes: int, policy: MemoryPolicy) -> None:
        self.capacity_bytes = capacity_bytes
        self.policy = policy
        self.resident: Dict[str, _ResidentTensor] = {}
        self.occupancy_bytes = 0
        self.peak_bytes = 0

    def _place(self, access: TensorAccess) -> bool:
        if access.name in self.resident:
            return True
        if not self.policy.wants_sram(access):
            return False
        if self.occupancy_bytes + access.size_bytes > self.capacity_bytes:
            return False
        self.resident[access.name] = _ResidentTensor(
            size_bytes=access.size_bytes,
            lifetime=access.lifetime,
        )
        self.occupancy_bytes += access.size_bytes
        self.peak_bytes = max(self.peak_bytes, self.occupancy_bytes)
        return True

    def access(self, access: TensorAccess) -> Tuple[int, int, int, int]:
        """Return HBM read/write and SRAM read/write bytes."""
        already_resident = access.name in self.resident
        placed = self._place(access)
        if not placed:
            if access.access == "read":
                return access.size_bytes, 0, 0, 0
            return 0, access.size_bytes, 0, 0

        if access.access == "read":
            hbm_fill = 0 if already_resident else access.size_bytes
            sram_fill = 0 if already_resident else access.size_bytes
            return hbm_fill, 0, access.size_bytes, sram_fill
        return 0, 0, 0, access.size_bytes

    def release(self, lifetime: Lifetime) -> None:
        names = [
            name for name, tensor in self.resident.items()
            if tensor.lifetime == lifetime
        ]
        for name in names:
            self.occupancy_bytes -= self.resident[name].size_bytes
            del self.resident[name]


def simulate(
    trace: WorkloadTrace,
    hardware: HardwareConfig,
    policy: MemoryPolicy,
) -> SimulationResult:
    state = _MemoryState(hardware.sram.capacity_bytes, policy)
    total_latency = 0.0
    compute_total = 0.0
    hbm_total_time = 0.0
    sram_total_time = 0.0
    total_flops = 0
    hbm_read = hbm_write = sram_read = sram_write = 0

    compute_rate = hardware.compute.peak_tops * hardware.compute.utilization * 1e12
    hbm_rate = hardware.hbm.bandwidth_gb_s * hardware.hbm.efficiency * 1e9
    sram_rate = hardware.sram.bandwidth_gb_s * hardware.sram.efficiency * 1e9
    if min(compute_rate, hbm_rate, sram_rate) <= 0:
        raise ValueError("effective hardware rates must be positive")

    for step in trace.steps:
        for operation in step.operations:
            op_hbm_read = op_hbm_write = op_sram_read = op_sram_write = 0
            for access in operation.reads + operation.writes:
                hr, hw, sr, sw = state.access(access)
                op_hbm_read += hr
                op_hbm_write += hw
                op_sram_read += sr
                op_sram_write += sw

            compute_time = operation.flops / compute_rate
            hbm_time = (op_hbm_read + op_hbm_write) / hbm_rate
            sram_time = (op_sram_read + op_sram_write) / sram_rate
            total_latency += max(compute_time, hbm_time, sram_time)
            compute_total += compute_time
            hbm_total_time += hbm_time
            sram_total_time += sram_time
            total_flops += operation.flops
            hbm_read += op_hbm_read
            hbm_write += op_hbm_write
            sram_read += op_sram_read
            sram_write += op_sram_write
            state.release("operation")
        state.release("step")
        if bool(step.metadata.get("request_end", False)):
            state.release("canvas")
            state.release("request")
    state.release("canvas")
    state.release("request")
    state.release("session")

    return SimulationResult(
        hardware_name=hardware.name,
        policy_name=policy.name,
        total_latency_s=total_latency,
        compute_time_s=compute_total,
        hbm_time_s=hbm_total_time,
        sram_time_s=sram_total_time,
        total_flops=total_flops,
        hbm_read_bytes=hbm_read,
        hbm_write_bytes=hbm_write,
        sram_read_bytes=sram_read,
        sram_write_bytes=sram_write,
        peak_sram_bytes=state.peak_bytes,
    )
