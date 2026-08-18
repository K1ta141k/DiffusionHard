from diffusion_accel.ir import DiffusionStep, Operation, TensorAccess, WorkloadTrace
from diffusion_accel.simulator import (
    AllHBMPolicy,
    CanvasSRAMPolicy,
    ComputeConfig,
    HardwareConfig,
    MemoryConfig,
    SramConfig,
    simulate,
)
from diffusion_accel.trace import SCHEMA_VERSION, synthetic_trace


def hardware(capacity_bytes: int = 16 * 1024 * 1024) -> HardwareConfig:
    return HardwareConfig(
        name="test",
        compute=ComputeConfig(peak_tops=1.0, utilization=1.0),
        hbm=MemoryConfig(bandwidth_gb_s=1.0, efficiency=1.0),
        sram=SramConfig(
            capacity_bytes=capacity_bytes,
            bandwidth_gb_s=10.0,
            efficiency=1.0,
        ),
    )


def test_roofline_uses_maximum_resource_time() -> None:
    trace = WorkloadTrace(
        schema_version=SCHEMA_VERSION,
        workload_name="roofline-test",
        steps=[
            DiffusionStep(
                canvas_id=0,
                step_id=0,
                active_tokens=1,
                changed_tokens=1,
                operations=[
                    Operation(
                        name="op",
                        flops=1_000_000_000_000,
                        reads=[
                            TensorAccess(
                                name="weight",
                                size_bytes=1_000_000_000,
                                access="read",
                                lifetime="request",
                                category="weight",
                            )
                        ],
                    )
                ],
            )
        ],
    )
    result = simulate(trace, hardware(), AllHBMPolicy())
    assert result.compute_time_s == 1.0
    assert result.hbm_time_s == 1.0
    assert result.total_latency_s == 1.0


def test_canvas_sram_reduces_hbm_traffic() -> None:
    trace = synthetic_trace(steps=3, layers=2, canvas_tokens=16, hidden_size=64)
    baseline = simulate(trace, hardware(), AllHBMPolicy())
    optimized = simulate(trace, hardware(), CanvasSRAMPolicy())

    assert optimized.hbm_bytes < baseline.hbm_bytes
    assert optimized.sram_bytes > 0
    assert optimized.peak_sram_bytes <= hardware().sram.capacity_bytes


def test_zero_sram_capacity_matches_all_hbm_traffic() -> None:
    trace = synthetic_trace(steps=2, layers=2, canvas_tokens=16, hidden_size=64)
    baseline = simulate(trace, hardware(capacity_bytes=0), AllHBMPolicy())
    optimized = simulate(trace, hardware(capacity_bytes=0), CanvasSRAMPolicy())

    assert optimized.hbm_bytes == baseline.hbm_bytes
    assert optimized.sram_bytes == 0
