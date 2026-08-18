"""Latency contracts for the model-specific real-time accelerator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from .model_spec import load_model_spec
from .model_specialization import specialization_inventory


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def analyze_realtime_target(path: Path, project_root: Path) -> Dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = load_model_spec(_resolve(project_root, str(data["model"])))
    manifest = model.hardware_manifest()
    full_forward_macs = int(manifest["forward_macs"]["total"])
    parameters = model.parameter_count
    specialization = specialization_inventory(model)
    target = data["latency_contract"]
    target_ms = float(target["target_generation_ms"])
    stretch_ms = float(target["stretch_generation_ms"])
    results: List[Dict[str, object]] = []

    for point in data["design_points"]:
        bits = int(point["weight_bits"])
        if bits == 8:
            weight_bytes = int(
                specialization[
                    "analytical_int8_plus_fp16_rotary_bytes_per_evaluation"
                ]
            )
        elif bits == 16:
            weight_bytes = int(
                specialization["fp16_bytes_per_evaluation_after_folding"]
            )
        else:
            rotary_values = int(specialization["fixed_rotary_values"])
            runtime_values = (
                int(
                    specialization[
                        "fp16_bytes_per_evaluation_after_folding"
                    ]
                )
                // 2
                - rotary_values
            )
            weight_bytes = (
                (runtime_values * bits + 7) // 8
                + rotary_values * 2
            )
        lanes = int(point["effective_mac_lanes"])
        clock_hz = float(point["clock_mhz"]) * 1_000_000.0
        utilization = float(point["compute_utilization"])
        ddr_bytes_per_second = float(point["effective_ddr_gbps"]) * 1e9
        evaluations = int(point["model_evaluations"])
        overhead_ms = float(point["fixed_generation_overhead_ms"])

        arithmetic_floor_ms = full_forward_macs / (lanes * clock_hz) * 1e3
        compute_bound_ms = arithmetic_floor_ms / utilization
        ddr_bound_ms = weight_bytes / ddr_bytes_per_second * 1e3
        evaluation_bound_ms = max(compute_bound_ms, ddr_bound_ms)
        generation_bound_ms = evaluations * evaluation_bound_ms + overhead_ms
        results.append(
            {
                "name": str(point["name"]),
                "weight_bits": bits,
                "weight_bytes": weight_bytes,
                "effective_mac_lanes": lanes,
                "clock_mhz": float(point["clock_mhz"]),
                "compute_utilization": utilization,
                "model_evaluations": evaluations,
                "arithmetic_floor_ms_per_evaluation": arithmetic_floor_ms,
                "utilization_adjusted_compute_ms_per_evaluation": compute_bound_ms,
                "ddr_floor_ms_per_evaluation": ddr_bound_ms,
                "estimated_generation_lower_bound_ms": generation_bound_ms,
                "meets_target": generation_bound_ms <= target_ms,
                "meets_stretch": generation_bound_ms <= stretch_ms,
            }
        )

    baseline = data["baseline"]
    baseline_ms = float(baseline["measured_model_forward_ms"])
    for result in results:
        result["speedup_over_measured_mps_model_forwards"] = (
            baseline_ms / float(result["estimated_generation_lower_bound_ms"])
        )

    return {
        "name": str(data["name"]),
        "model": model.model_id,
        "revision": model.revision,
        "canvas_tokens": int(target["canvas_tokens"]),
        "full_forward_macs": full_forward_macs,
        "model_specialization": specialization,
        "baseline": baseline,
        "latency_contract": target,
        "design_points": results,
        "limitations": [
            "analytical lower bounds, not placed-and-routed measurements",
            "uses full 64-position vocabulary projection for every evaluation",
            "eight-evaluation generation requires schedule tuning or distillation",
            "quality acceptance must be measured before the latency claim is valid",
        ],
    }
