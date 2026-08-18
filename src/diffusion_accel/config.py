"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .simulator import ComputeConfig, HardwareConfig, MemoryConfig, SramConfig


def load_hardware(path: Path) -> HardwareConfig:
    with path.open("r", encoding="utf-8") as handle:
        data: Dict[str, Any] = yaml.safe_load(handle)
    return HardwareConfig(
        name=str(data["name"]),
        compute=ComputeConfig(
            peak_tops=float(data["compute"]["peak_tops"]),
            utilization=float(data["compute"]["utilization"]),
        ),
        hbm=MemoryConfig(
            bandwidth_gb_s=float(data["hbm"]["bandwidth_gb_s"]),
            efficiency=float(data["hbm"]["efficiency"]),
        ),
        sram=SramConfig(
            capacity_bytes=int(float(data["sram"]["capacity_mb"]) * 1024 * 1024),
            bandwidth_gb_s=float(data["sram"]["bandwidth_gb_s"]),
            efficiency=float(data["sram"]["efficiency"]),
        ),
    )
