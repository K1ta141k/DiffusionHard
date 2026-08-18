from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_mlp_interstage_pipeline_rtl() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_mlp_interstage_pipeline")
    result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "tb_mlp_interstage_pipeline",
            "-o",
            str(build),
            str(RTL / "smoothquant_int8_vector_serial.sv"),
            str(RTL / "mlp_interstage_tile_bridge_bram.sv"),
            str(RTL / "mlp_interstage_pipeline.sv"),
            str(RTL / "tb_mlp_interstage_pipeline.sv"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    simulation = subprocess.run(
        ["vvp", str(build)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert simulation.returncode == 0, simulation.stdout + simulation.stderr
    assert "tb_mlp_interstage_pipeline: PASS" in simulation.stdout
