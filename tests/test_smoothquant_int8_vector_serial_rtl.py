from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_smoothquant_int8_vector_serial_rtl() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_smoothquant_int8_vector_serial")
    result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "tb_smoothquant_int8_vector_serial",
            "-o",
            str(build),
            str(RTL / "smoothquant_int8_vector_serial.sv"),
            str(RTL / "tb_smoothquant_int8_vector_serial.sv"),
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
    assert "tb_smoothquant_int8_vector_serial: PASS" in simulation.stdout
