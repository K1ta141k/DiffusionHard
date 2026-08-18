from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_axi512_read_master_handles_backpressure_counters_and_errors() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_axi512_read_burst_master")
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s",
            "tb_axi512_read_burst_master", "-o", str(build),
            str(RTL / "axi512_read_burst_master.sv"),
            str(RTL / "tb_axi512_read_burst_master.sv"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        ["vvp", str(build)], cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_axi512_read_burst_master: PASS" in run_result.stdout
