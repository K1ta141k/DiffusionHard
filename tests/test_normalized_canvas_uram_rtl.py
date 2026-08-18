from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_normalized_canvas_uram_layout() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_normalized_canvas_uram")
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s", "tb_normalized_canvas_uram",
            "-o", str(build), str(RTL / "normalized_canvas_uram.sv"),
            str(RTL / "tb_normalized_canvas_uram.sv"),
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
    assert "tb_normalized_canvas_uram: PASS" in run_result.stdout
