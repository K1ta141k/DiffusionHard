from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_mlp_block_controller_sequences_and_preloads_tiles() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_mlp_block_controller")
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s", "tb_mlp_block_controller",
            "-o", str(build), str(RTL / "mlp_block_controller.sv"),
            str(RTL / "tb_mlp_block_controller.sv"),
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
    assert "tb_mlp_block_controller: PASS" in run_result.stdout
