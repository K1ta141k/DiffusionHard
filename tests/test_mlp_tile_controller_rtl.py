from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_mlp_tile_controller_rtl() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")

    build = Path("/tmp/tb_mlp_tile_controller")
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "tb_mlp_tile_controller",
            "-o",
            str(build),
            str(ROOT / "rtl/tensor_engine/int8_mac_tile_pipelined.sv"),
            str(ROOT / "rtl/tensor_engine/mlp_tile_controller.sv"),
            str(ROOT / "rtl/tensor_engine/tb_mlp_tile_controller.sv"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        ["vvp", str(build)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_mlp_tile_controller: PASS" in run_result.stdout
