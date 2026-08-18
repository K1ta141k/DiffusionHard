from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_mlp_down_pingpong_pipeline_rtl() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_mlp_down_pingpong_pipeline")
    sources = [
        RTL / "int8_mac_tile_pipelined.sv",
        RTL / "mlp_tile_pingpong_controller.sv",
        RTL / "fixed_requantize.sv",
        RTL / "fixed_requantize_vector_serial.sv",
        RTL / "residual_add_saturating.sv",
        RTL / "mlp_down_pingpong_pipeline.sv",
        RTL / "tb_mlp_down_pingpong_pipeline.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "tb_mlp_down_pingpong_pipeline",
            "-o",
            str(build),
            *(str(source) for source in sources),
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
    assert "tb_mlp_down_pingpong_pipeline: PASS" in run_result.stdout


def test_residual_add_saturating_rtl() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_residual_add_saturating")
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "tb_residual_add_saturating",
            "-o",
            str(build),
            str(RTL / "residual_add_saturating.sv"),
            str(RTL / "tb_residual_add_saturating.sv"),
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
    assert "tb_residual_add_saturating: PASS" in run_result.stdout
