from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_streaming_qkv_scheduler_overlaps_group_tail() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_qkv_projection_output_tile_scheduler_streaming")
    sources = [
        "int8_mac_tile_pipelined.sv",
        "mixed_precision_mac_tile_pipelined.sv",
        "fixed_requantize.sv",
        "fixed_requantize_vector_serial.sv",
        "qkv_weight_tile_buffer.sv",
        "qkv_projection_output_tile_scheduler_streaming.sv",
        "tb_qkv_projection_output_tile_scheduler_streaming.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall",
            "-s", "tb_qkv_projection_output_tile_scheduler_streaming",
            "-o", str(build), *(str(RTL / item) for item in sources),
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
    match = re.search(r"PASS cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) < 500
