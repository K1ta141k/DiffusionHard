from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]
RTL=ROOT/"rtl/tensor_engine"


def test_grouped_canvas_streams_one_projection_k_tile_per_cycle()->None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build=Path("/tmp/tb_attention_projection_grouped_scheduler")
    sources=[
        "int8_mac_tile_pipelined.sv","mixed_precision_mac_tile_pipelined.sv",
        "fixed_requantize.sv","fixed_requantize_vector_serial.sv",
        "attention_projection_weight_tile_buffer.sv",
        "attention_projection_output_tile_scheduler.sv",
        "tb_attention_projection_grouped_scheduler.sv",
    ]
    compile_result=subprocess.run(
        ["iverilog","-g2012","-Wall","-s",
         "tb_attention_projection_grouped_scheduler","-o",str(build),
         *(str(RTL/source) for source in sources)],cwd=ROOT,check=False,
        capture_output=True,text=True,
    )
    assert compile_result.returncode==0,compile_result.stderr
    run_result=subprocess.run(
        ["vvp",str(build)],cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert run_result.returncode==0,run_result.stdout+run_result.stderr
    match=re.search(r"PASS cycles=(\d+) reads=(\d+) requests=(\d+) outputs=(\d+)",
                    run_result.stdout)
    assert match is not None
    assert tuple(int(value) for value in match.groups()[1:])==(384,384,16)
    assert int(match.group(1))<1500
