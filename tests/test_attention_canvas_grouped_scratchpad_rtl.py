from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]
RTL=ROOT/"rtl/tensor_engine"


def test_grouped_attention_canvas_returns_four_tokens_per_read()->None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build=Path("/tmp/tb_attention_canvas_grouped_scratchpad_banked")
    compile_result=subprocess.run(
        ["iverilog","-g2012","-Wall","-s",
         "tb_attention_canvas_grouped_scratchpad_banked","-o",str(build),
         str(RTL/"attention_canvas_grouped_scratchpad_banked.sv"),
         str(RTL/"tb_attention_canvas_grouped_scratchpad_banked.sv")],
        cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert compile_result.returncode==0,compile_result.stderr
    run_result=subprocess.run(
        ["vvp",str(build)],cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert run_result.returncode==0,run_result.stdout+run_result.stderr
    assert "tb_attention_canvas_grouped_scratchpad_banked: PASS" in run_result.stdout
