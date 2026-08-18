from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]
RTL=ROOT/"rtl/tensor_engine"


@pytest.mark.skipif(
    os.environ.get("DIFFUSION_ACCEL_RUN_DDIT_FABRIC_RTL")!="1",
    reason="set DIFFUSION_ACCEL_RUN_DDIT_FABRIC_RTL=1 for DDR-fed DDiT RTL",
)
def test_shared_parameter_fabric_drives_connected_ddit_block()->None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build=Path("/tmp/tb_ddit_block_with_parameter_fabric")
    compile_result=subprocess.run(
        ["iverilog","-g2012","-Wall","-s","tb_ddit_block_with_parameter_fabric",
         "-o",str(build),*(str(path) for path in sorted(RTL.glob("*.sv")))],
        cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert compile_result.returncode==0,compile_result.stderr
    run_result=subprocess.run(
        ["vvp",str(build)],cwd=ROOT,check=False,capture_output=True,text=True,
        timeout=1200,
    )
    assert run_result.returncode==0,run_result.stdout+run_result.stderr
    assert "tb_ddit_block_with_parameter_fabric: PASS" in run_result.stdout
    match=re.search(r"cycles=(\d+) transactions=(\d+) bytes=(\d+)",run_result.stdout)
    assert match is not None
    assert int(match.group(2))==4100
    assert int(match.group(3))==918400
