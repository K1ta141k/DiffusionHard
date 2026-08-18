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
    os.environ.get("DIFFUSION_ACCEL_RUN_DDIT_IMAGE_FABRIC_RTL")!="1",
    reason="set DIFFUSION_ACCEL_RUN_DDIT_IMAGE_FABRIC_RTL=1 for full image-port RTL",
)
def test_one_axi_port_preloads_constants_and_drives_connected_ddit_block()->None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build=Path("/tmp/tb_ddit_block_with_image_fabric")
    compile_result=subprocess.run(
        ["iverilog","-g2012","-Wall","-s","tb_ddit_block_with_image_fabric",
         "-o",str(build),*(str(path) for path in sorted(RTL.glob("*.sv")))],
        cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert compile_result.returncode==0,compile_result.stderr
    run_result=subprocess.run(
        ["vvp",str(build)],cwd=ROOT,check=False,capture_output=True,text=True,
        timeout=1200,
    )
    if os.environ.get("DIFFUSION_ACCEL_SHOW_LONG_RTL_OUTPUT")=="1":
        print(run_result.stdout,end="")
    assert run_result.returncode==0,run_result.stdout+run_result.stderr
    assert "tb_ddit_block_with_image_fabric: PASS" in run_result.stdout
    match=re.search(
        r"cycles=(\d+) transactions=(\d+) bytes=(\d+) constants=(\d+) dense=(\d+)",
        run_result.stdout,
    )
    assert match is not None
    assert int(match.group(2))==4276
    assert int(match.group(3))==929664
    assert int(match.group(4))==11264
    assert int(match.group(5))==918400
