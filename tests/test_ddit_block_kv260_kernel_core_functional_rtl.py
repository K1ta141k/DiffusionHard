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
    os.environ.get("DIFFUSION_ACCEL_RUN_KV260_CORE_RTL")!="1",
    reason="set DIFFUSION_ACCEL_RUN_KV260_CORE_RTL=1 for kernel DMA RTL",
)
def test_kv260_core_runs_residual_dma_block_and_output_dma()->None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build=Path("/tmp/tb_ddit_block_kv260_kernel_core")
    comp=subprocess.run(
        ["iverilog","-g2012","-Wall","-s","tb_ddit_block_kv260_kernel_core",
         "-o",str(build),*(str(path) for path in sorted(RTL.glob("*.sv")))],
        cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert comp.returncode==0,comp.stderr
    run=subprocess.run(
        ["vvp",str(build)],cwd=ROOT,check=False,capture_output=True,text=True,
        timeout=1200,
    )
    if os.environ.get("DIFFUSION_ACCEL_SHOW_LONG_RTL_OUTPUT")=="1":
        print(run.stdout,end="")
    assert run.returncode==0,run.stdout+run.stderr
    assert "tb_ddit_block_kv260_kernel_core: PASS" in run.stdout
    match=re.search(
        r"cycles=(\d+) reads=(\d+) read_bytes=(\d+) output_bytes=(\d+)",
        run.stdout,
    )
    assert match is not None
    assert int(match.group(2))==6324
    assert int(match.group(3))==1_191_808
    assert int(match.group(4))==72
