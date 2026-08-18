from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]
RTL=ROOT/"rtl/tensor_engine"


def test_complete_image_fabric_elaborates_with_one_external_axi_port(tmp_path:Path)->None:
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog is required")
    build=tmp_path/"ddit_block_with_image_fabric"
    result=subprocess.run(
        ["iverilog","-g2012","-Wall","-s","ddit_block_with_image_fabric",
         "-o",str(build),*(str(path) for path in sorted(RTL.glob("*.sv")))],
        cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
