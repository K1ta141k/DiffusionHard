from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_block_parameter_dma_reads_aligned_and_compact_records() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_mdlm_block_parameter_dma")
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s", "tb_mdlm_block_parameter_dma",
            "-o", str(build),
            str(RTL / "mdlm_block_parameter_address_generator.sv"),
            str(RTL / "axi512_read_burst_master.sv"),
            str(RTL / "mdlm_block_parameter_dma.sv"),
            str(RTL / "tb_mdlm_block_parameter_dma.sv"),
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
    assert "tb_mdlm_block_parameter_dma: PASS" in run_result.stdout
