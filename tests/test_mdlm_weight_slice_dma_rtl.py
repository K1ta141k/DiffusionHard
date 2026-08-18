from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_qkv_weight_slice_dma_loads_all_24_tiles() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_mdlm_weight_slice_dma")
    sources = [
        "mdlm_block_parameter_address_generator.sv",
        "axi512_read_burst_master.sv",
        "mdlm_block_parameter_dma.sv",
        "fixed_weight_record_adapter.sv",
        "mdlm_weight_slice_dma.sv",
        "tb_mdlm_weight_slice_dma.sv",
    ]
    compile_result = subprocess.run(
        ["iverilog", "-g2012", "-Wall", "-s", "tb_mdlm_weight_slice_dma",
         "-o", str(build), *(str(RTL / source) for source in sources)],
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
    assert "tb_mdlm_weight_slice_dma: PASS" in run_result.stdout
