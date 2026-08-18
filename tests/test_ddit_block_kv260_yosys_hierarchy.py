from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]
RTL=ROOT/"rtl/tensor_engine"


def test_production_kv260_top_resolves_in_yosys_hierarchy()->None:
    yosys=shutil.which("yosys")
    if yosys is None:
        pytest.skip("Yosys is required")
    sources=sorted(path for path in RTL.glob("*.sv") if not path.name.startswith("tb_"))
    quoted=" ".join(f'"{path}"' for path in sources)
    command=(
        "read_verilog -lib +/xilinx/cells_xtra.v; "
        f"read_verilog -sv -defer {quoted}; "
        "hierarchy -check -top ddit_block_kv260_axi_top; "
        "select -assert-count 1 "
        "t:*mixed_precision_packed_m8_mac_tile_pipelined; stat"
    )
    result=subprocess.run(
        [yosys,"-Q","-p",command],cwd=ROOT,check=False,
        capture_output=True,text=True,timeout=120,
    )
    assert result.returncode==0,result.stdout+result.stderr
    assert "Top module:  \\ddit_block_kv260_axi_top" in result.stdout
    assert "qkv_attention_projection_block_pipeline_packed_m8" in result.stdout
