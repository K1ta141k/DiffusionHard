from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


@pytest.mark.skipif(
    os.environ.get("DIFFUSION_ACCEL_RUN_DDIT_BLOCK_RTL") != "1",
    reason="set DIFFUSION_ACCEL_RUN_DDIT_BLOCK_RTL=1 for connected DDiT RTL",
)
@pytest.mark.parametrize(
    "packed_attention",
    [False, True],
    ids=["fixed18", "packed-m8"],
)
def test_attention_and_automatic_mlp_form_one_ddit_block(
    packed_attention: bool,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    suffix = "packed" if packed_attention else "fixed"
    build = Path(f"/tmp/tb_ddit_block_pipeline_{suffix}")
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall",
            *(["-DPACKED_ATTENTION"] if packed_attention else []),
            "-s", "tb_ddit_block_pipeline",
            "-o", str(build),
            *(str(path) for path in sorted(RTL.glob("*.sv"))),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        ["vvp", str(build)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_ddit_block_pipeline: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) == (40_777 if packed_attention else 44_425)
