from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_hidden_canvas_drives_three_pass_norm2_frontend() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_hidden_canvas_mlp_frontend")
    sources = [
        "unsigned_divider_iterative.sv",
        "unsigned_sqrt_iterative.sv",
        "layer_norm_q12_group.sv",
        "mlp_up_activation_quantizer.sv",
        "layer_norm_mlp_up_activation_frontend.sv",
        "hidden_canvas_group_replay.sv",
        "hidden_canvas_mlp_frontend.sv",
        "tb_hidden_canvas_mlp_frontend.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s",
            "tb_hidden_canvas_mlp_frontend", "-o", str(build),
            *(str(RTL / source) for source in sources),
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
    assert "tb_hidden_canvas_mlp_frontend: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) < 8000
