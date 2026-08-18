from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_hidden_canvas_replays_one_group_in_channel_order() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_hidden_canvas_group_replay")
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s",
            "tb_hidden_canvas_group_replay", "-o", str(build),
            str(RTL / "hidden_canvas_group_replay.sv"),
            str(RTL / "tb_hidden_canvas_group_replay.sv"),
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
    assert "tb_hidden_canvas_group_replay: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) < 1200
