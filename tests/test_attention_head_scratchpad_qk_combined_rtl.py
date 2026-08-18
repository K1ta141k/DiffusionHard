from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_combined_query_key_scratchpad_matches_interface(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    testbench = tmp_path / "tb_attention_head_scratchpad_qk_combined.sv"
    source = (RTL / "tb_attention_head_scratchpad_banked.sv").read_text(
        encoding="utf-8"
    )
    source = source.replace(
        "module tb_attention_head_scratchpad_banked;",
        "module tb_attention_head_scratchpad_qk_combined;",
    ).replace(
        "attention_head_scratchpad_banked dut(",
        "attention_head_scratchpad_qk_combined_banked dut(",
    ).replace(
        "tb_attention_head_scratchpad_banked: PASS",
        "tb_attention_head_scratchpad_qk_combined: PASS",
    )
    testbench.write_text(source, encoding="utf-8")
    build = tmp_path / "tb_attention_head_scratchpad_qk_combined"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "tb_attention_head_scratchpad_qk_combined",
            "-o",
            str(build),
            str(RTL / "attention_head_scratchpad_qk_combined_banked.sv"),
            str(testbench),
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
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_attention_head_scratchpad_qk_combined: PASS" in run_result.stdout
