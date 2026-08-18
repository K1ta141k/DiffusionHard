from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def _compile_and_run(top: str, sources: list[Path], build: Path) -> str:
    completed = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            top,
            "-o",
            str(build),
            *(str(source) for source in sources),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    run = subprocess.run(
        ["vvp", str(build)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return run.stdout


def test_fixed_requantize_rtl() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    stdout = _compile_and_run(
        "tb_fixed_requantize",
        [RTL / "fixed_requantize.sv", RTL / "tb_fixed_requantize.sv"],
        Path("/tmp/tb_fixed_requantize"),
    )
    assert "tb_fixed_requantize: PASS" in stdout


def test_gelu_q10_lut_is_current_and_accurate() -> None:
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(RTL / "generate_gelu_q10_lut.py"),
            "--out",
            str(RTL / "gelu_q10_lut.hex"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )
    words = [int(line, 16) for line in (RTL / "gelu_q10_lut.hex").read_text().splitlines()]
    assert len(words) == 1024
    signed = [word if word < 0x8000 else word - 0x10000 for word in words]
    maximum_error = 0.0
    coefficient = math.sqrt(2.0 / math.pi)
    for input_q in range(-8191, 8192):
        value = input_q / 1024.0
        expected = 0.5 * value * (
            1.0 + math.tanh(coefficient * (value + 0.044715 * value**3))
        )
        address = (input_q + 8192) >> 4
        actual = signed[address] / 1024.0
        maximum_error = max(maximum_error, abs(actual - expected))
    assert maximum_error < 0.019


def test_gelu_q10_lut_rtl() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    stdout = _compile_and_run(
        "tb_gelu_q10_lut",
        [RTL / "gelu_q10_lut.sv", RTL / "tb_gelu_q10_lut.sv"],
        Path("/tmp/tb_gelu_q10_lut"),
    )
    assert "tb_gelu_q10_lut: PASS" in stdout


def test_gelu_q10_lut_scalar_bram_rtl() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    stdout = _compile_and_run(
        "tb_gelu_q10_lut_scalar_bram",
        [
            RTL / "gelu_q10_lut_scalar_bram.sv",
            RTL / "tb_gelu_q10_lut_scalar_bram.sv",
        ],
        Path("/tmp/tb_gelu_q10_lut_scalar_bram"),
    )
    assert "tb_gelu_q10_lut_scalar_bram: PASS" in stdout
