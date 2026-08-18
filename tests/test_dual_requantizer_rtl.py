import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_dual_requantizer_rtl() -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        pytest.skip("Icarus Verilog is not installed")

    build = ROOT / ".pytest-cache-dual-requantizer-rtl"
    subprocess.run(
        [
            iverilog,
            "-g2012",
            "-s",
            "tb_dual_requantizer_q20",
            "-o",
            str(build),
            str(ROOT / "rtl/output_head/dual_requantizer_q20.sv"),
            str(ROOT / "rtl/output_head/tb_dual_requantizer_q20.sv"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        completed = subprocess.run(
            [vvp, str(build)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "all checks passed" in completed.stdout
    finally:
        build.unlink(missing_ok=True)
