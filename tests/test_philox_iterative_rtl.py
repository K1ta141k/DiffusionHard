import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_philox_iterative_rtl() -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        pytest.skip("Icarus Verilog is not installed")

    build = ROOT / ".pytest-cache-philox-iterative-rtl"
    subprocess.run(
        [
            iverilog,
            "-g2012",
            "-s",
            "tb_philox4x32_iterative",
            "-o",
            str(build),
            str(ROOT / "rtl/philox_gumbel/philox4x32_iterative.sv"),
            str(ROOT / "rtl/philox_gumbel/tb_philox4x32_iterative.sv"),
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
