import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_ordered_noisy_argmax_reducer_rtl() -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        pytest.skip("Icarus Verilog is not installed")

    build = ROOT / ".pytest-cache-ordered-noisy-argmax-reducer-rtl"
    subprocess.run(
        [
            iverilog,
            "-g2012",
            "-s",
            "tb_ordered_noisy_argmax_reducer",
            "-o",
            str(build),
            str(ROOT / "rtl/output_head/ordered_noisy_argmax_reducer.sv"),
            str(ROOT / "rtl/output_head/tb_ordered_noisy_argmax_reducer.sv"),
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
