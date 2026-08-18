import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_integrated_philox_gumbel_farm_stream_rtl() -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        pytest.skip("Icarus Verilog is not installed")

    build = ROOT / ".pytest-cache-philox-gumbel-farm-stream-rtl"
    sources = [
        "philox4x32_iterative.sv",
        "philox4x32_farm.sv",
        "gumbel_q10_dual.sv",
        "philox_gumbel_farm_stream.sv",
        "tb_philox_gumbel_farm_stream.sv",
    ]
    subprocess.run(
        [
            iverilog,
            "-g2012",
            "-s",
            "tb_philox_gumbel_farm_stream",
            "-o",
            str(build),
            *[str(ROOT / "rtl/philox_gumbel" / source) for source in sources],
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
