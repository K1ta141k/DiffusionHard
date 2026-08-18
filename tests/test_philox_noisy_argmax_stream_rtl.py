import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_philox_noisy_argmax_stream_rtl() -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        pytest.skip("Icarus Verilog is not installed")

    build = ROOT / ".pytest-cache-philox-noisy-argmax-stream-rtl"
    sources = [
        ROOT / "rtl/philox_gumbel/philox4x32_iterative.sv",
        ROOT / "rtl/philox_gumbel/philox4x32_farm.sv",
        ROOT / "rtl/philox_gumbel/gumbel_q10_dual.sv",
        ROOT / "rtl/philox_gumbel/philox_gumbel_farm_stream.sv",
        ROOT / "rtl/output_head/ordered_noisy_argmax_reducer.sv",
        ROOT / "rtl/output_head/philox_noisy_argmax_stream.sv",
        ROOT / "rtl/output_head/tb_philox_noisy_argmax_stream.sv",
    ]
    subprocess.run(
        [
            iverilog,
            "-g2012",
            "-s",
            "tb_philox_noisy_argmax_stream",
            "-o",
            str(build),
            *[str(source) for source in sources],
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
