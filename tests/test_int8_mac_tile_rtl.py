import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_int8_mac_tile_rtl() -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        pytest.skip("Icarus Verilog is not installed")

    build = ROOT / ".pytest-cache-int8-mac-tile-rtl"
    subprocess.run(
        [
            iverilog,
            "-g2012",
            "-Wall",
            "-s",
            "tb_int8_mac_tile",
            "-o",
            str(build),
            str(ROOT / "rtl/tensor_engine/int8_mac_tile.sv"),
            str(ROOT / "rtl/tensor_engine/tb_int8_mac_tile.sv"),
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
        assert "tb_int8_mac_tile: PASS" in completed.stdout
    finally:
        build.unlink(missing_ok=True)
