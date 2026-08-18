from pathlib import Path
import shutil
import subprocess

import pytest


def test_philox_gumbel_systemverilog_testbench(tmp_path: Path) -> None:
    compiler = shutil.which("iverilog")
    runtime = shutil.which("vvp")
    if compiler is None or runtime is None:
        pytest.skip("Icarus Verilog is unavailable")

    project_root = Path(__file__).resolve().parents[1]
    rtl_dir = project_root / "rtl" / "philox_gumbel"
    simulation = tmp_path / "philox_gumbel_simulation"
    subprocess.run(
        [
            compiler,
            "-g2012",
            "-Wall",
            "-s",
            "tb_philox_gumbel_stream",
            "-o",
            str(simulation),
            str(rtl_dir / "philox_gumbel_stream.sv"),
            str(rtl_dir / "tb_philox_gumbel_stream.sv"),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [runtime, str(simulation)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "tb_philox_gumbel_stream: all checks passed" in completed.stdout
