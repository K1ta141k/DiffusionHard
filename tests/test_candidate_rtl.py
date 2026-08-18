from pathlib import Path
import shutil
import subprocess

import pytest


def test_candidate_reveal_systemverilog_testbench(tmp_path: Path) -> None:
    compiler = shutil.which("iverilog")
    runtime = shutil.which("vvp")
    if compiler is None or runtime is None:
        pytest.skip("Icarus Verilog is unavailable")

    project_root = Path(__file__).resolve().parents[1]
    rtl_dir = project_root / "rtl" / "candidate_reveal"
    simulation = tmp_path / "candidate_reveal_simulation"
    subprocess.run(
        [
            compiler,
            "-g2012",
            "-Wall",
            "-s",
            "tb_candidate_reveal_stream",
            "-o",
            str(simulation),
            str(rtl_dir / "candidate_reveal_stream.sv"),
            str(rtl_dir / "tb_candidate_reveal_stream.sv"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [runtime, str(simulation)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "tb_candidate_reveal_stream: all checks passed" in completed.stdout
