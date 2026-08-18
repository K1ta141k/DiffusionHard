from pathlib import Path
import subprocess
import sys


def test_generated_gumbel_lut_is_current(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    generated = tmp_path / "gumbel_lut.hpp"
    subprocess.run(
        [
            sys.executable,
            str(project_root / "hls" / "rng" / "generate_gumbel_lut.py"),
            "--out",
            str(generated),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    checked_in = project_root / "hls" / "rng" / "gumbel_lut.hpp"
    assert generated.read_bytes() == checked_in.read_bytes()

    generated_mem = tmp_path / "gumbel_lut_q10.mem"
    subprocess.run(
        [
            sys.executable,
            str(project_root / "hls" / "rng" / "generate_gumbel_lut.py"),
            "--mem-out",
            str(generated_mem),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    checked_in_mem = (
        project_root / "rtl" / "philox_gumbel" / "gumbel_lut_q10.mem"
    )
    assert generated_mem.read_bytes() == checked_in_mem.read_bytes()
