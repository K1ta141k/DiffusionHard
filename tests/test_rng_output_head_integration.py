from pathlib import Path
import shutil
import subprocess

import pytest


def test_rng_output_head_native_integration(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")

    project_root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "rng_output_head_test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(project_root / "hls" / "rng" / "philox_rng.cpp"),
            str(project_root / "hls" / "output_head" / "output_head.cpp"),
            str(project_root / "hls" / "integration" / "rng_output_head_test.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "rng_output_head_test: all checks passed"
