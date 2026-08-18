from pathlib import Path
import shutil
import subprocess

import pytest


def test_philox_rng_native_kernel(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")

    project_root = Path(__file__).resolve().parents[1]
    kernel_dir = project_root / "hls" / "rng"
    executable = tmp_path / "philox_rng_test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(kernel_dir / "philox_rng.cpp"),
            str(kernel_dir / "philox_rng_test.cpp"),
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
    assert completed.stdout.strip() == "philox_rng_test: all checks passed"
