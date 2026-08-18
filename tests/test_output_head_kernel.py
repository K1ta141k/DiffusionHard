from pathlib import Path
import shutil
import subprocess

import pytest


def test_output_head_native_kernel(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")

    project_root = Path(__file__).resolve().parents[1]
    kernel_dir = project_root / "hls" / "output_head"
    executable = tmp_path / "output_head_test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(kernel_dir / "output_head.cpp"),
            str(kernel_dir / "output_head_test.cpp"),
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
    assert completed.stdout.strip() == "output_head_test: all checks passed"
