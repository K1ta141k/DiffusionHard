from pathlib import Path
import shutil
import subprocess

import pytest


def test_candidate_sampler_native_cpp_testbench(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native C++ compiler is unavailable")

    project_root = Path(__file__).resolve().parents[1]
    kernel_dir = project_root / "hls" / "candidate_sampler"
    executable = tmp_path / "candidate_sampler_test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(kernel_dir / "candidate_sampler.cpp"),
            str(kernel_dir / "candidate_sampler_test.cpp"),
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
    assert completed.stdout.strip() == "candidate_sampler_test: all checks passed"
