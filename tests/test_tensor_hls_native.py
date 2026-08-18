import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_tensor_tile_native_cpp(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is not installed")
    executable = tmp_path / "int8_mac_tile_test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(ROOT / "hls/tensor_engine/int8_mac_tile.cpp"),
            str(ROOT / "hls/tensor_engine/int8_mac_tile_test.cpp"),
            "-o",
            str(executable),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(executable)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "int8_mac_tile_test: all checks passed" in completed.stdout
