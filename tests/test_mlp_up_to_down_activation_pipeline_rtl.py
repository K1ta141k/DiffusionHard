from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_mlp_up_to_down_activation_pipeline_elaborates() -> None:
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog is required")
    build = Path("/tmp/mlp_up_to_down_activation_pipeline")
    sources = [
        RTL / "int8_mac_tile_pipelined.sv",
        RTL / "mlp_tile_pingpong_controller.sv",
        RTL / "fixed_requantize.sv",
        RTL / "gelu_q10_lut_scalar_bram.sv",
        RTL / "mlp_up_postprocess_serial.sv",
        RTL / "mlp_up_pingpong_pipeline.sv",
        RTL / "smoothquant_int8_vector_serial.sv",
        RTL / "mlp_interstage_tile_bridge_bram.sv",
        RTL / "mlp_interstage_pipeline.sv",
        RTL / "mlp_up_to_down_activation_pipeline.sv",
    ]
    result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "mlp_up_to_down_activation_pipeline",
            "-o",
            str(build),
            *(str(source) for source in sources),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
