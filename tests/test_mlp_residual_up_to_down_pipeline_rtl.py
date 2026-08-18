from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_mlp_residual_up_to_down_pipeline_elaborates() -> None:
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog is required")
    build = Path("/tmp/mlp_residual_up_to_down_pipeline")
    names = [
        "unsigned_divider_iterative.sv",
        "unsigned_sqrt_iterative.sv",
        "layer_norm_q12_group.sv",
        "mlp_up_activation_quantizer.sv",
        "layer_norm_mlp_up_activation_frontend.sv",
        "int8_mac_tile_pipelined.sv",
        "mlp_tile_pingpong_controller.sv",
        "fixed_requantize.sv",
        "gelu_q10_lut_scalar_bram.sv",
        "mlp_up_postprocess_serial.sv",
        "mlp_up_pingpong_pipeline.sv",
        "smoothquant_int8_vector_serial.sv",
        "mlp_interstage_tile_bridge_bram.sv",
        "mlp_interstage_pipeline.sv",
        "mlp_up_to_down_activation_pipeline.sv",
        "mlp_residual_up_to_down_pipeline.sv",
    ]
    result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "mlp_residual_up_to_down_pipeline",
            "-o",
            str(build),
            *(str(RTL / name) for name in names),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
