from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_hidden_canvas_mlp_block_runs_automatically_on_one_mac() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_hidden_canvas_automatic_mlp_block")
    sources = [
        "unsigned_divider_iterative.sv",
        "unsigned_sqrt_iterative.sv",
        "layer_norm_q12_group.sv",
        "mlp_up_activation_quantizer.sv",
        "layer_norm_mlp_up_activation_frontend.sv",
        "hidden_canvas_group_replay.sv",
        "hidden_canvas_mlp_frontend.sv",
        "int8_mac_tile_pipelined.sv",
        "mlp_tile_pingpong_controller.sv",
        "fixed_requantize.sv",
        "fixed_requantize_vector_serial.sv",
        "gelu_q10_lut_scalar_bram.sv",
        "mlp_up_postprocess_serial.sv",
        "mlp_up_pingpong_pipeline.sv",
        "smoothquant_int8_vector_serial.sv",
        "mlp_interstage_tile_bridge_bram.sv",
        "mlp_interstage_pipeline.sv",
        "mlp_up_to_down_activation_pipeline.sv",
        "residual_add_saturating.sv",
        "mlp_down_pingpong_pipeline.sv",
        "mlp_shared_up_down_pipeline.sv",
        "hidden_canvas_mlp_shared_pipeline.sv",
        "mlp_tile_load_sequencer.sv",
        "hidden_canvas_residual_load_sequencer.sv",
        "mlp_block_controller.sv",
        "hidden_canvas_automatic_mlp_block.sv",
        "tb_hidden_canvas_automatic_mlp_block.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s",
            "tb_hidden_canvas_automatic_mlp_block", "-o", str(build),
            *(str(RTL / source) for source in sources),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        ["vvp", str(build)], cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_hidden_canvas_automatic_mlp_block: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) < 20_000
