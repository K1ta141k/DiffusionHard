from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_wide_automatic_mlp_pairs_token_groups(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    source = (RTL / "tb_hidden_canvas_automatic_mlp_block.sv").read_text(
        encoding="utf-8"
    )
    source = source.replace(
        "module tb_hidden_canvas_automatic_mlp_block;",
        "module tb_hidden_canvas_automatic_mlp_block_wide;",
    ).replace(
        "localparam integer CLIENT_TAG_WIDTH=12;",
        "localparam integer CLIENT_TAG_WIDTH=14;",
    ).replace(
        ".TOKENS(4),.DOWN_INPUT_SIZE(768),.DOWN_OUTPUT_SIZE(12),",
        ".TOKENS(8),.DOWN_INPUT_SIZE(768),.DOWN_OUTPUT_SIZE(12),"
        ".MLP_M_LANES(8),",
    ).replace(
        "wire [1023:0] array_activations;",
        "wire [2047:0] array_activations;",
    ).replace(
        "wire [767:0] array_accumulators;",
        "wire [1535:0] array_accumulators;",
    ).replace(
        ".M_LANES(4),.N_LANES(6),.DATA_WIDTH(8),.ACC_WIDTH(32),",
        ".M_LANES(8),.N_LANES(6),.DATA_WIDTH(8),.ACC_WIDTH(32),",
    ).replace(
        "if(output_tile!==outputs_seen || output_group!==0 || outputs!==0)",
        "if(output_tile!==(outputs_seen/2) || "
        "output_group!==(outputs_seen%2) || outputs!==0)",
    ).replace(
        "if(canvas_reads!=386 || up_weights!=3072 || up_metadata!=128 ||",
        "if(canvas_reads!=772 || up_weights!=3072 || up_metadata!=128 ||",
    ).replace(
        "down_requests!=48 || outputs_seen!=2 || busy)",
        "down_requests!=48 || outputs_seen!=4 || busy)",
    ).replace(
        "tb_hidden_canvas_automatic_mlp_block: PASS",
        "tb_hidden_canvas_automatic_mlp_block_wide: PASS",
    ).replace(
        "endmodule\n",
        "endmodule\n",
        1,
    )
    testbench = tmp_path / "tb_hidden_canvas_automatic_mlp_block_wide.sv"
    testbench.write_text(source, encoding="utf-8")
    sources = [
        "unsigned_divider_iterative.sv",
        "unsigned_sqrt_iterative.sv",
        "layer_norm_q12_group.sv",
        "mlp_up_activation_quantizer.sv",
        "layer_norm_mlp_up_activation_frontend.sv",
        "hidden_canvas_group_replay.sv",
        "hidden_canvas_mlp_frontend.sv",
        "mlp_token_pair_adapters.sv",
        "int8_mac_tile_pipelined.sv",
        "mlp_tile_pingpong_controller.sv",
        "wide_synchronous_uram.sv",
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
        "hidden_canvas_mlp_wide_shared_pipeline.sv",
        "mlp_tile_load_sequencer.sv",
        "hidden_canvas_residual_load_sequencer.sv",
        "mlp_block_controller.sv",
        "hidden_canvas_automatic_mlp_block.sv",
    ]
    build = tmp_path / "tb_hidden_canvas_automatic_mlp_block_wide"
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s",
            "tb_hidden_canvas_automatic_mlp_block_wide", "-o", str(build),
            *(str(RTL / source_name) for source_name in sources), str(testbench),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        ["vvp", str(build)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_hidden_canvas_automatic_mlp_block_wide: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) < 20_000
