from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_qkv_attention_multihead_producer_fills_canvas() -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    build = Path("/tmp/tb_qkv_attention_multihead_canvas_pipeline")
    sources = [
        "int8_mac_tile_pipelined.sv",
        "mixed_precision_mac_tile_pipelined.sv",
        "fixed_requantize.sv",
        "fixed_requantize_vector_serial.sv",
        "qkv_weight_tile_buffer.sv",
        "qkv_projection_output_tile_scheduler.sv",
        "qkv_projection_output_tile_scheduler_streaming.sv",
        "qkv_head_tile_controller.sv",
        "qkv_head_projection_pipeline.sv",
        "qkv_head_output_router.sv",
        "qk_unrotated_scratchpad_paired_uram.sv",
        "rotary_constant_table_bram.sv",
        "rotary_qk_pair_serial.sv",
        "rotary_head_writeback_scheduler.sv",
        "qkv_head_staging_pipeline.sv",
        "attention_head_scratchpad_banked.sv",
        "attention_qk_group_scheduler.sv",
        "unsigned_divider_iterative.sv",
        "exp_neg_q16_lut_bram.sv",
        "attention_softmax_row_q16.sv",
        "attention_score_group_softmax_stream.sv",
        "attention_pv_group_scheduler.sv",
        "attention_group_pipeline.sv",
        "attention_head_pipeline.sv",
        "qkv_attention_head_pipeline.sv",
        "attention_canvas_scratchpad_banked.sv",
        "attention_canvas_grouped_scratchpad_banked.sv",
        "qkv_attention_multihead_canvas_pipeline.sv",
        "tb_qkv_attention_multihead_canvas_pipeline.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall",
            "-s", "tb_qkv_attention_multihead_canvas_pipeline",
            "-o", str(build), *(str(RTL / item) for item in sources),
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
    assert "tb_qkv_attention_multihead_canvas_pipeline: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) < 90000
