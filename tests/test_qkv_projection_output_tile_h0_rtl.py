from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from diffusion_accel.fixed_attention import fixed_qkv_projection_q12
from diffusion_accel.fixed_mlp import _load_tensors
from diffusion_accel.fixed_norm import fixed_layer_norm_q12


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"


def _packed_hex(values: list[int], width: int) -> str:
    mask = (1 << width) - 1
    packed = sum(
        (value & mask) << (index * width)
        for index, value in enumerate(values)
    )
    return f"{packed:0{(len(values) * width + 3) // 4}x}"


def test_qkv_projection_output_tile_matches_h0(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    if not golden_path.is_file() or not weights_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    goldens = _load_tensors(golden_path, ["folded.embedding"])
    weights = _load_tensors(
        weights_path, ["block_00.qkv.weight", "block_00.qkv.bias"]
    )
    _, normalized_q12, _ = fixed_layer_norm_q12(goldens["folded.embedding"][0])
    _, qkv_q12, details = fixed_qkv_projection_q12(
        normalized_q12,
        weights["block_00.qkv.weight"],
        weights["block_00.qkv.bias"],
    )
    weight_q = details["tensors"]["weight_int16"][:6]
    multipliers = details["tensors"]["requant_multiplier_q28"][:6]
    biases = details["tensors"]["bias_q12"][:6]

    normalized_hex = tmp_path / "qkv_normalized_q12.hex"
    normalized_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(normalized_q12[group * 4 + token_lane, tile * 32 + lane])
                    for token_lane in range(4)
                    for lane in range(32)
                ],
                18,
            )
            for group in range(16)
            for tile in range(24)
        )
        + "\n",
        encoding="utf-8",
    )
    weight_hex = tmp_path / "qkv_weight_tiles.hex"
    weight_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(weight_q[output_lane, tile * 32 + lane])
                    for output_lane in range(6)
                    for lane in range(32)
                ],
                16,
            )
            for tile in range(24)
        )
        + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "qkv_expected_q12.hex"
    expected_hex.write_text(
        "\n".join(
            _packed_hex([int(qkv_q12[token, lane]) for lane in range(6)], 18)
            for token in range(64)
        )
        + "\n",
        encoding="utf-8",
    )
    multiplier_hex = _packed_hex([int(value) for value in multipliers], 24)
    bias_hex = _packed_hex([int(value) for value in biases], 18)

    testbench = tmp_path / "tb_qkv_projection_output_tile_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_qkv_projection_output_tile_h0;
  reg clk=0,rst_n=0,start=0,metadata_valid=0,weight_valid=0,qkv_ready=1;
  reg [4:0] weight_input_tile=0;
  reg [3071:0] weight_data=0;
  wire start_ready,metadata_ready,weight_ready,norm_read_valid,norm_data_valid;
  wire qkv_valid,busy,done;
  wire [3:0] norm_group,qkv_group;
  wire [4:0] norm_input_tile;
  wire [8:0] qkv_output_tile;
  wire [3:0] requested_head,requested_channel_tile,qkv_head,qkv_channel_tile;
  wire [1:0] requested_kind,qkv_kind;
  wire [2:0] requested_valid_channels,qkv_valid_channels;
  wire [11:0] requested_global_row;
  reg norm_data_valid_reg=0;
  reg [2303:0] norm_data=0;
  wire [431:0] qkv_data;
  reg [2303:0] norm_mem [0:383];
  reg [3071:0] weight_mem [0:23];
  reg [107:0] expected_mem [0:63];
  integer tile,group_count=0,token_lane,lane,token,cycle_count=0;

  assign norm_data_valid=norm_data_valid_reg;
  qkv_head_projection_pipeline #(
    .KINDS(1),.CHANNEL_TILES(1),.LAST_TILE_VALID_CHANNELS(6)
  ) dut(
    .clk(clk),.rst_n(rst_n),.start(start),.start_ready(start_ready),
    .head_in(0),.metadata_valid(metadata_valid),.metadata_ready(metadata_ready),
    .metadata_head(0),.metadata_kind(0),.metadata_channel_tile(0),
    .metadata_multipliers_packed(144'h{multiplier_hex}),
    .metadata_biases_q12_packed(108'h{bias_hex}),
    .weight_tile_valid(weight_valid),
    .weight_tile_ready(weight_ready),.weight_input_tile(weight_input_tile),
    .weight_head(0),.weight_kind(0),.weight_channel_tile(0),
    .weight_int16_packed(weight_data),.normalized_read_valid(norm_read_valid),
    .requested_head(requested_head),.requested_kind(requested_kind),
    .requested_channel_tile(requested_channel_tile),
    .requested_valid_channels(requested_valid_channels),
    .requested_global_row(requested_global_row),
    .normalized_read_group(norm_group),
    .normalized_read_input_tile(norm_input_tile),
    .normalized_read_data_valid(norm_data_valid),
    .normalized_q12_packed(norm_data),.qkv_tile_valid(qkv_valid),
    .qkv_tile_ready(qkv_ready),.qkv_group(qkv_group),
    .qkv_head(qkv_head),.qkv_kind(qkv_kind),
    .qkv_channel_tile(qkv_channel_tile),
    .qkv_valid_channels(qkv_valid_channels),.qkv_q12_packed(qkv_data),
    .busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    norm_data_valid_reg<=norm_read_valid;
    if(norm_read_valid) norm_data<=norm_mem[norm_group*24+norm_input_tile];
    if(busy) cycle_count=cycle_count+1;
    #1;
    if(qkv_valid) begin
      if(qkv_group!==group_count || qkv_head!==0 || qkv_kind!==0 ||
         qkv_channel_tile!==0 || qkv_valid_channels!==6)
        $fatal(1,"QKV tile tag mismatch");
      for(token_lane=0;token_lane<4;token_lane=token_lane+1)
        for(lane=0;lane<6;lane=lane+1) begin
          token=group_count*4+token_lane;
          if($signed(qkv_data[(token_lane*6+lane)*18 +: 18])!==
             $signed(expected_mem[token][lane*18 +: 18]))
            $fatal(1,"QKV mismatch token %0d lane %0d",token,lane);
        end
      group_count=group_count+1;
    end
  end

  initial begin
    $readmemh("{normalized_hex}",norm_mem);$readmemh("{weight_hex}",weight_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    start=1;@(negedge clk);start=0;
    metadata_valid=1;wait(metadata_ready);@(posedge clk);
    @(negedge clk);metadata_valid=0;
    for(tile=0;tile<24;tile=tile+1) begin
      wait(weight_ready);@(negedge clk);weight_valid=1;
      weight_input_tile=tile;weight_data=weight_mem[tile];
    end
    @(negedge clk);weight_valid=0;
    wait(done);repeat(2) @(posedge clk);
    if(group_count!=16) $fatal(1,"missing QKV groups");
    if(busy) $fatal(1,"QKV scheduler remained busy");
    $display("tb_qkv_projection_output_tile_h0: PASS cycles=%0d",cycle_count);
    $finish;
  end
  initial begin repeat(5000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_qkv_projection_output_tile_h0"
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
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-s", "tb_qkv_projection_output_tile_h0",
            "-o", str(build), *(str(RTL / item) for item in sources),
            str(testbench),
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
    assert "tb_qkv_projection_output_tile_h0: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) == 428
