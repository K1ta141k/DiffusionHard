from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.fixed_attention import (
    fixed_attention_projection_q10,
    fixed_attention_q12,
)
from diffusion_accel.fixed_mlp import _load_tensors


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"
FULL_PROJECTION = os.environ.get("DIFFUSION_ACCEL_RUN_FULL_PROJECTION_RTL") == "1"
GROUPED_PROJECTION = os.environ.get(
    "DIFFUSION_ACCEL_RUN_GROUPED_PROJECTION_RTL"
) == "1"


def _packed_hex(values: list[int], width: int) -> str:
    mask = (1 << width) - 1
    packed = sum(
        (value & mask) << (index * width)
        for index, value in enumerate(values)
    )
    return f"{packed:0{(len(values) * width + 3) // 4}x}"


def test_attention_projection_residual_output_tile_matches_h0(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    if not golden_path.is_file() or not weights_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    goldens = _load_tensors(
        golden_path, ["folded.block_00.qkv", "folded.embedding"]
    )
    qkv = goldens["folded.block_00.qkv"][0]
    weights = _load_tensors(
        weights_path,
        ["rotary.cos", "rotary.sin", "block_00.attention_out.weight"],
    )
    _, attention_q12, _ = fixed_attention_q12(
        qkv, weights["rotary.cos"].float(), weights["rotary.sin"].float()
    )
    _, projection_q10, projection_details = fixed_attention_projection_q10(
        attention_q12, weights["block_00.attention_out.weight"]
    )
    residual_q10 = (
        goldens["folded.embedding"][0].double() * 1024.0
    ).round().to(dtype=projection_q10.dtype).clamp(-(1 << 23), (1 << 23) - 1)
    after_attention_q10 = (residual_q10 + projection_q10).clamp(
        -(1 << 23), (1 << 23) - 1
    )
    output_tiles = 128 if FULL_PROJECTION else 2
    output_channels = output_tiles * 6
    weight_q = projection_details["tensors"]["weight_int8"][:output_channels]
    multipliers = projection_details["tensors"][
        "requant_multiplier_q24"
    ][:output_channels]

    canvas_hex = tmp_path / "attention_canvas_q12.hex"
    canvas_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(attention_q12[token, head * 64 + channel])
                    for channel in range(64)
                ],
                18,
            )
            for head in range(12)
            for token in range(64)
        )
        + "\n",
        encoding="utf-8",
    )
    grouped_canvas_hex = tmp_path / "attention_canvas_grouped_q12.hex"
    grouped_canvas_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(attention_q12[group * 4 + token_lane, head * 64 + channel])
                    for token_lane in range(4)
                    for channel in range(64)
                ],
                18,
            )
            for head in range(12)
            for group in range(16)
        )
        + "\n",
        encoding="utf-8",
    )
    weight_hex = tmp_path / "attention_projection_weight_tiles.hex"
    weight_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(
                        weight_q[
                            output_tile * 6 + output_lane,
                            input_tile * 32 + input_lane,
                        ]
                    )
                    for output_lane in range(6)
                    for input_lane in range(32)
                ],
                8,
            )
            for output_tile in range(output_tiles)
            for input_tile in range(24)
        )
        + "\n",
        encoding="utf-8",
    )
    residual_hex = tmp_path / "attention_residual_q10.hex"
    residual_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(residual_q10[group * 4 + token_lane, output_tile * 6 + output_lane])
                    for token_lane in range(4)
                    for output_lane in range(6)
                ],
                24,
            )
            for group in range(16)
            for output_tile in range(128)
        )
        + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "attention_projection_expected_q10.hex"
    expected_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(after_attention_q10[token, channel])
                    for channel in range(output_channels)
                ],
                24,
            )
            for token in range(64)
        )
        + "\n",
        encoding="utf-8",
    )
    multiplier_path = tmp_path / "attention_projection_multipliers.hex"
    multiplier_path.write_text(
        "\n".join(
            _packed_hex(
                [int(value) for value in multipliers[tile * 6 : (tile + 1) * 6]],
                24,
            )
            for tile in range(output_tiles)
        )
        + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_attention_projection_output_tile_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_attention_projection_output_tile_h0;
  reg clk=0,rst_n=0,start=0,weight_valid=0,block_ready=1;
  reg metadata_valid=0;
  reg [6:0] metadata_output_tile=0,weight_output_tile=0;
  reg [143:0] metadata_multipliers=0;
  reg residual_load_valid=0;
  reg [3:0] residual_load_group=0;
  reg [6:0] residual_load_output_tile=0;
  reg [575:0] residual_load_data=0;
  reg [4:0] weight_input_tile=0;
  reg [1535:0] weight_packed=0;
  reg replay_read_valid=0;
  reg [3:0] replay_read_group=0;
  reg [6:0] replay_read_tile=0;
  wire replay_data_valid;
  wire [575:0] replay_data;
  wire start_ready,metadata_ready,weight_ready,canvas_read_valid,canvas_data_valid;
  wire [3:0] canvas_read_head,projection_group;
  wire [5:0] canvas_read_token;
  wire canvas_group_read_valid;wire [3:0] canvas_group_read_head;
  wire [3:0] canvas_group_read_group;reg canvas_group_data_valid=0;
  reg [4607:0] canvas_group_data=0;
  reg canvas_data_valid_reg=0;
  reg [1151:0] canvas_data=0;
  wire block_valid,busy,done;
  wire [3:0] block_group;
  wire [6:0] block_output_tile;
  wire [575:0] block_packed;
  reg [1151:0] canvas_mem [0:767];
  reg [4607:0] grouped_canvas_mem [0:191];
  reg [1535:0] weight_mem [0:{output_tiles * 24 - 1}];
  reg [{output_channels * 24 - 1}:0] expected_mem [0:63];
  reg [143:0] multiplier_mem [0:{output_tiles - 1}];
  reg [575:0] residual_mem [0:2047];
  integer tile,input_tile,residual_index,tile_count=0,token_lane,output_lane,token;
  integer cycle_count=0;

  assign canvas_data_valid=canvas_data_valid_reg;
  attention_projection_block_pipeline #(
    .OUTPUT_TILES({output_tiles}),.ENABLE_RESIDUAL_REPLAY(1),
    .GROUPED_CANVAS({1 if GROUPED_PROJECTION else 0})
  ) dut(
    .clk(clk),.rst_n(rst_n),.block_start(start),.block_start_ready(start_ready),
    .residual_load_valid(residual_load_valid),
    .residual_load_group(residual_load_group),
    .residual_load_output_tile(residual_load_output_tile),
    .residual_load_q10_packed(residual_load_data),
    .residual_replay_read_valid(replay_read_valid),
    .residual_replay_read_group(replay_read_group),
    .residual_replay_read_output_tile(replay_read_tile),
    .residual_replay_read_data_valid(replay_data_valid),
    .residual_replay_read_q10_packed(replay_data),
    .metadata_valid(metadata_valid),.metadata_ready(metadata_ready),
    .metadata_output_tile(metadata_output_tile),
    .metadata_multipliers_packed(metadata_multipliers),
    .weight_tile_valid(weight_valid),.weight_tile_ready(weight_ready),
    .weight_output_tile(weight_output_tile),
    .weight_input_tile(weight_input_tile),.weight_int8_packed(weight_packed),
    .requested_output_tile(),
    .canvas_read_valid(canvas_read_valid),.canvas_read_head(canvas_read_head),
    .canvas_read_token(canvas_read_token),
    .canvas_read_data_valid(canvas_data_valid),
    .canvas_read_data_packed(canvas_data),
    .canvas_group_read_valid(canvas_group_read_valid),
    .canvas_group_read_head(canvas_group_read_head),
    .canvas_group_read_group(canvas_group_read_group),
    .canvas_group_read_data_valid(canvas_group_data_valid),
    .canvas_group_read_data_packed(canvas_group_data),
    .block_tile_valid(block_valid),.block_tile_ready(block_ready),
    .block_group(block_group),.block_output_tile(block_output_tile),
    .block_q10_packed(block_packed),.busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    canvas_data_valid_reg<=canvas_read_valid;
    canvas_group_data_valid<=canvas_group_read_valid;
    if(canvas_read_valid)
      canvas_data<=canvas_mem[canvas_read_head*64+canvas_read_token];
    if(canvas_group_read_valid)
      canvas_group_data<=grouped_canvas_mem[
        canvas_group_read_head*16+canvas_group_read_group];
    if(busy) cycle_count=cycle_count+1;
    #1;
    if(block_valid) begin
      if(block_group!==(tile_count%16) || block_output_tile!==(tile_count/16))
        $fatal(1,"projection tile tag mismatch");
      for(token_lane=0;token_lane<4;token_lane=token_lane+1)
        for(output_lane=0;output_lane<6;output_lane=output_lane+1) begin
          token=(tile_count%16)*4+token_lane;
          if($signed(block_packed[(token_lane*6+output_lane)*24 +: 24])!==
             $signed(expected_mem[token][((tile_count/16)*6+output_lane)*24 +: 24]))
            $fatal(1,"projection mismatch token %0d output %0d",token,output_lane);
        end
      tile_count=tile_count+1;
    end
  end

  initial begin
    $readmemh("{canvas_hex}",canvas_mem);
    $readmemh("{grouped_canvas_hex}",grouped_canvas_mem);
    $readmemh("{weight_hex}",weight_mem);
    $readmemh("{expected_hex}",expected_mem);$readmemh("{residual_hex}",residual_mem);
    $readmemh("{multiplier_path}",multiplier_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(residual_index=0;residual_index<2048;residual_index=residual_index+1) begin
      @(negedge clk);residual_load_valid=1;
      residual_load_group=residual_index/128;
      residual_load_output_tile=residual_index%128;
      residual_load_data=residual_mem[residual_index];
    end
    @(negedge clk);residual_load_valid=0;
    start=1;@(negedge clk);start=0;
    for(tile=0;tile<{output_tiles};tile=tile+1) begin
      @(negedge clk);metadata_output_tile=tile;
      metadata_multipliers=multiplier_mem[tile];metadata_valid=1;
      wait(metadata_ready);@(posedge clk);@(negedge clk);metadata_valid=0;
      weight_output_tile=tile;
      for(input_tile=0;input_tile<24;input_tile=input_tile+1) begin
        @(negedge clk);weight_valid=1;
        weight_input_tile=input_tile;
        weight_packed=weight_mem[tile*24+input_tile];
        wait(weight_ready);@(posedge clk);@(negedge clk);weight_valid=0;
      end
    end
    wait(done);repeat(2) @(posedge clk);
    for(tile=0;tile<{output_tiles};tile=tile+1) begin
      @(negedge clk);replay_read_valid=1;replay_read_group=7;
      replay_read_tile=tile;
      @(posedge clk);#1;
      if(!replay_data_valid) $fatal(1,"post-attention replay valid missing");
      for(token_lane=0;token_lane<4;token_lane=token_lane+1)
        for(output_lane=0;output_lane<6;output_lane=output_lane+1)
          if($signed(replay_data[(token_lane*6+output_lane)*24 +: 24])!==
             $signed(expected_mem[28+token_lane][(tile*6+output_lane)*24 +: 24]))
            $fatal(1,"post-attention replay mismatch tile %0d lane %0d",
                   tile,output_lane);
    end
    @(negedge clk);replay_read_valid=0;
    if(tile_count!={output_tiles * 16}) $fatal(1,"missing projection groups");
    if(busy) $fatal(1,"projection scheduler remained busy");
    $display("tb_attention_projection_output_tile_h0: PASS cycles=%0d",cycle_count);
    $finish;
  end
  initial begin
    repeat({output_tiles * 4000}) @(posedge clk);
    $display("timeout controller=%0d output=%0d child=%0d requested=%0d weight_ready=%0d metadata_ready=%0d",
      dut.state,dut.active_output_tile,dut.output_tile_pipeline.projection.state,
      dut.requested_output_tile,weight_ready,metadata_ready);
    $fatal(1,"timeout");
  end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_attention_projection_output_tile_h0"
    sources = [
        "int8_mac_tile_pipelined.sv",
        "mixed_precision_mac_tile_pipelined.sv",
        "fixed_requantize.sv",
        "fixed_requantize_vector_serial.sv",
        "attention_projection_weight_tile_buffer.sv",
        "attention_projection_output_tile_scheduler.sv",
        "attention_residual_canvas_uram.sv",
        "attention_projection_residual_output_tile.sv",
        "attention_projection_block_pipeline.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-s",
            "tb_attention_projection_output_tile_h0", "-o", str(build),
            *(str(RTL / item) for item in sources), str(testbench),
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
    if FULL_PROJECTION or GROUPED_PROJECTION:
        print(run_result.stdout, end="")
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_attention_projection_output_tile_h0: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    cycles = int(match.group(1))
    if FULL_PROJECTION:
        assert 360_000 <= cycles <= 380_000
    elif GROUPED_PROJECTION:
        assert cycles == 1_413
    else:
        assert cycles == 5_221
