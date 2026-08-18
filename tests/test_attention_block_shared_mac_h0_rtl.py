from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from diffusion_accel.fixed_attention import (
    fixed_attention_projection_q10,
    fixed_attention_q12,
    fixed_rotary_q12,
)
from diffusion_accel.fixed_mlp import _load_tensors


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


def test_attention_and_projection_share_one_mac_on_h0(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    if not golden_path.is_file() or not weights_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    goldens = _load_tensors(
        golden_path, ["folded.block_00.qkv", "folded.embedding"]
    )
    weights = _load_tensors(
        weights_path,
        ["rotary.cos", "rotary.sin", "block_00.attention_out.weight"],
    )
    qkv = goldens["folded.block_00.qkv"][0]
    cosine = weights["rotary.cos"].float()
    sine = weights["rotary.sin"].float()
    _, attention_q12, _ = fixed_attention_q12(qkv, cosine, sine)
    _, _, query_q, key_q, rotary_details = fixed_rotary_q12(qkv, cosine, sine)
    value_q = rotary_details["tensors"]["qkv_q12"][:, 2]
    one_head_canvas = torch.zeros_like(attention_q12)
    one_head_canvas[:, :64] = attention_q12[:, :64]
    _, projection_q10, projection_details = fixed_attention_projection_q10(
        one_head_canvas, weights["block_00.attention_out.weight"]
    )
    residual_q10 = torch.round(
        goldens["folded.embedding"][0].double() * 1024.0
    ).to(torch.int64).clamp(-(1 << 23), (1 << 23) - 1)
    expected_q10 = (residual_q10 + projection_q10).clamp(
        -(1 << 23), (1 << 23) - 1
    )
    weight_q = projection_details["tensors"]["weight_int8"][:6]
    multipliers = projection_details["tensors"][
        "requant_multiplier_q24"
    ][:6]

    qkv_hex = tmp_path / "shared_mac_head0_qkv.hex"
    qkv_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(query_q[token, 0, channel]),
                    int(key_q[token, 0, channel]),
                    int(value_q[token, 0, channel]),
                ],
                18,
            )
            for token in range(64)
            for channel in range(64)
        )
        + "\n",
        encoding="utf-8",
    )
    weight_hex = tmp_path / "shared_mac_projection_weights.hex"
    weight_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(weight_q[output_lane, input_tile * 32 + input_lane])
                    for output_lane in range(6)
                    for input_lane in range(32)
                ],
                8,
            )
            for input_tile in range(24)
        )
        + "\n",
        encoding="utf-8",
    )
    residual_hex = tmp_path / "shared_mac_residual.hex"
    residual_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(residual_q10[group * 4 + token_lane, tile * 6 + lane])
                    for token_lane in range(4)
                    for lane in range(6)
                ],
                24,
            )
            for group in range(16)
            for tile in range(128)
        )
        + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "shared_mac_expected.hex"
    expected_hex.write_text(
        "\n".join(
            _packed_hex(
                [int(expected_q10[token, lane]) for lane in range(6)], 24
            )
            for token in range(64)
        )
        + "\n",
        encoding="utf-8",
    )
    multiplier_hex = _packed_hex([int(value) for value in multipliers], 24)

    testbench = tmp_path / "tb_attention_block_shared_mac_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_attention_block_shared_mac_h0;
  reg clk=0,rst_n=0,start=0,qkv_valid=0,residual_valid=0;
  reg metadata_valid=0,weight_valid=0,block_ready=1;
  reg [3:0] qkv_head=0,residual_group=0;
  reg [5:0] qkv_token=0,qkv_channel=0;
  reg signed [17:0] q=0,k=0,v=0;
  reg [6:0] residual_tile=0,metadata_tile=0,weight_output_tile=0;
  reg [4:0] weight_input_tile=0;
  reg [575:0] residual_data=0;
  reg [143:0] metadata_multipliers=144'h{multiplier_hex};
  reg [1535:0] weight_data=0;
  wire start_ready,qkv_ready,metadata_ready,weight_ready,block_valid,busy,done;
  wire [3:0] expected_head,block_group;
  wire [6:0] requested_tile,block_output_tile;
  wire [575:0] block_data;
  reg [53:0] qkv_mem [0:4095];
  reg [1535:0] weight_mem [0:23];
  reg [575:0] residual_mem [0:2047];
  reg [143:0] expected_mem [0:63];
  integer index,input_tile,residual_index,tile_count=0,token_lane,lane,token;
  integer cycle_count=0,attention_requests=0,projection_requests=0;

  attention_block_shared_mac_pipeline #(
    .HEADS(1),.PROJECTION_OUTPUT_TILES(1)
  ) dut(
    .clk(clk),.rst_n(rst_n),.start(start),.start_ready(start_ready),
    .qkv_load_valid(qkv_valid),.qkv_load_ready(qkv_ready),
    .qkv_load_head(qkv_head),.qkv_load_token(qkv_token),
    .qkv_load_channel(qkv_channel),.qkv_load_query_q12(q),
    .qkv_load_key_q12(k),.qkv_load_value_q12(v),.expected_head(expected_head),
    .residual_load_valid(residual_valid),.residual_load_group(residual_group),
    .residual_load_output_tile(residual_tile),
    .residual_load_q10_packed(residual_data),
    .projection_metadata_valid(metadata_valid),
    .projection_metadata_ready(metadata_ready),
    .projection_metadata_output_tile(metadata_tile),
    .projection_multipliers_packed(metadata_multipliers),
    .projection_weight_valid(weight_valid),
    .projection_weight_ready(weight_ready),
    .projection_weight_output_tile(weight_output_tile),
    .projection_weight_input_tile(weight_input_tile),
    .projection_weight_int8_packed(weight_data),
    .requested_projection_output_tile(requested_tile),
    .block_tile_valid(block_valid),.block_tile_ready(block_ready),
    .block_group(block_group),.block_output_tile(block_output_tile),
    .block_q10_packed(block_data),.busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    if(busy) cycle_count=cycle_count+1;
    if(dut.state==1 && dut.shared_array_valid) attention_requests=attention_requests+1;
    if(dut.state==2 && dut.shared_array_valid) projection_requests=projection_requests+1;
    #1;
    if(block_valid) begin
      if(block_group!==tile_count || block_output_tile!==0)
        $fatal(1,"shared block output tag mismatch");
      for(token_lane=0;token_lane<4;token_lane=token_lane+1)
        for(lane=0;lane<6;lane=lane+1) begin
          token=tile_count*4+token_lane;
          if($signed(block_data[(token_lane*6+lane)*24 +: 24])!==
             $signed(expected_mem[token][lane*24 +: 24]))
            $fatal(1,"shared block mismatch token %0d lane %0d",token,lane);
        end
      tile_count=tile_count+1;
    end
  end

  initial begin
    $readmemh("{qkv_hex}",qkv_mem);$readmemh("{weight_hex}",weight_mem);
    $readmemh("{residual_hex}",residual_mem);$readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(residual_index=0;residual_index<2048;residual_index=residual_index+1) begin
      @(negedge clk);residual_valid=1;residual_group=residual_index/128;
      residual_tile=residual_index%128;residual_data=residual_mem[residual_index];
    end
    @(negedge clk);residual_valid=0;start=1;@(negedge clk);start=0;
    wait(qkv_ready);
    for(index=0;index<4096;index=index+1) begin
      @(negedge clk);qkv_valid=1;qkv_token=index/64;qkv_channel=index%64;
      q=qkv_mem[index][17:0];k=qkv_mem[index][35:18];v=qkv_mem[index][53:36];
    end
    @(negedge clk);qkv_valid=0;metadata_valid=1;
    wait(metadata_ready);@(posedge clk);@(negedge clk);metadata_valid=0;
    for(input_tile=0;input_tile<24;input_tile=input_tile+1) begin
      @(negedge clk);weight_valid=1;weight_input_tile=input_tile;
      weight_data=weight_mem[input_tile];
      wait(weight_ready);@(posedge clk);@(negedge clk);weight_valid=0;
    end
    wait(done);repeat(2) @(posedge clk);
    if(tile_count!=16) $fatal(1,"missing shared block outputs");
    if(attention_requests==0 || projection_requests==0)
      $fatal(1,"both phases did not use the shared array");
    if(busy) $fatal(1,"shared block remained busy");
    $display("tb_attention_block_shared_mac_h0: PASS cycles=%0d attention_requests=%0d projection_requests=%0d",
      cycle_count,attention_requests,projection_requests);
    $finish;
  end
  initial begin repeat(35000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_attention_block_shared_mac_h0"
    sources = [
        "attention_head_scratchpad_banked.sv",
        "int8_mac_tile_pipelined.sv",
        "mixed_precision_mac_tile_pipelined.sv",
        "attention_qk_group_scheduler.sv",
        "unsigned_divider_iterative.sv",
        "exp_neg_q16_lut_bram.sv",
        "attention_softmax_row_q16.sv",
        "attention_score_group_softmax_stream.sv",
        "attention_pv_group_scheduler.sv",
        "attention_group_pipeline.sv",
        "attention_head_pipeline.sv",
        "attention_multihead_controller.sv",
        "attention_canvas_scratchpad_banked.sv",
        "attention_multihead_canvas_pipeline.sv",
        "fixed_requantize.sv",
        "fixed_requantize_vector_serial.sv",
        "attention_projection_weight_tile_buffer.sv",
        "attention_projection_output_tile_scheduler.sv",
        "attention_residual_canvas_uram.sv",
        "attention_projection_residual_output_tile.sv",
        "attention_projection_block_pipeline.sv",
        "attention_block_shared_mac_pipeline.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-s", "tb_attention_block_shared_mac_h0",
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
    assert "tb_attention_block_shared_mac_h0: PASS" in run_result.stdout
    match = re.search(
        r"cycles=(\d+) attention_requests=(\d+) projection_requests=(\d+)",
        run_result.stdout,
    )
    assert match is not None
    assert int(match.group(1)) == 21595
    assert int(match.group(2)) == 1056
    assert int(match.group(3)) == 384
