from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.fixed_attention import (
    fixed_attention_q12,
    fixed_qkv_projection_q12,
    fixed_rotary_q12,
)
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


def test_fixed_qkv_rotary_writeback_drives_exact_attention_head(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    if not golden_path.is_file() or not weights_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    goldens = _load_tensors(golden_path, ["folded.embedding"])
    weights = _load_tensors(
        weights_path,
        [
            "block_00.qkv.weight",
            "block_00.qkv.bias",
            "rotary.cos",
            "rotary.sin",
        ],
    )
    _, normalized_q12, _ = fixed_layer_norm_q12(goldens["folded.embedding"][0])
    fixed_qkv, fixed_qkv_q12, _ = fixed_qkv_projection_q12(
        normalized_q12,
        weights["block_00.qkv.weight"],
        weights["block_00.qkv.bias"],
    )
    cosine = weights["rotary.cos"].float()
    sine = weights["rotary.sin"].float()
    _, _, _, _, rotary_details = fixed_rotary_q12(fixed_qkv, cosine, sine)
    _, attention_q12, _ = fixed_attention_q12(fixed_qkv, cosine, sine)
    qkv_view = fixed_qkv_q12.view(64, 3, 12, 64)
    cosine_q15 = rotary_details["tensors"]["cosine_q15"]
    sine_q15 = rotary_details["tensors"]["sine_q15"]

    unrotated_hex = tmp_path / "fixed_qkv_head0_unrotated.hex"
    unrotated_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(qkv_view[token, 0, 0, channel]),
                    int(qkv_view[token, 1, 0, channel]),
                    int(qkv_view[token, 2, 0, channel]),
                ],
                18,
            )
            for token in range(64)
            for channel in range(64)
        )
        + "\n",
        encoding="utf-8",
    )
    constants_hex = tmp_path / "fixed_qkv_rotary_constants.hex"
    constants_hex.write_text(
        "\n".join(
            _packed_hex(
                [int(cosine_q15[token, pair]), int(sine_q15[token, pair])],
                16,
            )
            for token in range(64)
            for pair in range(32)
        )
        + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "fixed_qkv_attention_head0.hex"
    expected_hex.write_text(
        "\n".join(
            f"{int(attention_q12[token, channel]) & 0x3FFFF:05x}"
            for token in range(64)
            for channel in range(64)
        )
        + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_rotary_to_attention_head_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_rotary_to_attention_head_h0;
  reg clk=0,rst_n=0,rotary_start=0,head_start=0,qk_load=0,v_load=0;
  reg constant_load=0,attention_ready=1;
  reg [5:0] source_token=0,source_channel=0,constant_token=0;
  reg [4:0] constant_pair=0;
  reg signed [17:0] source_q=0,source_k=0,source_v=0;
  reg signed [15:0] load_cos=0,load_sin=0;
  wire rotary_start_ready,qk_read_valid,qk_data_valid,constant_read_valid;
  wire constant_data_valid,query_write,key_write,rotary_busy,rotary_done;
  wire [5:0] qk_read_token,qk_token_out,constant_read_token;
  wire [5:0] constant_token_out,write_token,write_channel;
  wire [4:0] qk_read_pair,qk_pair_out,constant_read_pair,constant_pair_out;
  wire signed [17:0] q_first,q_second,k_first,k_second,write_q,write_k;
  wire signed [15:0] cosine,sine;
  wire head_start_ready,attention_valid,head_busy,head_done;
  wire [3:0] attention_group,attention_output_tile;
  wire [2:0] attention_valid_channels;
  wire [431:0] attention_data;
  reg [53:0] qkv_mem [0:4095];
  reg [31:0] constant_mem [0:2047];
  reg [17:0] expected_mem [0:4095];
  integer index,tile_count=0,token_lane,lane,token,channel,cycle_count=0;

  qk_unrotated_scratchpad_banked unrotated(
    .clk(clk),.query_load_valid(qk_load),.key_load_valid(qk_load),
    .load_token(source_token),.load_channel(source_channel),
    .load_query_q12(source_q),.load_key_q12(source_k),
    .read_valid(qk_read_valid),.read_token(qk_read_token),
    .read_pair(qk_read_pair),.read_data_valid(qk_data_valid),
    .read_token_out(qk_token_out),.read_pair_out(qk_pair_out),
    .query_first_q12(q_first),.query_second_q12(q_second),
    .key_first_q12(k_first),.key_second_q12(k_second));
  rotary_constant_table_bram constants(
    .clk(clk),.load_valid(constant_load),.load_token(constant_token),
    .load_pair(constant_pair),.load_cosine_q15(load_cos),.load_sine_q15(load_sin),
    .read_valid(constant_read_valid),.read_token(constant_read_token),
    .read_pair(constant_read_pair),.read_data_valid(constant_data_valid),
    .read_token_out(constant_token_out),.read_pair_out(constant_pair_out),
    .cosine_q15(cosine),.sine_q15(sine));
  rotary_head_writeback_scheduler rotary(
    .clk(clk),.rst_n(rst_n),.start(rotary_start),
    .start_ready(rotary_start_ready),.head_in(0),
    .qk_read_valid(qk_read_valid),.qk_read_token(qk_read_token),
    .qk_read_pair(qk_read_pair),.qk_read_data_valid(qk_data_valid),
    .qk_read_token_out(qk_token_out),.qk_read_pair_out(qk_pair_out),
    .query_first_q12(q_first),.query_second_q12(q_second),
    .key_first_q12(k_first),.key_second_q12(k_second),
    .constant_read_valid(constant_read_valid),
    .constant_read_token(constant_read_token),
    .constant_read_pair(constant_read_pair),
    .constant_read_data_valid(constant_data_valid),
    .constant_read_token_out(constant_token_out),
    .constant_read_pair_out(constant_pair_out),.cosine_q15(cosine),.sine_q15(sine),
    .query_write_valid(query_write),.key_write_valid(key_write),
    .write_token(write_token),.write_channel(write_channel),
    .write_query_q12(write_q),.write_key_q12(write_k),
    .busy(rotary_busy),.done(rotary_done));
  attention_head_pipeline head(
    .clk(clk),.rst_n(rst_n),.load_valid(1'b0),
    .query_load_valid(query_write),.key_load_valid(key_write),
    .value_load_valid(v_load),
    .load_token(rotary_busy?write_token:source_token),
    .load_channel(rotary_busy?write_channel:source_channel),
    .load_query_q12(write_q),.load_key_q12(write_k),.load_value_q12(source_v),
    .start(head_start),.start_ready(head_start_ready),
    .attention_tile_valid(attention_valid),.attention_tile_ready(attention_ready),
    .attention_group(attention_group),.attention_output_tile(attention_output_tile),
    .attention_valid_channels(attention_valid_channels),
    .attention_q12_packed(attention_data),.busy(head_busy),.done(head_done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    if(head_busy) cycle_count=cycle_count+1;
    #1;
    if(attention_valid) begin
      if(attention_group!==(tile_count/11) || attention_output_tile!==(tile_count%11))
        $fatal(1,"fixed-QKV attention tag mismatch");
      for(token_lane=0;token_lane<4;token_lane=token_lane+1)
        for(lane=0;lane<attention_valid_channels;lane=lane+1) begin
          token=attention_group*4+token_lane;channel=attention_output_tile*6+lane;
          if($signed(attention_data[(token_lane*6+lane)*18 +: 18])!==
             $signed(expected_mem[token*64+channel]))
            $fatal(1,"fixed-QKV attention mismatch token %0d channel %0d",token,channel);
        end
      tile_count=tile_count+1;
    end
  end

  initial begin
    $readmemh("{unrotated_hex}",qkv_mem);$readmemh("{constants_hex}",constant_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(index=0;index<4096;index=index+1) begin
      @(negedge clk);qk_load=1;v_load=1;source_token=index/64;source_channel=index%64;
      source_q=qkv_mem[index][17:0];source_k=qkv_mem[index][35:18];
      source_v=qkv_mem[index][53:36];
    end
    @(negedge clk);qk_load=0;v_load=0;
    for(index=0;index<2048;index=index+1) begin
      @(negedge clk);constant_load=1;constant_token=index/32;constant_pair=index%32;
      load_cos=constant_mem[index][15:0];load_sin=constant_mem[index][31:16];
    end
    @(negedge clk);constant_load=0;rotary_start=1;
    @(negedge clk);rotary_start=0;wait(rotary_done);
    @(negedge clk);head_start=1;@(negedge clk);head_start=0;
    wait(head_done);repeat(2) @(posedge clk);
    if(tile_count!=176) $fatal(1,"missing fixed-QKV attention tiles");
    if(head_busy) $fatal(1,"fixed-QKV attention head remained busy");
    $display("tb_rotary_to_attention_head_h0: PASS attention_cycles=%0d",cycle_count);
    $finish;
  end
  initial begin repeat(40000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_rotary_to_attention_head_h0"
    sources = [
        "qk_unrotated_scratchpad_banked.sv",
        "rotary_constant_table_bram.sv",
        "rotary_qk_pair_serial.sv",
        "rotary_head_writeback_scheduler.sv",
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
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-s", "tb_rotary_to_attention_head_h0",
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
    assert "tb_rotary_to_attention_head_h0: PASS" in run_result.stdout
    match = re.search(r"attention_cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) == 15056
