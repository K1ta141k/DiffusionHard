from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.fixed_attention import fixed_attention_q12, fixed_rotary_q12
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


def test_attention_qk_softmax_group_matches_h0_rtl(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    if not golden_path.is_file() or not weights_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    name = "folded.block_00.qkv"
    qkv = _load_tensors(golden_path, [name])[name][0]
    tables = _load_tensors(weights_path, ["rotary.cos", "rotary.sin"])
    cosine = tables["rotary.cos"].float()
    sine = tables["rotary.sin"].float()
    _, _, details = fixed_attention_q12(qkv, cosine, sine)
    _, _, query_q, key_q, rotary_details = fixed_rotary_q12(
        qkv, cosine, sine
    )
    value_q = rotary_details["tensors"]["qkv_q12"][:, 2]
    probabilities = details["tensors"]["probabilities_q16"][0, :4]

    qkv_hex = tmp_path / "qk_softmax_head0_qkv.hex"
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
    expected_hex = tmp_path / "qk_softmax_expected.hex"
    expected_hex.write_text(
        "\n".join(
            f"{int(probabilities[query, key]):04x}"
            for query in range(4)
            for key in range(64)
        )
        + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_attention_qk_softmax_group_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_attention_qk_softmax_group_h0;
  reg clk=0,rst_n=0,load_valid=0,start=0,probability_group_ready=1;
  reg [5:0] load_token=0,load_channel=0;
  reg signed [17:0] load_q=0,load_k=0,load_v=0;
  wire q_read_valid,k_read_valid,q_data_valid,k_data_valid;
  wire [5:0] q_read_token,k_read_token;
  wire [1:0] q_read_block,k_read_block;
  wire [287:0] q_data,k_data;
  wire start_ready,score_valid,score_ready,qk_busy,qk_done;
  wire [3:0] score_group,score_key_tile;
  wire [2:0] score_valid_keys;
  wire [431:0] scores_packed;
  wire probability_group_valid,softmax_busy,softmax_done;
  wire [3:0] probability_group;
  wire [4095:0] probabilities_packed;
  reg [53:0] qkv_mem [0:4095];
  reg [15:0] expected_mem [0:255];
  integer index,query,key;

  attention_head_scratchpad_banked scratch(
    .clk(clk),.load_valid(load_valid),.load_token(load_token),
    .load_channel(load_channel),.load_query_q12(load_q),.load_key_q12(load_k),
    .load_value_q12(load_v),.query_read_valid(q_read_valid),
    .query_read_token(q_read_token),.query_read_channel_block(q_read_block),
    .query_data_valid(q_data_valid),.query_data_packed(q_data),
    .key_read_valid(k_read_valid),.key_read_token(k_read_token),
    .key_read_channel_block(k_read_block),.key_data_valid(k_data_valid),
    .key_data_packed(k_data),.value_read_valid(1'b0),.value_read_key_block(2'b0),
    .value_read_channel(6'b0),.value_data_valid(),.value_data_packed());
  attention_qk_group_scheduler qk(
    .clk(clk),.rst_n(rst_n),.start(start),.group_in(4'd0),
    .start_ready(start_ready),.query_read_valid(q_read_valid),
    .query_read_token(q_read_token),.query_read_channel_block(q_read_block),
    .query_data_valid(q_data_valid),.query_data_packed(q_data),
    .key_read_valid(k_read_valid),.key_read_token(k_read_token),
    .key_read_channel_block(k_read_block),.key_data_valid(k_data_valid),
    .key_data_packed(k_data),.score_valid(score_valid),.score_ready(score_ready),
    .score_group(score_group),.score_key_tile(score_key_tile),
    .score_valid_keys(score_valid_keys),.scores_q10_packed(scores_packed),
    .busy(qk_busy),.done(qk_done));
  attention_score_group_softmax softmax_group(
    .clk(clk),.rst_n(rst_n),.score_tile_valid(score_valid),
    .score_tile_ready(score_ready),.score_tile_group(score_group),
    .score_key_tile(score_key_tile),.score_valid_keys(score_valid_keys),
    .scores_q10_packed(scores_packed),
    .probability_group_valid(probability_group_valid),
    .probability_group_ready(probability_group_ready),
    .probability_group(probability_group),
    .probabilities_q16_packed(probabilities_packed),
    .busy(softmax_busy),.done(softmax_done));
  always #2 clk=~clk;
  initial begin
    $readmemh("{qkv_hex}",qkv_mem);$readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(index=0;index<4096;index=index+1) begin
      @(negedge clk);load_valid=1;load_token=index/64;load_channel=index%64;
      load_q=qkv_mem[index][17:0];load_k=qkv_mem[index][35:18];
      load_v=qkv_mem[index][53:36];
    end
    @(negedge clk);load_valid=0;start=1;@(negedge clk);start=0;
    wait(probability_group_valid);#1;
    if(probability_group!==0) $fatal(1,"probability group mismatch");
    for(query=0;query<4;query=query+1) for(key=0;key<64;key=key+1)
      if(probabilities_packed[(query*64+key)*16 +: 16]!==
         expected_mem[query*64+key])
        $fatal(1,"H0 probability mismatch query %0d key %0d",query,key);
    @(posedge clk);#1;repeat(2) @(posedge clk);
    if(!softmax_done) begin end
    $display("tb_attention_qk_softmax_group_h0: PASS");$finish;
  end
  initial begin repeat(12000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_attention_qk_softmax_group_h0"
    sources = [
        "attention_head_scratchpad_banked.sv",
        "int8_mac_tile_pipelined.sv",
        "mixed_precision_mac_tile_pipelined.sv",
        "attention_qk_group_scheduler.sv",
        "unsigned_divider_iterative.sv",
        "exp_neg_q16_lut_bram.sv",
        "attention_softmax_row_q16.sv",
        "attention_score_group_softmax.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-s", "tb_attention_qk_softmax_group_h0",
            "-o", str(build), *(str(RTL / name) for name in sources),
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
    assert "tb_attention_qk_softmax_group_h0: PASS" in run_result.stdout
