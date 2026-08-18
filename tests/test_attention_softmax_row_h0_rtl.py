from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.fixed_attention import fixed_attention_q12
from diffusion_accel.fixed_mlp import _load_tensors


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"


def test_attention_softmax_row_matches_h0_block0_rtl(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    if not golden_path.is_file() or not weights_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    name = "folded.block_00.qkv"
    qkv = _load_tensors(golden_path, [name])[name][0]
    tables = _load_tensors(weights_path, ["rotary.cos", "rotary.sin"])
    _, _, details = fixed_attention_q12(
        qkv,
        tables["rotary.cos"].float(),
        tables["rotary.sin"].float(),
    )
    scores = details["tensors"]["scores_q10"][0, 0]
    probabilities = details["tensors"]["probabilities_q16"][0, 0]
    scores_hex = tmp_path / "softmax_scores.hex"
    scores_hex.write_text(
        "\n".join(f"{int(value) & 0x3FFFF:05x}" for value in scores) + "\n",
        encoding="utf-8",
    )
    probabilities_hex = tmp_path / "softmax_probabilities.hex"
    probabilities_hex.write_text(
        "\n".join(f"{int(value):04x}" for value in probabilities) + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_attention_softmax_row_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_attention_softmax_row_h0;
  reg clk=0,rst_n=0,start=0,score_valid=0,probability_ready=1;
  reg [3:0] head_in=4'd0;
  reg [5:0] query_in=6'd0;
  reg signed [17:0] score_q10=0;
  wire start_ready,score_ready,probability_valid;
  wire [3:0] head_out;
  wire [5:0] query_out,key_out;
  wire [15:0] probability_q16;
  wire busy,done;
  reg [17:0] score_mem [0:63];
  reg [15:0] expected_mem [0:63];
  integer index,output_count=0;
  attention_softmax_row_q16 #(
    .LUT_FILE("rtl/tensor_engine/exp_neg_q16_lut.hex")
  ) dut(
    .clk(clk),.rst_n(rst_n),.start(start),.head_in(head_in),
    .query_in(query_in),.start_ready(start_ready),.score_valid(score_valid),
    .score_ready(score_ready),.score_q10(score_q10),
    .probability_valid(probability_valid),.probability_ready(probability_ready),
    .head_out(head_out),.query_out(query_out),.key_out(key_out),
    .probability_q16(probability_q16),.busy(busy),.done(done));
  always #2 clk=~clk;
  always @(posedge clk) begin
    #1;
    if(probability_valid && probability_ready) begin
      if(head_out!==4'd0 || query_out!==6'd0) $fatal(1,"row tag mismatch");
      if(key_out!==output_count) $fatal(1,"key tag mismatch");
      if(probability_q16!==expected_mem[output_count])
        $fatal(1,"H0 probability mismatch at key %0d got %0d expected %0d",
               output_count,probability_q16,expected_mem[output_count]);
      output_count=output_count+1;
    end
  end
  initial begin
    $readmemh("{scores_hex}",score_mem);
    $readmemh("{probabilities_hex}",expected_mem);
    repeat(3) @(posedge clk); @(negedge clk); rst_n=1; start=1;
    @(negedge clk); start=0;
    for(index=0;index<64;index=index+1) begin
      @(negedge clk);
      if(!score_ready) $fatal(1,"score input not ready");
      score_q10=score_mem[index]; score_valid=1;
    end
    @(negedge clk); score_valid=0;
    wait(done); repeat(2) @(posedge clk);
    if(output_count!=64) $fatal(1,"missing probability outputs");
    if(busy) $fatal(1,"softmax row remained busy");
    $display("tb_attention_softmax_row_h0: PASS"); $finish;
  end
  initial begin repeat(1000) @(posedge clk); $fatal(1,"timeout"); end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_attention_softmax_row_h0"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_attention_softmax_row_h0",
            "-o",
            str(build),
            str(RTL / "unsigned_divider_iterative.sv"),
            str(RTL / "exp_neg_q16_lut_bram.sv"),
            str(RTL / "attention_softmax_row_q16.sv"),
            str(testbench),
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
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_attention_softmax_row_h0: PASS" in run_result.stdout
