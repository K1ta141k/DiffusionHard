from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_mixed_precision_pair_multiplier_is_exact_in_both_modes(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    testbench = tmp_path / "tb_mixed_precision_token_pair_multiplier.sv"
    testbench.write_text(
        """`timescale 1ns/1ps
module tb_mixed_precision_token_pair_multiplier;
  reg clk=0,rst_n=0,valid_in=0,mode=0;
  reg signed [17:0] aa=0,aw=0;reg signed [7:0] a0=0,a1=0,mw=0;
  wire valid_out,mode_out;wire signed [35:0] ap;
  wire signed [17:0] p0,p1;
  reg expected_mode [0:1];reg signed [17:0] eaa[0:1],eaw[0:1];
  reg signed [7:0] ea0[0:1],ea1[0:1],emw[0:1];
  integer sent=0,seen=0,i,slot,seed=32'h10293847;
  mixed_precision_token_pair_multiplier dut(.clk(clk),.rst_n(rst_n),
    .valid_in(valid_in),.narrow_int8_mode(mode),
    .attention_activation(aa),.attention_weight(aw),
    .mlp_activation_0(a0),.mlp_activation_1(a1),.mlp_weight(mw),
    .valid_out(valid_out),.narrow_int8_mode_out(mode_out),
    .attention_product(ap),.mlp_offset_product_0(p0),
    .mlp_offset_product_1(p1));
  always #2 clk=~clk;
  always @(posedge clk) begin
    if(valid_in) begin slot=sent%2;expected_mode[slot]=mode;eaa[slot]=aa;
      eaw[slot]=aw;ea0[slot]=a0;ea1[slot]=a1;emw[slot]=mw;sent=sent+1;end
    #1;
    if(valid_out) begin slot=seen%2;
      if(mode_out!==expected_mode[slot]) $fatal(1,"mode mismatch");
      if(mode_out) begin
        if(p0!==($signed(ea0[slot])+18'sd128)*$signed(emw[slot]) ||
           p1!==($signed(ea1[slot])+18'sd128)*$signed(emw[slot]))
          $fatal(1,"MLP offset product mismatch");
      end else if(ap!==$signed(eaa[slot])*$signed(eaw[slot]))
        $fatal(1,"attention product mismatch");
      seen=seen+1;
    end
  end
  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(i=0;i<20000;i=i+1) begin @(negedge clk);valid_in=1;mode=i[0];
      aa=$random(seed);aw=$random(seed);a0=$random(seed);a1=$random(seed);
      mw=$random(seed);end
    @(negedge clk);valid_in=0;wait(seen==sent);repeat(3) @(posedge clk);
    if(seen!=20000) $fatal(1,"mixed pair count");
    $display("tb_mixed_precision_token_pair_multiplier: PASS vectors=%0d",seen);
    $finish;
  end
  initial begin repeat(21000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_mixed_precision_token_pair_multiplier"
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s",
            "tb_mixed_precision_token_pair_multiplier", "-o", str(build),
            str(RTL / "mixed_precision_token_pair_multiplier.sv"),
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
        timeout=30,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "PASS vectors=20000" in run_result.stdout
