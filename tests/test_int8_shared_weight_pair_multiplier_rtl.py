from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_shared_weight_pair_multiplier_is_bit_exact(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    testbench = tmp_path / "tb_int8_shared_weight_pair_multiplier.sv"
    testbench.write_text(
        """`timescale 1ns/1ps
module tb_int8_shared_weight_pair_multiplier;
  reg clk=0,rst_n=0,valid_in=0;
  reg signed [7:0] activation_0=0,activation_1=0,weight=0;
  wire valid_out;wire signed [15:0] product_0,product_1;
  reg signed [7:0] expected_a0 [0:1];
  reg signed [7:0] expected_a1 [0:1];
  reg signed [7:0] expected_w [0:1];
  integer sent=0,seen=0,index,expected_index,a0,a1,w,seed=32'h13579bdf;
  int8_shared_weight_pair_multiplier dut(
    .clk(clk),.rst_n(rst_n),.valid_in(valid_in),
    .activation_0(activation_0),.activation_1(activation_1),.weight(weight),
    .valid_out(valid_out),.product_0(product_0),.product_1(product_1));
  always #2 clk=~clk;
  always @(posedge clk) begin
    if(valid_in) begin
      expected_a0[sent%2]=activation_0;
      expected_a1[sent%2]=activation_1;
      expected_w[sent%2]=weight;
      sent=sent+1;
    end
    #1;
    if(valid_out) begin
      expected_index=seen%2;
      if(product_0!==expected_a0[expected_index]*expected_w[expected_index] ||
         product_1!==expected_a1[expected_index]*expected_w[expected_index])
        $fatal(1,"packed pair mismatch a0=%0d a1=%0d w=%0d got=%0d,%0d",
          expected_a0[expected_index],expected_a1[expected_index],
          expected_w[expected_index],
          product_0,product_1);
      seen=seen+1;
    end
  end
  task issue(input integer va0,input integer va1,input integer vw);
    begin @(negedge clk);valid_in=1;activation_0=va0;
      activation_1=va1;weight=vw;end
  endtask
  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    issue(-128,-128,-128);issue(-128,127,127);
    issue(127,-128,-1);issue(127,127,1);
    issue(-1,0,-128);issue(0,1,127);
    issue(-127,126,-127);issue(126,-127,126);
    for(index=0;index<20000;index=index+1) begin
      a0=$random(seed);a1=$random(seed);w=$random(seed);
      issue(a0,a1,w);
    end
    @(negedge clk);valid_in=0;
    wait(seen==sent);repeat(3) @(posedge clk);
    if(seen!=20008) $fatal(1,"packed pair count %0d",seen);
    $display("tb_int8_shared_weight_pair_multiplier: PASS vectors=%0d",seen);
    $finish;
  end
  initial begin repeat(25000) @(posedge clk);
    $fatal(1,"timeout sent=%0d seen=%0d valid=%0d",sent,seen,valid_out);end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_int8_shared_weight_pair_multiplier"
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s",
            "tb_int8_shared_weight_pair_multiplier", "-o", str(build),
            str(RTL / "int8_shared_weight_pair_multiplier.sv"),
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
    assert "PASS vectors=20008" in run_result.stdout
