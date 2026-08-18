from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_packed_m8_mac_matches_conventional_array(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    testbench = tmp_path / "tb_int8_mac_tile_packed_m8.sv"
    testbench.write_text(
        """`timescale 1ns/1ps
module tb_int8_mac_tile_packed_m8;
  reg clk=0,rst_n=0,valid_in=0,clear=0,last=0;
  reg [7:0] tag_in=0;
  reg [8*32*8-1:0] activations=0;
  reg [6*32*8-1:0] weights=0;
  wire conventional_valid,packed_valid;
  wire [7:0] conventional_tag,packed_tag;
  wire [8*6*32-1:0] conventional_results,packed_results;
  reg [8*6*32-1:0] expected [0:15];reg expected_valid [0:15];
  integer dot,k,lane,seed=32'h2468ace1,conventional_seen=0,packed_seen=0;
  int8_mac_tile_pipelined #(.M_LANES(8),.N_LANES(6),.TAG_WIDTH(8)) reference(
    .clk(clk),.rst_n(rst_n),.valid_in(valid_in),
    .clear_accumulators(clear),.last_k_tile(last),.tag_in(tag_in),
    .activations_packed(activations),.weights_packed(weights),
    .valid_out(conventional_valid),.tag_out(conventional_tag),
    .accumulators_packed(conventional_results));
  int8_mac_tile_packed_m8_pipelined #(.N_LANES(6),.TAG_WIDTH(8)) dut(
    .clk(clk),.rst_n(rst_n),.valid_in(valid_in),
    .clear_accumulators(clear),.last_k_tile(last),.tag_in(tag_in),
    .activations_packed(activations),.weights_packed(weights),
    .valid_out(packed_valid),.tag_out(packed_tag),
    .accumulators_packed(packed_results));
  always #2 clk=~clk;
  always @(posedge clk) begin
    #1;
    if(conventional_valid) begin
      expected[conventional_tag]=conventional_results;
      expected_valid[conventional_tag]=1;
      conventional_seen=conventional_seen+1;
    end
    if(packed_valid) begin
      if(!expected_valid[packed_tag]) $fatal(1,"packed result arrived early");
      if(packed_results!==expected[packed_tag])
        $fatal(1,"packed M8 MAC mismatch tag=%0d",packed_tag);
      packed_seen=packed_seen+1;
    end
  end
  initial begin
    for(dot=0;dot<16;dot=dot+1) expected_valid[dot]=0;
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(dot=0;dot<16;dot=dot+1) begin
      for(k=0;k<32;k=k+1) begin
        @(negedge clk);valid_in=1;clear=(k==0);last=(k==31);tag_in=dot;
        for(lane=0;lane<8*32;lane=lane+1)
          activations[lane*8+:8]=$random(seed);
        for(lane=0;lane<6*32;lane=lane+1)
          weights[lane*8+:8]=$random(seed);
      end
    end
    @(negedge clk);valid_in=0;clear=0;last=0;
    wait(packed_seen==16);repeat(3) @(posedge clk);
    if(conventional_seen!=16) $fatal(1,"reference output count");
    $display("tb_int8_mac_tile_packed_m8: PASS results=%0d",packed_seen);
    $finish;
  end
  initial begin repeat(1000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_int8_mac_tile_packed_m8"
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s",
            "tb_int8_mac_tile_packed_m8", "-o", str(build),
            str(RTL / "int8_mac_tile_pipelined.sv"),
            str(RTL / "int8_shared_weight_pair_multiplier.sv"),
            str(RTL / "int8_mac_tile_packed_m8_pipelined.sv"),
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
        timeout=60,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "PASS results=16" in run_result.stdout
