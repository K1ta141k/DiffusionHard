from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_dual_mode_array_matches_attention_and_mlp_references(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    testbench = tmp_path / "tb_mixed_precision_packed_m8_mac_tile.sv"
    testbench.write_text(
        """`timescale 1ns/1ps
module tb_mixed_precision_packed_m8_mac_tile;
  reg clk=0,rst_n=0,valid_in=0,mode=0,clear=0,last=0;
  reg [7:0] tag_in=0;reg [4*32*18-1:0] aa=0;
  reg [6*32*18-1:0] aw=0;reg [8*32*8-1:0] ma=0;
  reg [6*32*8-1:0] mw=0;
  wire unified_valid,unified_mode;wire [7:0] unified_tag;
  wire [4*6*48-1:0] unified_attention;
  wire [8*6*32-1:0] unified_mlp;
  wire attention_valid;wire [7:0] attention_tag;
  wire [4*6*48-1:0] attention_results;
  wire mlp_valid;wire [7:0] mlp_tag;wire [8*6*32-1:0] mlp_results;
  reg [4*6*48-1:0] expected_attention[0:15];
  reg [8*6*32-1:0] expected_mlp[0:15];reg expected_valid[0:15];
  integer dot,k,lane,seed=32'h55aa1234,unified_seen=0;
  mixed_precision_packed_m8_mac_tile_pipelined #(.TAG_WIDTH(8)) dut(
    .clk(clk),.rst_n(rst_n),.valid_in(valid_in),.narrow_int8_mode(mode),
    .clear_accumulators(clear),.last_k_tile(last),.tag_in(tag_in),
    .attention_activations_packed(aa),.attention_weights_packed(aw),
    .mlp_activations_packed(ma),.mlp_weights_packed(mw),
    .valid_out(unified_valid),.narrow_int8_mode_out(unified_mode),
    .tag_out(unified_tag),.attention_accumulators_packed(unified_attention),
    .mlp_accumulators_packed(unified_mlp));
  mixed_precision_mac_tile_pipelined #(
    .M_LANES(4),.N_LANES(6),.TAG_WIDTH(8)
  ) attention_reference(.clk(clk),.rst_n(rst_n),
    .valid_in(valid_in&&!mode),.narrow_int8_mode(1'b0),
    .clear_accumulators(clear),.last_k_tile(last),.tag_in(tag_in),
    .activations_packed(aa),.weights_packed(aw),.valid_out(attention_valid),
    .tag_out(attention_tag),.accumulators_packed(attention_results));
  int8_mac_tile_packed_m8_pipelined #(.TAG_WIDTH(8)) mlp_reference(
    .clk(clk),.rst_n(rst_n),.valid_in(valid_in&&mode),
    .clear_accumulators(clear),.last_k_tile(last),.tag_in(tag_in),
    .activations_packed(ma),.weights_packed(mw),.valid_out(mlp_valid),
    .tag_out(mlp_tag),.accumulators_packed(mlp_results));
  always #2 clk=~clk;
  always @(posedge clk) begin #1;
    if(attention_valid) begin expected_attention[attention_tag]=attention_results;
      expected_valid[attention_tag]=1;end
    if(mlp_valid) begin expected_mlp[mlp_tag]=mlp_results;
      expected_valid[mlp_tag]=1;end
    if(unified_valid) begin
      if(!expected_valid[unified_tag]) $fatal(1,"unified result arrived early");
      if(unified_mode && unified_mlp!==expected_mlp[unified_tag])
        $fatal(1,"unified MLP mismatch tag=%0d",unified_tag);
      if(!unified_mode && unified_attention!==expected_attention[unified_tag])
        $fatal(1,"unified attention mismatch tag=%0d",unified_tag);
      unified_seen=unified_seen+1;
    end
  end
  task drive_dot(input integer dot_tag,input integer dot_mode);
    begin
      for(k=0;k<32;k=k+1) begin @(negedge clk);valid_in=1;mode=dot_mode;
        clear=(k==0);last=(k==31);tag_in=dot_tag;
        for(lane=0;lane<4*32;lane=lane+1) aa[lane*18+:18]=$random(seed);
        for(lane=0;lane<6*32;lane=lane+1) aw[lane*18+:18]=$random(seed);
        for(lane=0;lane<8*32;lane=lane+1) ma[lane*8+:8]=$random(seed);
        for(lane=0;lane<6*32;lane=lane+1) mw[lane*8+:8]=$random(seed);
      end
    end
  endtask
  initial begin
    for(dot=0;dot<16;dot=dot+1) expected_valid[dot]=0;
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(dot=0;dot<8;dot=dot+1) drive_dot(dot,0);
    @(negedge clk);valid_in=0;clear=0;last=0;wait(unified_seen==8);
    for(dot=8;dot<16;dot=dot+1) drive_dot(dot,1);
    @(negedge clk);valid_in=0;clear=0;last=0;wait(unified_seen==16);
    repeat(3) @(posedge clk);
    $display("tb_mixed_precision_packed_m8_mac_tile: PASS results=%0d",unified_seen);
    $finish;
  end
  initial begin repeat(1000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_mixed_precision_packed_m8_mac_tile"
    sources = [
        "int8_mac_tile_pipelined.sv",
        "mixed_precision_mac_tile_pipelined.sv",
        "int8_shared_weight_pair_multiplier.sv",
        "int8_mac_tile_packed_m8_pipelined.sv",
        "mixed_precision_token_pair_multiplier.sv",
        "mixed_precision_dsp48e2_cascade_cell.sv",
        "mixed_precision_packed_m8_mac_tile_pipelined.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s",
            "tb_mixed_precision_packed_m8_mac_tile", "-o", str(build),
            *(str(RTL / source) for source in sources), str(testbench),
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
        timeout=240,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "PASS results=16" in run_result.stdout
