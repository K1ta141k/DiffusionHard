from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_paired_uram_qk_scratchpad_preserves_all_values(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    testbench = tmp_path / "tb_qk_unrotated_scratchpad_paired_uram.sv"
    testbench.write_text(
        """`timescale 1ns/1ps
module tb_qk_unrotated_scratchpad_paired_uram;
  reg clk=0,query_load_valid=0,key_load_valid=0,read_valid=0;
  reg [5:0] load_token=0,load_channel=0,read_token=0;
  reg [4:0] read_pair=0;
  reg signed [17:0] load_query=0,load_key=0;
  wire read_data_valid;wire [5:0] read_token_out;wire [4:0] read_pair_out;
  wire signed [17:0] query_first,query_second,key_first,key_second;
  integer token,channel,pair,reads=0;
  integer first_index,second_index;
  qk_unrotated_scratchpad_paired_uram dut(
    .clk(clk),.query_load_valid(query_load_valid),
    .key_load_valid(key_load_valid),.load_token(load_token),
    .load_channel(load_channel),.load_query_q12(load_query),
    .load_key_q12(load_key),.read_valid(read_valid),
    .read_token(read_token),.read_pair(read_pair),
    .read_data_valid(read_data_valid),.read_token_out(read_token_out),
    .read_pair_out(read_pair_out),.query_first_q12(query_first),
    .query_second_q12(query_second),.key_first_q12(key_first),
    .key_second_q12(key_second));
  always #2 clk=~clk;
  always @(posedge clk) begin
    #1;
    if(read_data_valid) begin
      first_index=read_token_out*64+read_pair_out;
      second_index=first_index+32;
      if(query_first!==first_index || query_second!==second_index)
        $fatal(1,"query mismatch token=%0d pair=%0d got=%0d,%0d",
          read_token_out,read_pair_out,query_first,query_second);
      if(key_first!==-first_index || key_second!==-second_index)
        $fatal(1,"key mismatch token=%0d pair=%0d got=%0d,%0d",
          read_token_out,read_pair_out,key_first,key_second);
      reads=reads+1;
    end
  end
  initial begin
    repeat(3) @(posedge clk);
    for(token=0;token<64;token=token+1) begin
      for(channel=0;channel<64;channel=channel+1) begin
        @(negedge clk);query_load_valid=1;key_load_valid=1;
        load_token=token;load_channel=channel;
        load_query=token*64+channel;load_key=-(token*64+channel);
      end
    end
    @(negedge clk);query_load_valid=0;key_load_valid=0;
    for(token=0;token<64;token=token+1) begin
      for(pair=0;pair<32;pair=pair+1) begin
        @(negedge clk);read_valid=1;read_token=token;read_pair=pair;
      end
    end
    @(negedge clk);read_valid=0;repeat(3) @(posedge clk);
    if(reads!=2048)$fatal(1,"read count mismatch %0d",reads);
    $display("tb_qk_unrotated_scratchpad_paired_uram: PASS reads=%0d",reads);
    $finish;
  end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_qk_unrotated_scratchpad_paired_uram"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "tb_qk_unrotated_scratchpad_paired_uram",
            "-o",
            str(build),
            str(RTL / "qk_unrotated_scratchpad_paired_uram.sv"),
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
    assert "tb_qk_unrotated_scratchpad_paired_uram: PASS" in run_result.stdout
