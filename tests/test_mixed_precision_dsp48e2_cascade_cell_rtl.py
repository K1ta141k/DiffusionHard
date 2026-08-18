from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_behavioral_dsp_cascade_cell_matches_raw_and_accumulated_products(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    testbench = tmp_path / "tb_mixed_precision_dsp48e2_cascade_cell.sv"
    testbench.write_text(
        """`timescale 1ns/1ps
module tb_mixed_precision_dsp48e2_cascade_cell;
  reg clk=0,rst_n=0,cascade_enable=0;
  reg signed [26:0] activation=0;
  reg signed [17:0] weight=0;
  reg signed [47:0] cascade_in=0;
  wire signed [47:0] result,cascade_out;
  reg signed [44:0] expected_product=0;
  reg signed [47:0] expected_result=0;
  integer index,seed=32'h47e2a55a;
  mixed_precision_dsp48e2_cascade_cell dut(
    .clk(clk),.rst_n(rst_n),.cascade_enable(cascade_enable),
    .selected_activation(activation),.selected_weight(weight),
    .cascade_in(cascade_in),.result(result),.cascade_out(cascade_out));
  always #2 clk=~clk;
  initial begin
    repeat(3) @(posedge clk); @(negedge clk); rst_n=1;
    for(index=0;index<20000;index=index+1) begin
      activation=$random(seed); weight=$random(seed);
      cascade_in={$random(seed),$random(seed)};
      cascade_enable=index[0];
      expected_product=$signed(activation)*$signed(weight);
      expected_result=cascade_enable
        ? $signed(expected_product)+$signed(cascade_in)
        : $signed(expected_product);
      @(posedge clk); #1;
      if($signed(result)!==$signed(expected_result))
        $fatal(1,"result mismatch index=%0d",index);
      if(cascade_out!==result) $fatal(1,"cascade output mismatch");
      @(negedge clk);
    end
    $display("tb_mixed_precision_dsp48e2_cascade_cell: PASS vectors=%0d",index);
    $finish;
  end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_mixed_precision_dsp48e2_cascade_cell"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "tb_mixed_precision_dsp48e2_cascade_cell",
            "-o",
            str(build),
            str(RTL / "mixed_precision_dsp48e2_cascade_cell.sv"),
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
