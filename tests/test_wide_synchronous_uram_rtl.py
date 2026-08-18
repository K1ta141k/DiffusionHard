from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_wide_synchronous_uram_round_trip(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    testbench = tmp_path / "tb_wide_synchronous_uram.sv"
    testbench.write_text(
        """`timescale 1ns/1ps
module tb_wide_synchronous_uram;
  reg clk=0,write_valid=0,read_valid=0;
  reg [3:0] write_address=0,read_address=0;
  reg [199:0] write_data=0;
  wire read_data_valid;
  wire [199:0] read_data;
  wide_synchronous_uram #(.WIDTH(200),.DEPTH(16)) dut(
    .clk(clk),.write_valid(write_valid),.write_address(write_address),
    .write_data(write_data),.read_valid(read_valid),
    .read_address(read_address),.read_data_valid(read_data_valid),
    .read_data(read_data));
  always #2 clk=~clk;
  initial begin
    @(negedge clk);write_valid=1;write_address=3;
    write_data=200'h123456789abcdef0123456789abcdef0123456789abcdef012;
    @(negedge clk);write_address=11;
    write_data=200'hfedcba9876543210fedcba9876543210fedcba9876543210f;
    @(negedge clk);write_valid=0;read_valid=1;read_address=3;
    @(posedge clk);#1;
    if(!read_data_valid || read_data!==
       200'h123456789abcdef0123456789abcdef0123456789abcdef012)
      $fatal(1,"first wide read mismatch");
    @(negedge clk);read_address=11;
    @(posedge clk);#1;
    if(!read_data_valid || read_data!==
       200'hfedcba9876543210fedcba9876543210fedcba9876543210f)
      $fatal(1,"second wide read mismatch");
    $display("tb_wide_synchronous_uram: PASS");$finish;
  end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_wide_synchronous_uram"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_wide_synchronous_uram",
            "-o",
            str(build),
            str(RTL / "wide_synchronous_uram.sv"),
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
    assert "tb_wide_synchronous_uram: PASS" in run_result.stdout
