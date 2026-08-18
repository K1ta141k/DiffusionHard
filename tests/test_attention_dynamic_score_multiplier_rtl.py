from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
DENOMINATOR = 127 * 127 * (1 << 17)


def _round_ties_to_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled = 2 * remainder
    if doubled > denominator or (doubled == denominator and quotient & 1):
        quotient += 1
    return quotient


def test_dynamic_score_multiplier_is_exact(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")

    generator = random.Random(0xD1FF0510)
    maxima = [
        (1, 1),
        (1, 131072),
        (131072, 1),
        (131072, 131072),
        (127, 127),
        (128, 128),
        (131071, 131071),
    ]
    maxima.extend(
        (generator.randint(1, 131072), generator.randint(1, 131072))
        for _ in range(20001)
    )
    query_hex = tmp_path / "query_maxima.hex"
    key_hex = tmp_path / "key_maxima.hex"
    expected_hex = tmp_path / "expected_q28.hex"
    query_hex.write_text(
        "\n".join(f"{query:05x}" for query, _ in maxima) + "\n",
        encoding="utf-8",
    )
    key_hex.write_text(
        "\n".join(f"{key:05x}" for _, key in maxima) + "\n",
        encoding="utf-8",
    )
    expected_hex.write_text(
        "\n".join(
            f"{_round_ties_to_even(query * key * (1 << 28), DENOMINATOR):08x}"
            for query, key in maxima
        )
        + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_dynamic_score_multiplier.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_dynamic_score_multiplier;
  localparam VECTORS={len(maxima)};
  reg clk=0,rst_n=0,valid_in=0;
  reg [15:0] tag_in=0;
  reg [17:0] query_maximum=0,key_maximum=0;
  wire valid_out;
  wire [15:0] tag_out;
  wire [31:0] multiplier_q28;
  reg [17:0] query_mem [0:VECTORS-1];
  reg [17:0] key_mem [0:VECTORS-1];
  reg [31:0] expected_mem [0:VECTORS-1];
  integer index,seen=0;
  attention_dynamic_score_multiplier_q28 #(.TAG_WIDTH(16)) dut(
    .clk(clk),.rst_n(rst_n),.valid_in(valid_in),.tag_in(tag_in),
    .query_maximum(query_maximum),.key_maximum(key_maximum),
    .valid_out(valid_out),.tag_out(tag_out),.multiplier_q28(multiplier_q28));
  always #2 clk=~clk;
  always @(posedge clk) begin
    #1;
    if(valid_out) begin
      if(tag_out!==seen) $fatal(1,"tag mismatch at %0d",seen);
      if(multiplier_q28!==expected_mem[seen])
        $fatal(1,"multiplier mismatch at %0d got=%0d expected=%0d",
               seen,multiplier_q28,expected_mem[seen]);
      seen=seen+1;
    end
  end
  initial begin
    $readmemh("{query_hex}",query_mem);
    $readmemh("{key_hex}",key_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk); @(negedge clk); rst_n=1;
    for(index=0;index<VECTORS;index=index+1) begin
      @(negedge clk); valid_in=1; tag_in=index;
      query_maximum=query_mem[index]; key_maximum=key_mem[index];
    end
    @(negedge clk); valid_in=0;
    wait(seen==VECTORS); repeat(2) @(posedge clk);
    $display("tb_dynamic_score_multiplier: PASS vectors=%0d",seen);
    $finish;
  end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_dynamic_score_multiplier"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_dynamic_score_multiplier",
            "-o",
            str(build),
            str(RTL / "attention_dynamic_score_multiplier_q28.sv"),
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
    assert "tb_dynamic_score_multiplier: PASS vectors=20008" in run_result.stdout
