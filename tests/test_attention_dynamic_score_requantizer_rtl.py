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


def _symmetric_round_shift(value: int, shift: int) -> int:
    magnitude = (abs(value) + (1 << (shift - 1))) >> shift
    return -magnitude if value < 0 else magnitude


def test_dynamic_score_requantizer_is_bit_exact(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")

    generator = random.Random(0xA77E170)
    vectors = [
        (1, 1, 0),
        (131072, 131072, 1_032_256),
        (131072, 131072, -1_032_256),
        (131071, 131071, 1_048_575),
        (131071, 131071, -1_048_576),
    ]
    vectors.extend(
        (
            generator.randint(1, 131072),
            generator.randint(1, 131072),
            generator.randint(-1_032_256, 1_032_256),
        )
        for _ in range(20003)
    )
    maxima_hex = tmp_path / "maxima.hex"
    dots_hex = tmp_path / "dots.hex"
    expected_hex = tmp_path / "expected_scores.hex"
    maxima_hex.write_text(
        "\n".join(
            f"{((query << 18) | key):09x}" for query, key, _ in vectors
        )
        + "\n",
        encoding="utf-8",
    )
    dots_hex.write_text(
        "\n".join(f"{dot & ((1 << 21) - 1):06x}" for _, _, dot in vectors)
        + "\n",
        encoding="utf-8",
    )
    expected = []
    for query, key, dot in vectors:
        multiplier = _round_ties_to_even(
            query * key * (1 << 28), DENOMINATOR
        )
        score = _symmetric_round_shift(dot * multiplier, 28)
        expected.append(max(-131072, min(131071, score)))
    expected_hex.write_text(
        "\n".join(f"{score & 0x3FFFF:05x}" for score in expected) + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_dynamic_score_requantizer.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_dynamic_score_requantizer;
  localparam VECTORS={len(vectors)};
  reg clk=0,rst_n=0,valid_in=0;
  reg [15:0] tag_in=0;
  reg signed [20:0] dot_product_int8=0;
  reg [17:0] query_maximum=0,key_maximum=0;
  wire valid_out;
  wire [15:0] tag_out;
  wire signed [17:0] score_q10;
  reg [35:0] maxima_mem [0:VECTORS-1];
  reg [20:0] dot_mem [0:VECTORS-1];
  reg [17:0] expected_mem [0:VECTORS-1];
  integer index,seen=0;
  attention_dynamic_score_requantizer_q10 #(.TAG_WIDTH(16)) dut(
    .clk(clk),.rst_n(rst_n),.valid_in(valid_in),.tag_in(tag_in),
    .dot_product_int8(dot_product_int8),.query_maximum(query_maximum),
    .key_maximum(key_maximum),.valid_out(valid_out),.tag_out(tag_out),
    .score_q10(score_q10));
  always #2 clk=~clk;
  always @(posedge clk) begin
    #1;
    if(valid_out) begin
      if(tag_out!==seen) $fatal(1,"tag mismatch at %0d",seen);
      if(score_q10!==expected_mem[seen])
        $fatal(1,"score mismatch at %0d got=%0d expected=%0d",
               seen,score_q10,$signed(expected_mem[seen]));
      seen=seen+1;
    end
  end
  initial begin
    $readmemh("{maxima_hex}",maxima_mem);
    $readmemh("{dots_hex}",dot_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk); @(negedge clk); rst_n=1;
    for(index=0;index<VECTORS;index=index+1) begin
      @(negedge clk); valid_in=1; tag_in=index;
      query_maximum=maxima_mem[index][35:18];
      key_maximum=maxima_mem[index][17:0]; dot_product_int8=dot_mem[index];
    end
    @(negedge clk); valid_in=0;
    wait(seen==VECTORS); repeat(2) @(posedge clk);
    $display("tb_dynamic_score_requantizer: PASS vectors=%0d",seen);
    $finish;
  end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_dynamic_score_requantizer"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_dynamic_score_requantizer",
            "-o",
            str(build),
            str(RTL / "attention_dynamic_score_multiplier_q28.sv"),
            str(RTL / "attention_dynamic_score_requantizer_q10.sv"),
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
    assert "tb_dynamic_score_requantizer: PASS vectors=20008" in run_result.stdout
