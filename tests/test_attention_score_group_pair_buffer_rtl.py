from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def _packed_hex(values: list[int], width: int) -> str:
    mask = (1 << width) - 1
    packed = sum(
        (value & mask) << (index * width)
        for index, value in enumerate(values)
    )
    return f"{packed:0{(len(values) * width + 3) // 4}x}"


def test_score_group_pair_buffer_streams_lower_and_replays_upper(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")

    pair_tiles = []
    expected_tiles = []
    for tile in range(11):
        lower = [1000 + tile * 24 + index for index in range(24)]
        upper = [-1000 - tile * 24 - index for index in range(24)]
        pair_tiles.append(_packed_hex(lower + upper, 18))
        expected_tiles.append(_packed_hex(lower, 18))
    for tile in range(11):
        upper = [-1000 - tile * 24 - index for index in range(24)]
        expected_tiles.append(_packed_hex(upper, 18))
    pair_hex = tmp_path / "pair_tiles.hex"
    expected_hex = tmp_path / "expected_tiles.hex"
    pair_hex.write_text("\n".join(pair_tiles) + "\n", encoding="utf-8")
    expected_hex.write_text("\n".join(expected_tiles) + "\n", encoding="utf-8")

    testbench = tmp_path / "tb_score_group_pair_buffer.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_score_group_pair_buffer;
  reg clk=0,rst_n=0,pair_tile_valid=0,score_tile_ready=0;
  wire pair_tile_ready,score_tile_valid,busy,done;
  reg [2:0] pair_group=3'd3;
  reg [5:0] pair_key_tile=0;
  reg [2:0] pair_valid_keys=6;
  reg [863:0] pair_scores_q10_packed=0;
  wire [3:0] score_group;
  wire [5:0] score_key_tile;
  wire [2:0] score_valid_keys;
  wire [431:0] scores_q10_packed;
  reg [863:0] pair_mem [0:10];
  reg [431:0] expected_mem [0:21];
  integer input_tile=0,output_tile=0,cycles=0;
  attention_score_group_pair_buffer dut(
    .clk(clk),.rst_n(rst_n),.pair_tile_valid(pair_tile_valid),
    .pair_tile_ready(pair_tile_ready),.pair_group(pair_group),
    .pair_key_tile(pair_key_tile),.pair_valid_keys(pair_valid_keys),
    .pair_scores_q10_packed(pair_scores_q10_packed),
    .score_tile_valid(score_tile_valid),.score_tile_ready(score_tile_ready),
    .score_group(score_group),.score_key_tile(score_key_tile),
    .score_valid_keys(score_valid_keys),.scores_q10_packed(scores_q10_packed),
    .busy(busy),.done(done));
  always #2 clk=~clk;
  always @(negedge clk) begin
    if(rst_n) begin
      cycles=cycles+1;
      score_tile_ready=(cycles%5)!=1;
      if(input_tile<11) begin
        pair_tile_valid=1; pair_key_tile=input_tile;
        pair_valid_keys=(input_tile==10)?4:6;
        pair_scores_q10_packed=pair_mem[input_tile];
      end else pair_tile_valid=0;
    end
  end
  always @(posedge clk) begin
    if(pair_tile_valid && pair_tile_ready) input_tile=input_tile+1;
    if(score_tile_valid && score_tile_ready) begin
      if(score_group!==((output_tile<11)?4'd6:4'd7))
        $fatal(1,"score group mismatch at %0d",output_tile);
      if(score_key_tile!==(output_tile%11))
        $fatal(1,"score key tile mismatch at %0d",output_tile);
      if(score_valid_keys!==(((output_tile%11)==10)?4:6))
        $fatal(1,"valid key count mismatch at %0d",output_tile);
      if(scores_q10_packed!==expected_mem[output_tile])
        $fatal(1,"score payload mismatch at %0d",output_tile);
      output_tile=output_tile+1;
    end
  end
  initial begin
    $readmemh("{pair_hex}",pair_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk); @(negedge clk); rst_n=1;
    wait(done); repeat(2) @(posedge clk);
    if(input_tile!=11 || output_tile!=22) $fatal(1,"missing score tiles");
    $display("tb_score_group_pair_buffer: PASS inputs=%0d outputs=%0d",
             input_tile,output_tile);
    $finish;
  end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_score_group_pair_buffer"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_score_group_pair_buffer",
            "-o",
            str(build),
            str(RTL / "attention_score_group_pair_buffer.sv"),
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
    assert "tb_score_group_pair_buffer: PASS inputs=11 outputs=22" in run_result.stdout
