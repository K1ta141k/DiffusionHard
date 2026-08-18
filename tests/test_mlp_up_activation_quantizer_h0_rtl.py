from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from diffusion_accel.fixed_mlp import (
    _load_tensors,
    quantize_up_activation_fixed,
)


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"


def _packed_hex(values: list[int], width: int) -> str:
    mask = (1 << width) - 1
    packed = sum((value & mask) << (index * width) for index, value in enumerate(values))
    digits = (len(values) * width + 3) // 4
    return f"{packed:0{digits}x}"


def test_mlp_up_activation_quantizer_matches_h0_group_rtl(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    if not (PACKAGE / "golden_tensors.safetensors").is_file():
        pytest.skip("H0 hardware package is unavailable")

    golden_name = "folded.block_00.norm2_unaffine"
    weight_name = "block_00.mlp_up.weight"
    normalized = _load_tensors(
        PACKAGE / "golden_tensors.safetensors", [golden_name]
    )[golden_name][0]
    weight = _load_tensors(
        PACKAGE / "folded_fp16_weights.safetensors", [weight_name]
    )[weight_name]
    alpha = 0.75
    smoothing = (
        normalized.abs().amax(dim=0).clamp_min(1e-8).pow(alpha)
        / weight.abs().amax(dim=0).clamp_min(1e-8).pow(1.0 - alpha)
    ).clamp_min(1e-8)
    activation_q, token_factors, tensors, _ = quantize_up_activation_fixed(
        normalized, smoothing
    )

    normalized_q12 = torch.round(normalized[:4].double() * 4096.0).to(
        torch.int64
    )
    normalized_q12 = normalized_q12.clamp(-(1 << 17), (1 << 17) - 1)
    normalized_hex = tmp_path / "normalized_q12.hex"
    normalized_hex.write_text(
        "\n".join(
            _packed_hex(
                [int(normalized_q12[token, channel]) for token in range(4)],
                18,
            )
            for channel in range(768)
        )
        + "\n",
        encoding="utf-8",
    )
    reciprocal_hex = tmp_path / "reciprocal_q15.hex"
    reciprocal_hex.write_text(
        "\n".join(f"{int(value):05x}" for value in tensors["reciprocal"])
        + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "expected_tiles.hex"
    expected_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(activation_q[token, tile * 32 + channel])
                    for token in range(4)
                    for channel in range(32)
                ],
                8,
            )
            for tile in range(24)
        )
        + "\n",
        encoding="utf-8",
    )
    expected_factors = _packed_hex(
        [int(token_factors[token]) for token in range(4)], 16
    )
    testbench = tmp_path / "tb_h0_activation_quantizer.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_h0_activation_quantizer;
  reg clk=0, rst_n=0, start=0, start_pass2=0, input_valid=0;
  reg [3:0] group_in=0;
  wire start_ready, pass2_ready, input_ready;
  reg [71:0] normalized_q12_packed=0;
  reg [17:0] smoothing_reciprocal_q15=0;
  wire token_factor_valid;
  wire [3:0] token_factor_group;
  wire [63:0] token_factors_packed;
  wire activation_load_valid;
  wire [3:0] activation_load_group;
  wire [4:0] activation_load_k_tile;
  wire [1023:0] activation_load_data;
  wire busy, done;
  reg [71:0] normalized_mem [0:767];
  reg [17:0] reciprocal_mem [0:767];
  reg [1023:0] expected_tiles [0:23];
  integer channel, tile_count=0;
  mlp_up_activation_quantizer dut(
    .clk(clk),.rst_n(rst_n),.start(start),.group_in(group_in),
    .start_ready(start_ready),.start_pass2(start_pass2),
    .pass2_ready(pass2_ready),.input_valid(input_valid),
    .input_ready(input_ready),.normalized_q12_packed(normalized_q12_packed),
    .smoothing_reciprocal_q15(smoothing_reciprocal_q15),
    .token_factor_valid(token_factor_valid),.token_factor_group(token_factor_group),
    .token_factors_packed(token_factors_packed),
    .activation_load_valid(activation_load_valid),
    .activation_load_group(activation_load_group),
    .activation_load_k_tile(activation_load_k_tile),
    .activation_load_data(activation_load_data),.busy(busy),.done(done));
  always #2 clk=~clk;
  task stream_pass;
    begin
      for(channel=0;channel<768;channel=channel+1) begin
        @(negedge clk);
        if(!input_ready) $fatal(1,"input not ready");
        normalized_q12_packed=normalized_mem[channel];
        smoothing_reciprocal_q15=reciprocal_mem[channel];
        input_valid=1;
      end
      @(negedge clk); input_valid=0;
    end
  endtask
  always @(posedge clk) begin
    #1;
    if(token_factor_valid && token_factors_packed !== 64'h{expected_factors})
      $fatal(1,"H0 token factor mismatch");
    if(activation_load_valid) begin
      if(activation_load_k_tile !== tile_count) $fatal(1,"H0 tile tag mismatch");
      if(activation_load_data !== expected_tiles[tile_count])
        $fatal(1,"H0 activation tile mismatch at %0d",tile_count);
      tile_count=tile_count+1;
    end
  end
  initial begin
    $readmemh("{normalized_hex}",normalized_mem);
    $readmemh("{reciprocal_hex}",reciprocal_mem);
    $readmemh("{expected_hex}",expected_tiles);
    repeat(3) @(posedge clk); @(negedge clk); rst_n=1; start=1;
    @(negedge clk); start=0; stream_pass(); wait(pass2_ready);
    @(negedge clk); start_pass2=1; @(negedge clk); start_pass2=0;
    stream_pass(); wait(done); repeat(2) @(posedge clk);
    if(tile_count!=24) $fatal(1,"missing H0 tiles");
    $display("tb_h0_activation_quantizer: PASS"); $finish;
  end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_h0_activation_quantizer"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_h0_activation_quantizer",
            "-o",
            str(build),
            str(RTL / "unsigned_divider_iterative.sv"),
            str(RTL / "mlp_up_activation_quantizer.sv"),
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
    assert "tb_h0_activation_quantizer: PASS" in run_result.stdout
