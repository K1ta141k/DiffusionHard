from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from diffusion_accel.fixed_mlp import _load_tensors, quantize_up_activation_fixed
from diffusion_accel.fixed_norm import fixed_layer_norm_q12


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"


def _packed_hex(values: list[int], width: int) -> str:
    mask = (1 << width) - 1
    packed = sum(
        (value & mask) << (index * width)
        for index, value in enumerate(values)
    )
    return f"{packed:0{(len(values) * width + 3) // 4}x}"


def test_layer_norm_mlp_up_frontend_matches_h0_rtl(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    if not golden_path.is_file() or not weights_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    input_name = "folded.block_00.after_attention"
    norm_name = "folded.block_00.norm2_unaffine"
    weight_name = "block_00.mlp_up.weight"
    goldens = _load_tensors(golden_path, [input_name, norm_name])
    residual = goldens[input_name][0, :4]
    calibration_norm = goldens[norm_name][0]
    weight = _load_tensors(weights_path, [weight_name])[weight_name]
    smoothing = (
        calibration_norm.abs().amax(dim=0).clamp_min(1e-8).pow(0.75)
        / weight.abs().amax(dim=0).clamp_min(1e-8).pow(0.25)
    ).clamp_min(1e-8)
    fixed_norm, _, _ = fixed_layer_norm_q12(residual)
    activation_q, token_factors, tensors, _ = quantize_up_activation_fixed(
        fixed_norm, smoothing
    )
    residual_q10 = torch.round(residual.double() * 1024.0).to(torch.int64)
    residual_q10 = residual_q10.clamp(-(1 << 23), (1 << 23) - 1)

    residual_hex = tmp_path / "frontend_residual_q10.hex"
    residual_hex.write_text(
        "\n".join(
            _packed_hex(
                [int(residual_q10[token, channel]) for token in range(4)], 24
            )
            for channel in range(768)
        )
        + "\n",
        encoding="utf-8",
    )
    reciprocal_hex = tmp_path / "frontend_reciprocal_q15.hex"
    reciprocal_hex.write_text(
        "\n".join(f"{int(value):05x}" for value in tensors["reciprocal"])
        + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "frontend_expected_tiles.hex"
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

    testbench = tmp_path / "tb_layer_norm_mlp_up_frontend_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_layer_norm_mlp_up_frontend_h0;
  reg clk=0, rst_n=0, start=0, residual_input_valid=0;
  reg [3:0] group_in=4'd3;
  wire start_ready, residual_input_ready;
  reg [95:0] residual_q10_packed=0;
  reg [17:0] smoothing_reciprocal_q15=0;
  wire token_factor_valid;
  wire [3:0] token_factor_group;
  wire [63:0] token_factors_packed;
  wire activation_load_valid;
  wire [3:0] activation_load_group;
  wire [4:0] activation_load_k_tile;
  wire [1023:0] activation_load_data;
  wire busy, done;
  reg [95:0] residual_mem [0:767];
  reg [17:0] reciprocal_mem [0:767];
  reg [1023:0] expected_tiles [0:23];
  integer channel, pass, tile_count=0, factor_count=0;
  layer_norm_mlp_up_activation_frontend dut(
    .clk(clk),.rst_n(rst_n),.start(start),.group_in(group_in),
    .start_ready(start_ready),.residual_input_valid(residual_input_valid),
    .residual_input_ready(residual_input_ready),
    .residual_q10_packed(residual_q10_packed),
    .smoothing_reciprocal_q15(smoothing_reciprocal_q15),
    .token_factor_valid(token_factor_valid),
    .token_factor_group(token_factor_group),
    .token_factors_packed(token_factors_packed),
    .activation_load_valid(activation_load_valid),
    .activation_load_group(activation_load_group),
    .activation_load_k_tile(activation_load_k_tile),
    .activation_load_data(activation_load_data),.busy(busy),.done(done));
  always #2 clk=~clk;
  task stream_one_pass;
    begin
      wait(residual_input_ready);
      for(channel=0;channel<768;channel=channel+1) begin
        @(negedge clk);
        if(!residual_input_ready) $fatal(1,"input not ready in pass %0d",pass);
        residual_q10_packed=residual_mem[channel];
        smoothing_reciprocal_q15=reciprocal_mem[channel];
        residual_input_valid=1;
      end
      @(negedge clk); residual_input_valid=0;
    end
  endtask
  always @(posedge clk) begin
    #1;
    if(token_factor_valid) begin
      if(token_factor_group!==4'd3) $fatal(1,"factor group mismatch");
      if(token_factors_packed!==64'h{expected_factors})
        $fatal(1,"frontend token factor mismatch");
      factor_count=factor_count+1;
    end
    if(activation_load_valid) begin
      if(activation_load_group!==4'd3) $fatal(1,"tile group mismatch");
      if(activation_load_k_tile!==tile_count) $fatal(1,"tile tag mismatch");
      if(activation_load_data!==expected_tiles[tile_count])
        $fatal(1,"frontend activation mismatch at tile %0d",tile_count);
      tile_count=tile_count+1;
    end
  end
  initial begin
    $readmemh("{residual_hex}",residual_mem);
    $readmemh("{reciprocal_hex}",reciprocal_mem);
    $readmemh("{expected_hex}",expected_tiles);
    repeat(3) @(posedge clk); @(negedge clk); rst_n=1; start=1;
    @(negedge clk); start=0;
    for(pass=0;pass<3;pass=pass+1) stream_one_pass();
    wait(done); repeat(2) @(posedge clk);
    if(factor_count!=1) $fatal(1,"expected one factor vector");
    if(tile_count!=24) $fatal(1,"expected 24 activation tiles");
    if(busy) $fatal(1,"frontend remained busy");
    $display("tb_layer_norm_mlp_up_frontend_h0: PASS"); $finish;
  end
  initial begin repeat(15000) @(posedge clk); $fatal(1,"timeout"); end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_layer_norm_mlp_up_frontend_h0"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_layer_norm_mlp_up_frontend_h0",
            "-o",
            str(build),
            str(RTL / "unsigned_divider_iterative.sv"),
            str(RTL / "unsigned_sqrt_iterative.sv"),
            str(RTL / "layer_norm_q12_group.sv"),
            str(RTL / "mlp_up_activation_quantizer.sv"),
            str(RTL / "layer_norm_mlp_up_activation_frontend.sv"),
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
    assert "tb_layer_norm_mlp_up_frontend_h0: PASS" in run_result.stdout
