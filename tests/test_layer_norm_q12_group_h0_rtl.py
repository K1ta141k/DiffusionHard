from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from diffusion_accel.fixed_mlp import _load_tensors
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
    digits = (len(values) * width + 3) // 4
    return f"{packed:0{digits}x}"


def test_layer_norm_q12_group_matches_h0_block0_rtl(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    if not golden_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    name = "folded.block_00.after_attention"
    activation = _load_tensors(golden_path, [name])[name][0, :4]
    _, expected_q12, _ = fixed_layer_norm_q12(activation)
    input_q10 = torch.round(activation.double() * 1024.0).to(torch.int64)
    input_q10 = input_q10.clamp(-(1 << 23), (1 << 23) - 1)

    input_hex = tmp_path / "layer_norm_h0_input_q10.hex"
    input_hex.write_text(
        "\n".join(
            _packed_hex(
                [int(input_q10[token, channel]) for token in range(4)], 24
            )
            for channel in range(768)
        )
        + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "layer_norm_h0_expected_q12.hex"
    expected_hex.write_text(
        "\n".join(
            _packed_hex(
                [int(expected_q12[token, channel]) for token in range(4)], 18
            )
            for channel in range(768)
        )
        + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_layer_norm_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_layer_norm_h0;
  reg clk=0, rst_n=0, start=0, start_replay=0, final_replay=0;
  reg input_valid=0;
  reg [3:0] group_in=4'd2;
  wire start_ready, replay_ready, input_ready;
  reg [95:0] input_q10_packed=0;
  wire output_valid;
  wire [3:0] output_group;
  wire [9:0] output_channel;
  wire [71:0] output_q12_packed;
  wire busy, done;
  reg [95:0] input_mem [0:767];
  reg [71:0] expected_mem [0:767];
  integer channel, output_count=0;
  layer_norm_q12_group dut(
    .clk(clk),.rst_n(rst_n),.start(start),.group_in(group_in),
    .start_ready(start_ready),.start_replay(start_replay),
    .final_replay(final_replay),.replay_ready(replay_ready),
    .input_valid(input_valid),.input_ready(input_ready),
    .input_q10_packed(input_q10_packed),.output_valid(output_valid),
    .output_group(output_group),.output_channel(output_channel),
    .output_q12_packed(output_q12_packed),.busy(busy),.done(done));
  always #2 clk=~clk;
  task stream_input;
    begin
      for(channel=0;channel<768;channel=channel+1) begin
        @(negedge clk);
        if(!input_ready) $fatal(1,"input not ready");
        input_q10_packed=input_mem[channel];
        input_valid=1;
      end
      @(negedge clk); input_valid=0;
    end
  endtask
  always @(posedge clk) begin
    #1;
    if(output_valid) begin
      if(output_group!==4'd2) $fatal(1,"group mismatch");
      if(output_channel!==output_count) $fatal(1,"channel mismatch");
      if(output_q12_packed!==expected_mem[output_count])
        $fatal(1,
          "H0 mismatch channel %0d got=%h expected=%h mean=%0d,%0d,%0d,%0d variance=%0d,%0d,%0d,%0d root=%0d,%0d,%0d,%0d inv=%0d,%0d,%0d,%0d",
          output_count,output_q12_packed,expected_mem[output_count],
          dut.means_q10[0],dut.means_q10[1],dut.means_q10[2],dut.means_q10[3],
          dut.variances_q20[0],dut.variances_q20[1],dut.variances_q20[2],dut.variances_q20[3],
          dut.sqrt_roots_q16[0],dut.sqrt_roots_q16[1],dut.sqrt_roots_q16[2],dut.sqrt_roots_q16[3],
          dut.inverse_q18[0],dut.inverse_q18[1],dut.inverse_q18[2],dut.inverse_q18[3]);
      output_count=output_count+1;
    end
  end
  initial begin
    $readmemh("{input_hex}",input_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk); @(negedge clk); rst_n=1; start=1;
    @(negedge clk); start=0; stream_input(); wait(replay_ready);
    @(negedge clk); final_replay=1; start_replay=1;
    @(negedge clk); start_replay=0; stream_input(); wait(done);
    repeat(2) @(posedge clk);
    if(output_count!=768) $fatal(1,"missing H0 outputs");
    $display("tb_layer_norm_h0: PASS"); $finish;
  end
  initial begin repeat(10000) @(posedge clk); $fatal(1,"timeout"); end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_layer_norm_h0"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_layer_norm_h0",
            "-o",
            str(build),
            str(RTL / "unsigned_divider_iterative.sv"),
            str(RTL / "unsigned_sqrt_iterative.sv"),
            str(RTL / "layer_norm_q12_group.sv"),
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
    assert "tb_layer_norm_h0: PASS" in run_result.stdout
