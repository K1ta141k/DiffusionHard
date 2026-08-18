from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.fixed_attention import fixed_rotary_q12
from diffusion_accel.fixed_mlp import _load_tensors


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


def test_rotary_qk_pair_matches_h0_block0_rtl(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    if not golden_path.is_file() or not weights_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    name = "folded.block_00.qkv"
    qkv = _load_tensors(golden_path, [name])[name][0, :4]
    tables = _load_tensors(weights_path, ["rotary.cos", "rotary.sin"])
    _, _, query_q, key_q, details = fixed_rotary_q12(
        qkv,
        tables["rotary.cos"][:4].float(),
        tables["rotary.sin"][:4].float(),
    )
    source_q = details["tensors"]["qkv_q12"]
    cosine_q = details["tensors"]["cosine_q15"]
    sine_q = details["tensors"]["sine_q15"]

    input_vectors = []
    constants = []
    expected_vectors = []
    for token in range(4):
        for head in range(12):
            for pair in range(32):
                input_vectors.append(
                    _packed_hex(
                        [
                            int(source_q[token, 0, head, pair]),
                            int(source_q[token, 0, head, pair + 32]),
                            int(source_q[token, 1, head, pair]),
                            int(source_q[token, 1, head, pair + 32]),
                        ],
                        18,
                    )
                )
                constants.append(
                    _packed_hex(
                        [int(cosine_q[token, pair]), int(sine_q[token, pair])],
                        16,
                    )
                )
                expected_vectors.append(
                    _packed_hex(
                        [
                            int(query_q[token, head, pair]),
                            int(query_q[token, head, pair + 32]),
                            int(key_q[token, head, pair]),
                            int(key_q[token, head, pair + 32]),
                        ],
                        18,
                    )
                )

    input_hex = tmp_path / "rotary_input.hex"
    input_hex.write_text("\n".join(input_vectors) + "\n", encoding="utf-8")
    constant_hex = tmp_path / "rotary_constants.hex"
    constant_hex.write_text("\n".join(constants) + "\n", encoding="utf-8")
    expected_hex = tmp_path / "rotary_expected.hex"
    expected_hex.write_text(
        "\n".join(expected_vectors) + "\n", encoding="utf-8"
    )

    testbench = tmp_path / "tb_rotary_qk_pair_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_rotary_qk_pair_h0;
  localparam integer COUNT=1536;
  reg clk=0, rst_n=0, valid_in=0;
  reg [3:0] group_in=4'd7, head_in=0;
  reg [1:0] token_in=0;
  reg [4:0] pair_in=0;
  reg signed [17:0] qf=0,qs=0,kf=0,ks=0;
  reg signed [15:0] cos_q=0,sin_q=0;
  wire ready_in,valid_out;
  wire [3:0] group_out,head_out;
  wire [1:0] token_out;
  wire [4:0] pair_out;
  wire signed [17:0] qfo,qso,kfo,kso;
  reg [71:0] input_mem [0:COUNT-1];
  reg [31:0] constant_mem [0:COUNT-1];
  reg [71:0] expected_mem [0:COUNT-1];
  integer index, output_count=0;
  rotary_qk_pair_serial dut(
    .clk(clk),.rst_n(rst_n),.valid_in(valid_in),.ready_in(ready_in),
    .group_in(group_in),.token_in(token_in),.head_in(head_in),
    .pair_in(pair_in),.query_first_q12(qf),.query_second_q12(qs),
    .key_first_q12(kf),.key_second_q12(ks),.cosine_q15(cos_q),
    .sine_q15(sin_q),.valid_out(valid_out),.group_out(group_out),
    .token_out(token_out),.head_out(head_out),.pair_out(pair_out),
    .query_first_rotated_q12(qfo),.query_second_rotated_q12(qso),
    .key_first_rotated_q12(kfo),.key_second_rotated_q12(kso));
  always #2 clk=~clk;
  always @(posedge clk) begin
    #1;
    if(valid_out) begin
      if(group_out!==4'd7) $fatal(1,"group mismatch");
      if(token_out!==(output_count/(12*32))) $fatal(1,"token mismatch");
      if(head_out!==((output_count/32)%12)) $fatal(1,"head mismatch");
      if(pair_out!==(output_count%32)) $fatal(1,"pair mismatch");
      if({{kso,kfo,qso,qfo}}!==expected_mem[output_count])
        $fatal(1,"H0 rotary mismatch at vector %0d",output_count);
      output_count=output_count+1;
    end
  end
  initial begin
    $readmemh("{input_hex}",input_mem);
    $readmemh("{constant_hex}",constant_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk); @(negedge clk); rst_n=1;
    for(index=0;index<COUNT;index=index+1) begin
      @(negedge clk);
      valid_in=1;
      token_in=index/(12*32); head_in=(index/32)%12; pair_in=index%32;
      qf=input_mem[index][17:0]; qs=input_mem[index][35:18];
      kf=input_mem[index][53:36]; ks=input_mem[index][71:54];
      cos_q=constant_mem[index][15:0]; sin_q=constant_mem[index][31:16];
    end
    @(negedge clk); valid_in=0;
    wait(output_count==COUNT); repeat(2) @(posedge clk);
    $display("tb_rotary_qk_pair_h0: PASS"); $finish;
  end
  initial begin repeat(3000) @(posedge clk); $fatal(1,"timeout"); end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_rotary_qk_pair_h0"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_rotary_qk_pair_h0",
            "-o",
            str(build),
            str(RTL / "rotary_qk_pair_serial.sv"),
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
    assert "tb_rotary_qk_pair_h0: PASS" in run_result.stdout
