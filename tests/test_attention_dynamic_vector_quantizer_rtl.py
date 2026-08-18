from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from diffusion_accel.fixed_attention import (
    _load_tensors,
    _symmetric_round_shift,
    fixed_rotary_q12,
)


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


def test_dynamic_qk_scale_and_int8_quantizer_match_reference(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    if not (PACKAGE / "golden_tensors.safetensors").is_file():
        pytest.skip("H0 hardware package is unavailable")

    qkv = _load_tensors(
        PACKAGE / "golden_tensors.safetensors", ["folded.block_00.qkv"]
    )["folded.block_00.qkv"][0]
    tables = _load_tensors(
        PACKAGE / "folded_fp16_weights.safetensors",
        ["rotary.cos", "rotary.sin"],
    )
    _, _, query, key, _ = fixed_rotary_q12(
        qkv, tables["rotary.cos"].float(), tables["rotary.sin"].float()
    )
    query_vectors = query[:4, 0].to(torch.int64)
    key_vectors = key[:4, 0].to(torch.int64)
    query_vectors = torch.cat(
        [
            query_vectors,
            torch.zeros((1, 64), dtype=torch.int64),
            torch.tensor(
                [[-131072, 131071, -1, 1] * 16], dtype=torch.int64
            ),
        ]
    )
    key_vectors = torch.cat(
        [
            key_vectors,
            torch.zeros((1, 64), dtype=torch.int64),
            torch.tensor(
                [[131071, -131072, 32768, -32768] * 16],
                dtype=torch.int64,
            ),
        ]
    )
    vector_count = query_vectors.shape[0]
    query_maxima = query_vectors.abs().amax(dim=1).clamp(min=1)
    key_maxima = key_vectors.abs().amax(dim=1).clamp(min=1)
    query_multipliers = torch.round(
        127 * (1 << 17) / query_maxima.double()
    ).to(torch.int64)
    key_multipliers = torch.round(
        127 * (1 << 17) / key_maxima.double()
    ).to(torch.int64)
    query_int8 = _symmetric_round_shift(
        query_vectors * query_multipliers[:, None], 17
    ).clamp(-127, 127)
    key_int8 = _symmetric_round_shift(
        key_vectors * key_multipliers[:, None], 17
    ).clamp(-127, 127)

    query_hex = tmp_path / "query_q12.hex"
    key_hex = tmp_path / "key_q12.hex"
    query_hex.write_text(
        "\n".join(f"{int(value) & 0x3FFFF:05x}" for value in query_vectors.flatten())
        + "\n",
        encoding="utf-8",
    )
    key_hex.write_text(
        "\n".join(f"{int(value) & 0x3FFFF:05x}" for value in key_vectors.flatten())
        + "\n",
        encoding="utf-8",
    )
    expected_scale_hex = tmp_path / "expected_scales.hex"
    expected_scale_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(query_maxima[token]),
                    int(key_maxima[token]),
                    int(query_multipliers[token]),
                    int(key_multipliers[token]),
                ],
                32,
            )
            for token in range(vector_count)
        )
        + "\n",
        encoding="utf-8",
    )

    quantizer_values = []
    quantizer_multipliers = []
    quantizer_expected = []
    for token in range(vector_count):
        for source, quantized, multiplier in (
            (query_vectors, query_int8, query_multipliers),
            (key_vectors, key_int8, key_multipliers),
        ):
            for block in range(4):
                vector_slice = source[token, block * 16 : (block + 1) * 16]
                expected_slice = quantized[
                    token, block * 16 : (block + 1) * 16
                ]
                quantizer_values.append(
                    _packed_hex([int(value) for value in vector_slice], 18)
                )
                quantizer_multipliers.append(int(multiplier[token]))
                quantizer_expected.append(
                    _packed_hex([int(value) for value in expected_slice], 8)
                )
    values_hex = tmp_path / "quantizer_values.hex"
    multiplier_hex = tmp_path / "quantizer_multipliers.hex"
    expected_int8_hex = tmp_path / "expected_int8.hex"
    values_hex.write_text("\n".join(quantizer_values) + "\n", encoding="utf-8")
    multiplier_hex.write_text(
        "\n".join(f"{value:06x}" for value in quantizer_multipliers) + "\n",
        encoding="utf-8",
    )
    expected_int8_hex.write_text(
        "\n".join(quantizer_expected) + "\n", encoding="utf-8"
    )

    testbench = tmp_path / "tb_dynamic_qk_quantizer.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_dynamic_qk_quantizer;
  localparam VECTORS={vector_count};
  localparam CHUNKS={len(quantizer_values)};
  reg clk=0,rst_n=0,scale_input_valid=0,quantizer_valid=0;
  wire scale_input_ready,scale_valid;
  reg [5:0] token_in=0,channel_in=0;
  reg signed [17:0] query_q12_in=0,key_q12_in=0;
  wire [5:0] scale_token;
  wire [17:0] query_maximum,key_maximum;
  wire [23:0] query_multiplier_q17,key_multiplier_q17;
  reg [7:0] quantizer_tag=0;
  reg [287:0] values_q12_packed=0;
  reg [23:0] multiplier_q17=0;
  wire quantizer_output_valid;
  wire [7:0] quantizer_output_tag;
  wire [127:0] values_int8_packed;
  reg [17:0] query_mem [0:VECTORS*64-1];
  reg [17:0] key_mem [0:VECTORS*64-1];
  reg [127:0] expected_scales [0:VECTORS-1];
  reg [287:0] quantizer_values [0:CHUNKS-1];
  reg [23:0] quantizer_multipliers [0:CHUNKS-1];
  reg [127:0] expected_int8 [0:CHUNKS-1];
  integer token,channel,scale_count=0,chunk,quantized_count=0;

  attention_dynamic_vector_scale_tracker scale_tracker(
    .clk(clk),.rst_n(rst_n),.valid_in(scale_input_valid),
    .ready_in(scale_input_ready),.token_in(token_in),.channel_in(channel_in),
    .query_q12_in(query_q12_in),.key_q12_in(key_q12_in),
    .scale_valid(scale_valid),.scale_token(scale_token),
    .query_maximum(query_maximum),.key_maximum(key_maximum),
    .query_multiplier_q17(query_multiplier_q17),
    .key_multiplier_q17(key_multiplier_q17));
  attention_dynamic_int8_quantizer_16 quantizer(
    .clk(clk),.rst_n(rst_n),.valid_in(quantizer_valid),
    .tag_in(quantizer_tag),.values_q12_packed(values_q12_packed),
    .multiplier_q17(multiplier_q17),.valid_out(quantizer_output_valid),
    .tag_out(quantizer_output_tag),.values_int8_packed(values_int8_packed));

  always #2 clk=~clk;
  always @(posedge clk) begin
    #1;
    if(scale_valid) begin
      if(scale_token!==scale_count) $fatal(1,"scale token mismatch");
      if({{8'b0,key_multiplier_q17,8'b0,query_multiplier_q17,
           {{14{{1'b0}}}},key_maximum,{{14{{1'b0}}}},query_maximum}}
         !==expected_scales[scale_count])
        $fatal(1,"scale mismatch at token %0d",scale_count);
      scale_count=scale_count+1;
    end
    if(quantizer_output_valid) begin
      if(quantizer_output_tag!==quantized_count)
        $fatal(1,"quantizer tag mismatch");
      if(values_int8_packed!==expected_int8[quantized_count])
        $fatal(1,"quantizer data mismatch at chunk %0d",quantized_count);
      quantized_count=quantized_count+1;
    end
  end

  initial begin
    $readmemh("{query_hex}",query_mem);
    $readmemh("{key_hex}",key_mem);
    $readmemh("{expected_scale_hex}",expected_scales);
    $readmemh("{values_hex}",quantizer_values);
    $readmemh("{multiplier_hex}",quantizer_multipliers);
    $readmemh("{expected_int8_hex}",expected_int8);
    repeat(3) @(posedge clk); @(negedge clk); rst_n=1;
    for(token=0;token<VECTORS;token=token+1)
        for(channel=0;channel<64;channel=channel+1) begin
          @(negedge clk);
          token_in=token; channel_in=channel;
          query_q12_in=query_mem[token*64+channel];
          key_q12_in=key_mem[token*64+channel];
          #1; if(!scale_input_ready) $fatal(1,"scale tracker stalled");
          scale_input_valid=1;
        end
    @(negedge clk); scale_input_valid=0;
    wait(scale_count==VECTORS);
    for(chunk=0;chunk<CHUNKS;chunk=chunk+1) begin
      @(negedge clk); quantizer_tag=chunk;
      values_q12_packed=quantizer_values[chunk];
      multiplier_q17=quantizer_multipliers[chunk]; quantizer_valid=1;
    end
    @(negedge clk); quantizer_valid=0;
    wait(quantized_count==CHUNKS); repeat(2) @(posedge clk);
    $display("tb_dynamic_qk_quantizer: PASS scales=%0d chunks=%0d",
             scale_count,quantized_count);
    $finish;
  end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_dynamic_qk_quantizer"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_dynamic_qk_quantizer",
            "-o",
            str(build),
            str(RTL / "unsigned_divider_iterative.sv"),
            str(RTL / "attention_dynamic_vector_scale_tracker.sv"),
            str(RTL / "attention_dynamic_int8_quantizer_16.sv"),
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
    assert "tb_dynamic_qk_quantizer: PASS scales=6 chunks=48" in run_result.stdout
