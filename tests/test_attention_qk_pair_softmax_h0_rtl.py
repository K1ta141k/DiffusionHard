from __future__ import annotations

import math
import re
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


def test_packed_qk_pair_feeds_existing_softmax_bit_exactly(
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
    query = query[:, 0].to(torch.int64)
    key = key[:, 0].to(torch.int64)
    query_maxima = query.abs().amax(dim=1).clamp(min=1)
    key_maxima = key.abs().amax(dim=1).clamp(min=1)
    query_multipliers = torch.round(
        127 * (1 << 17) / query_maxima.double()
    ).to(torch.int64)
    key_multipliers = torch.round(
        127 * (1 << 17) / key_maxima.double()
    ).to(torch.int64)
    query_int8 = _symmetric_round_shift(
        query * query_multipliers[:, None], 17
    ).clamp(-127, 127)
    key_int8 = _symmetric_round_shift(
        key * key_multipliers[:, None], 17
    ).clamp(-127, 127)
    dots = query_int8 @ key_int8.t()
    score_multipliers = torch.round(
        query_maxima[:, None].double()
        * key_maxima[None, :].double()
        * (1 << 28)
        / (127 * 127 * (1 << 17))
    ).to(torch.int64)
    scores = _symmetric_round_shift(dots * score_multipliers, 28).clamp(
        -(1 << 17), (1 << 17) - 1
    )
    deltas = (scores - scores.amax(dim=-1, keepdim=True)).clamp(
        min=-16384, max=0
    )
    addresses = ((-deltas + 8) >> 4).clamp(max=1024)
    exponential_lut = torch.tensor(
        [
            min(65535, round(math.exp(-index / 64.0) * 65536))
            for index in range(1025)
        ],
        dtype=torch.int64,
    )
    exponentials = exponential_lut[addresses]
    exponential_sums = exponentials.sum(dim=-1, keepdim=True)
    reciprocals_q14 = ((1 << 30) + exponential_sums // 2) // exponential_sums
    probabilities = (
        (exponentials * reciprocals_q14 + (1 << 13)) >> 14
    ).clamp(max=65535)

    query_blocks = tmp_path / "query_blocks.hex"
    key_blocks = tmp_path / "key_blocks.hex"
    query_blocks.write_text(
        "\n".join(
            _packed_hex(
                [int(value) for value in query[token, block * 16 : (block + 1) * 16]],
                18,
            )
            for token in range(64)
            for block in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    key_blocks.write_text(
        "\n".join(
            _packed_hex(
                [int(value) for value in key[token, block * 16 : (block + 1) * 16]],
                18,
            )
            for token in range(64)
            for block in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    scale_files = {}
    for name, values, width in (
        ("query_max", query_maxima, 18),
        ("key_max", key_maxima, 18),
        ("query_multiplier", query_multipliers, 24),
        ("key_multiplier", key_multipliers, 24),
    ):
        path = tmp_path / f"{name}.hex"
        path.write_text(
            "\n".join(
                f"{int(value):0{(width + 3) // 4}x}" for value in values
            )
            + "\n",
            encoding="utf-8",
        )
        scale_files[name] = path
    expected_hex = tmp_path / "expected_probabilities.hex"
    expected_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(probabilities[group * 4 + row, key_index])
                    for row in range(4)
                    for key_index in range(64)
                ],
                16,
            )
            for group in range(16)
        )
        + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_packed_qk_pair_softmax_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_packed_qk_pair_softmax_h0;
  reg clk=0,rst_n=0,scale_load_valid=0,start=0;
  reg [2:0] group_pair_in=0;
  reg [5:0] scale_load_token=0;
  reg [17:0] scale_load_query_maximum=0,scale_load_key_maximum=0;
  reg [23:0] scale_load_query_multiplier_q17=0;
  reg [23:0] scale_load_key_multiplier_q17=0;
  wire start_ready,query_read_valid,key_read_valid;
  wire [5:0] query_read_token,key_read_token;
  wire [1:0] query_read_channel_block,key_read_channel_block;
  reg query_data_valid=0,key_data_valid=0;
  reg [287:0] query_data_q12_packed=0,key_data_q12_packed=0;
  wire probability_group_valid;
  wire [3:0] probability_group;
  wire [4095:0] probabilities_q16_packed;
  wire busy,done;
  reg [287:0] query_blocks [0:255],key_blocks [0:255];
  reg [17:0] query_max_mem [0:63],key_max_mem [0:63];
  reg [23:0] query_multiplier_mem [0:63],key_multiplier_mem [0:63];
  reg [4095:0] expected_mem [0:15];
  integer token,pair,probability_count=0,cycles=0;
  attention_qk_group_pair_softmax_pipeline dut(
    .clk(clk),.rst_n(rst_n),.scale_load_valid(scale_load_valid),
    .scale_load_token(scale_load_token),
    .scale_load_query_maximum(scale_load_query_maximum),
    .scale_load_key_maximum(scale_load_key_maximum),
    .scale_load_query_multiplier_q17(scale_load_query_multiplier_q17),
    .scale_load_key_multiplier_q17(scale_load_key_multiplier_q17),
    .start(start),.group_pair_in(group_pair_in),.start_ready(start_ready),
    .query_read_valid(query_read_valid),.query_read_token(query_read_token),
    .query_read_channel_block(query_read_channel_block),
    .query_data_valid(query_data_valid),.query_data_q12_packed(query_data_q12_packed),
    .key_read_valid(key_read_valid),.key_read_token(key_read_token),
    .key_read_channel_block(key_read_channel_block),
    .key_data_valid(key_data_valid),.key_data_q12_packed(key_data_q12_packed),
    .probability_group_valid(probability_group_valid),
    .probability_group_ready(1'b1),.probability_group(probability_group),
    .probabilities_q16_packed(probabilities_q16_packed),
    .mac_request_valid(),.mac_request_clear(),.mac_request_last(),
    .mac_request_tag(),.mac_request_activations_int8(),
    .mac_request_weights_int8(),.mac_response_valid(1'b0),
    .mac_response_tag(0),.mac_response_accumulators(0),
    .busy(busy),.done(done));
  always #2 clk=~clk;
  always @(posedge clk) begin
    query_data_valid<=query_read_valid; key_data_valid<=key_read_valid;
    if(query_read_valid)
      query_data_q12_packed<=query_blocks[query_read_token*4+query_read_channel_block];
    if(key_read_valid)
      key_data_q12_packed<=key_blocks[key_read_token*4+key_read_channel_block];
    if(busy) cycles<=cycles+1;
    #1;
    if(probability_group_valid) begin
      if(probability_group!==probability_count)
        $fatal(1,"probability group mismatch");
      if(probabilities_q16_packed!==expected_mem[probability_count])
        $fatal(1,"probability mismatch at group %0d",probability_count);
      probability_count=probability_count+1;
    end
  end
  initial begin
    $readmemh("{query_blocks}",query_blocks);
    $readmemh("{key_blocks}",key_blocks);
    $readmemh("{scale_files['query_max']}",query_max_mem);
    $readmemh("{scale_files['key_max']}",key_max_mem);
    $readmemh("{scale_files['query_multiplier']}",query_multiplier_mem);
    $readmemh("{scale_files['key_multiplier']}",key_multiplier_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk); @(negedge clk); rst_n=1;
    for(token=0;token<64;token=token+1) begin
      @(negedge clk); scale_load_valid=1; scale_load_token=token;
      scale_load_query_maximum=query_max_mem[token];
      scale_load_key_maximum=key_max_mem[token];
      scale_load_query_multiplier_q17=query_multiplier_mem[token];
      scale_load_key_multiplier_q17=key_multiplier_mem[token];
    end
    @(negedge clk); scale_load_valid=0;
    for(pair=0;pair<8;pair=pair+1) begin
      group_pair_in=pair; start=1;
      @(negedge clk); start=0;
      wait(done); @(negedge clk);
    end
    repeat(2) @(posedge clk);
    if(probability_count!=16) $fatal(1,"missing probability groups");
    $display("tb_packed_qk_pair_softmax_h0: PASS probabilities=4096 cycles=%0d",
             cycles);
    $finish;
  end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_packed_qk_pair_softmax_h0"
    sources = [
        "unsigned_divider_iterative.sv",
        "exp_neg_q16_lut_bram.sv",
        "attention_softmax_row_q16.sv",
        "attention_score_group_softmax_stream.sv",
        "attention_dynamic_int8_quantizer_16.sv",
        "attention_dynamic_score_multiplier_q28.sv",
        "attention_dynamic_score_requantizer_q10.sv",
        "mixed_precision_token_pair_multiplier.sv",
        "mixed_precision_packed_m8_mac_tile_pipelined.sv",
        "attention_qk_group_pair_scheduler_packed_m8.sv",
        "attention_score_group_pair_buffer.sv",
        "attention_qk_group_pair_softmax_pipeline.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_packed_qk_pair_softmax_h0",
            "-o",
            str(build),
            *(str(RTL / source) for source in sources),
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
        timeout=180,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_packed_qk_pair_softmax_h0: PASS probabilities=4096" in run_result.stdout
    cycle_match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert cycle_match is not None
    assert int(cycle_match.group(1)) == 7272
