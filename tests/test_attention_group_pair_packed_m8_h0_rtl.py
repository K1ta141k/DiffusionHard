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


@pytest.mark.parametrize(
    "internal_mac",
    [True, False],
    ids=["internal-array", "external-array"],
)
def test_packed_qk_and_fixed18_pv_share_array_on_h0(
    tmp_path: Path,
    internal_mac: bool,
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
    value = (
        torch.round(qkv.view(64, 3, 12, 64)[:, 2, 0].double() * 4096.0)
        .to(torch.int64)
        .clamp(-(1 << 17), (1 << 17) - 1)
    )
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
    sums = exponentials.sum(dim=-1, keepdim=True)
    reciprocals_q14 = ((1 << 30) + sums // 2) // sums
    probabilities = (
        (exponentials * reciprocals_q14 + (1 << 13)) >> 14
    ).clamp(max=65535)
    expected_attention = _symmetric_round_shift(
        probabilities @ value, 16
    ).clamp(-(1 << 17), (1 << 17) - 1)

    vector_files = {}
    for name, tensor, transpose in (
        ("query", query, False),
        ("key", key, False),
        ("value", value, True),
    ):
        path = tmp_path / f"{name}_blocks.hex"
        if not transpose:
            lines = [
                _packed_hex(
                    [int(v) for v in tensor[token, block * 16 : (block + 1) * 16]],
                    18,
                )
                for token in range(64)
                for block in range(4)
            ]
        else:
            lines = [
                _packed_hex(
                    [int(tensor[block * 16 + lane, channel]) for lane in range(16)],
                    18,
                )
                for block in range(4)
                for channel in range(64)
            ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        vector_files[name] = path
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
                f"{int(v):0{(width + 3) // 4}x}" for v in values
            )
            + "\n",
            encoding="utf-8",
        )
        scale_files[name] = path
    expected_hex = tmp_path / "expected_attention_tiles.hex"
    expected_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(expected_attention[group * 4 + row, tile * 6 + lane])
                    if tile * 6 + lane < 64
                    else 0
                    for row in range(4)
                    for lane in range(6)
                ],
                18,
            )
            for group in range(16)
            for tile in range(11)
        )
        + "\n",
        encoding="utf-8",
    )
    expected_probability_hex = tmp_path / "expected_probabilities.hex"
    expected_probability_hex.write_text(
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
    expected_score_hex = tmp_path / "expected_score_pairs.hex"
    expected_score_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(scores[group_pair * 8 + row, tile * 6 + lane])
                    if tile * 6 + lane < 64
                    else 0
                    for row in range(8)
                    for lane in range(6)
                ],
                18,
            )
            for group_pair in range(8)
            for tile in range(11)
        )
        + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_attention_group_pair_packed_m8_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_attention_group_pair_packed_m8_h0;
  reg clk=0,rst_n=0,scale_load_valid=0,start=0;
  reg [2:0] group_pair_in=0;
  reg [5:0] scale_load_token=0;
  reg [17:0] scale_load_query_maximum=0,scale_load_key_maximum=0;
  reg [23:0] scale_load_query_multiplier_q17=0;
  reg [23:0] scale_load_key_multiplier_q17=0;
  wire start_ready,query_read_valid,key_read_valid,value_read_valid;
  wire [5:0] query_read_token,key_read_token,value_read_channel;
  wire [1:0] query_read_channel_block,key_read_channel_block,value_read_key_block;
  reg query_data_valid=0,key_data_valid=0,value_data_valid=0;
  reg [287:0] query_data=0,key_data=0,value_data=0;
  wire attention_tile_valid,busy,done;
  wire [3:0] attention_group,attention_output_tile;
  wire [2:0] attention_valid_channels;
  wire [431:0] attention_q12_packed;
  wire external_request_valid,external_request_narrow;
  wire external_request_clear,external_request_last;
  wire [5:0] external_request_tag,external_response_tag;
  wire [2303:0] external_attention_activations;
  wire [3455:0] external_attention_weights;
  wire [2047:0] external_narrow_activations;
  wire [1535:0] external_narrow_weights;
  wire external_response_valid,external_response_narrow;
  wire [1151:0] external_attention_accumulators;
  wire [1535:0] external_narrow_accumulators;
  reg [287:0] query_mem [0:255],key_mem [0:255],value_mem [0:255];
  reg [17:0] query_max_mem [0:63],key_max_mem [0:63];
  reg [23:0] query_multiplier_mem [0:63],key_multiplier_mem [0:63];
  reg [431:0] expected_mem [0:175];
  reg [4095:0] expected_probability_mem [0:15];
  reg [863:0] expected_score_mem [0:87];
  integer token,pair,tile_count=0,score_count=0,probability_count=0,cycles=0;
  attention_group_pair_pipeline_packed_m8 #(
    .INTERNAL_MAC({1 if internal_mac else 0})
  ) dut(
    .clk(clk),.rst_n(rst_n),.scale_load_valid(scale_load_valid),
    .scale_load_token(scale_load_token),
    .scale_load_query_maximum(scale_load_query_maximum),
    .scale_load_key_maximum(scale_load_key_maximum),
    .scale_load_query_multiplier_q17(scale_load_query_multiplier_q17),
    .scale_load_key_multiplier_q17(scale_load_key_multiplier_q17),
    .start(start),.group_pair_in(group_pair_in),.start_ready(start_ready),
    .query_read_valid(query_read_valid),.query_read_token(query_read_token),
    .query_read_channel_block(query_read_channel_block),
    .query_data_valid(query_data_valid),.query_data_q12_packed(query_data),
    .key_read_valid(key_read_valid),.key_read_token(key_read_token),
    .key_read_channel_block(key_read_channel_block),
    .key_data_valid(key_data_valid),.key_data_q12_packed(key_data),
    .value_read_valid(value_read_valid),.value_read_key_block(value_read_key_block),
    .value_read_channel(value_read_channel),.value_data_valid(value_data_valid),
    .value_data_q12_packed(value_data),.attention_tile_valid(attention_tile_valid),
    .attention_tile_ready(1'b1),.attention_group(attention_group),
    .attention_output_tile(attention_output_tile),
    .attention_valid_channels(attention_valid_channels),
    .attention_q12_packed(attention_q12_packed),
    .mac_request_valid(external_request_valid),
    .mac_request_narrow_int8_mode(external_request_narrow),
    .mac_request_clear(external_request_clear),
    .mac_request_last(external_request_last),
    .mac_request_tag(external_request_tag),
    .mac_request_attention_activations(external_attention_activations),
    .mac_request_attention_weights(external_attention_weights),
    .mac_request_narrow_activations(external_narrow_activations),
    .mac_request_narrow_weights(external_narrow_weights),
    .mac_response_valid(external_response_valid),
    .mac_response_narrow_int8_mode(external_response_narrow),
    .mac_response_tag(external_response_tag),
    .mac_response_attention_accumulators(external_attention_accumulators),
    .mac_response_narrow_accumulators(external_narrow_accumulators),
    .busy(busy),.done(done));
  mixed_precision_packed_m8_mac_tile_pipelined #(
    .N_LANES(6),.TAG_WIDTH(6)
  ) external_array(
    .clk(clk),.rst_n(rst_n),.valid_in(external_request_valid),
    .narrow_int8_mode(external_request_narrow),
    .clear_accumulators(external_request_clear),
    .last_k_tile(external_request_last),.tag_in(external_request_tag),
    .attention_activations_packed(external_attention_activations),
    .attention_weights_packed(external_attention_weights),
    .mlp_activations_packed(external_narrow_activations),
    .mlp_weights_packed(external_narrow_weights),
    .valid_out(external_response_valid),
    .narrow_int8_mode_out(external_response_narrow),
    .tag_out(external_response_tag),
    .attention_accumulators_packed(external_attention_accumulators),
    .mlp_accumulators_packed(external_narrow_accumulators));
  always #2 clk=~clk;
  always @(posedge clk) begin
    query_data_valid<=query_read_valid; key_data_valid<=key_read_valid;
    value_data_valid<=value_read_valid;
    if(query_read_valid) query_data<=query_mem[query_read_token*4+query_read_channel_block];
    if(key_read_valid) key_data<=key_mem[key_read_token*4+key_read_channel_block];
    if(value_read_valid) value_data<=value_mem[value_read_key_block*64+value_read_channel];
    if(busy) cycles<=cycles+1;
    #1;
    if(dut.qk_softmax.pair_score_valid) begin
      if(dut.qk_softmax.pair_scores!==expected_score_mem[score_count]) begin
        $display("score actual=%0d expected=%0d",
                 $signed(dut.qk_softmax.pair_scores[17:0]),
                 $signed(expected_score_mem[score_count][17:0]));
        $fatal(1,"score mismatch at %0d",score_count);
      end
      score_count=score_count+1;
    end
    if(dut.probability_group_valid && dut.probability_group_ready) begin
      if(dut.probability_group!==probability_count
         || dut.probabilities_q16!==expected_probability_mem[probability_count]) begin
        $display("probability actual=%0d expected=%0d",
                 dut.probabilities_q16[15:0],
                 expected_probability_mem[probability_count][15:0]);
        $fatal(1,"probability mismatch at %0d",probability_count);
      end
      probability_count=probability_count+1;
    end
    if(attention_tile_valid) begin
      if(attention_group!==(tile_count/11)
         || attention_output_tile!==(tile_count%11))
        $fatal(1,"attention tile tag mismatch at %0d",tile_count);
      if(attention_valid_channels!==(((tile_count%11)==10)?4:6))
        $fatal(1,"attention valid-channel mismatch at %0d",tile_count);
      if(attention_q12_packed!==expected_mem[tile_count]) begin
        $display("first actual=%0d expected=%0d",
                 $signed(attention_q12_packed[17:0]),
                 $signed(expected_mem[tile_count][17:0]));
        $fatal(1,"attention data mismatch at %0d",tile_count);
      end
      tile_count=tile_count+1;
    end
  end
  initial begin
    $readmemh("{vector_files['query']}",query_mem);
    $readmemh("{vector_files['key']}",key_mem);
    $readmemh("{vector_files['value']}",value_mem);
    $readmemh("{scale_files['query_max']}",query_max_mem);
    $readmemh("{scale_files['key_max']}",key_max_mem);
    $readmemh("{scale_files['query_multiplier']}",query_multiplier_mem);
    $readmemh("{scale_files['key_multiplier']}",key_multiplier_mem);
    $readmemh("{expected_hex}",expected_mem);
    $readmemh("{expected_probability_hex}",expected_probability_mem);
    $readmemh("{expected_score_hex}",expected_score_mem);
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
    if(tile_count!=176 || score_count!=88 || probability_count!=16)
      $fatal(1,"missing connected attention results");
    $display("tb_attention_group_pair_packed_m8_h0: PASS values=4096 cycles=%0d",
             cycles);
    $finish;
  end
endmodule
""",
        encoding="utf-8",
    )
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
        "attention_pv_group_scheduler.sv",
        "attention_group_pair_pipeline_packed_m8.sv",
    ]
    build = tmp_path / "tb_attention_group_pair_packed_m8_h0"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_attention_group_pair_packed_m8_h0",
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
    assert "tb_attention_group_pair_packed_m8_h0: PASS values=4096" in run_result.stdout
    cycle_match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert cycle_match is not None
    assert int(cycle_match.group(1)) == 11216
