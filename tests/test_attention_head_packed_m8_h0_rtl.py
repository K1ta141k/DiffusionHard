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
def test_packed_m8_head_tracks_scales_and_matches_all_h0_outputs(
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

    qkv_hex = tmp_path / "packed_head0_qkv.hex"
    qkv_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(query[token, channel]),
                    int(key[token, channel]),
                    int(value[token, channel]),
                ],
                18,
            )
            for token in range(64)
            for channel in range(64)
        )
        + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "packed_head0_expected.hex"
    expected_hex.write_text(
        "\n".join(
            f"{int(expected_attention[token, channel]) & 0x3FFFF:05x}"
            for token in range(64)
            for channel in range(64)
        )
        + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_attention_head_packed_m8_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_attention_head_packed_m8_h0;
  reg clk=0,rst_n=0,load_valid=0,start=0,attention_tile_ready=1;
  reg [5:0] load_token=0,load_channel=0;
  reg signed [17:0] load_q=0,load_k=0,load_v=0;
  wire start_ready,scales_ready,attention_tile_valid,busy,done;
  wire [3:0] attention_group,attention_output_tile;
  wire [2:0] attention_valid_channels;
  wire [431:0] attention_packed;
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
  reg [53:0] qkv_mem [0:4095];
  reg [17:0] expected_mem [0:4095];
  integer index,tile_count=0,query_row,channel_lane,channel,token;
  integer cycle_count=0,scale_wait_cycles=0;
  attention_head_pipeline_packed_m8 #(
    .INTERNAL_MAC({1 if internal_mac else 0})
  ) dut(
    .clk(clk),.rst_n(rst_n),.load_valid(load_valid),
    .query_load_valid(1'b0),.key_load_valid(1'b0),
    .value_load_valid(1'b0),.load_token(load_token),
    .load_channel(load_channel),.value_load_token(6'b0),
    .value_load_channel(6'b0),.load_query_q12(load_q),
    .load_key_q12(load_k),.load_value_q12(load_v),.start(start),
    .start_ready(start_ready),.scales_ready(scales_ready),
    .attention_tile_valid(attention_tile_valid),
    .attention_tile_ready(attention_tile_ready),
    .attention_group(attention_group),
    .attention_output_tile(attention_output_tile),
    .attention_valid_channels(attention_valid_channels),
    .attention_q12_packed(attention_packed),
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
    if(busy) cycle_count=cycle_count+1;
    #1;
    if(attention_tile_valid) begin
      if(attention_group!==(tile_count/11)
         || attention_output_tile!==(tile_count%11))
        $fatal(1,"packed-head tile tag mismatch at %0d",tile_count);
      if(attention_valid_channels!==(((tile_count%11)==10)?4:6))
        $fatal(1,"packed-head valid channel mismatch");
      for(query_row=0;query_row<4;query_row=query_row+1)
        for(channel_lane=0;channel_lane<attention_valid_channels;
            channel_lane=channel_lane+1) begin
          token=attention_group*4+query_row;
          channel=attention_output_tile*6+channel_lane;
          if($signed(attention_packed[(query_row*6+channel_lane)*18 +: 18])
             !==$signed(expected_mem[token*64+channel]))
            $fatal(1,"packed-head mismatch token %0d channel %0d",
                   token,channel);
        end
      tile_count=tile_count+1;
    end
  end
  initial begin
    $readmemh("{qkv_hex}",qkv_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(index=0;index<4096;index=index+1) begin
      @(negedge clk);load_valid=1;load_token=index/64;
      load_channel=index%64;load_q=qkv_mem[index][17:0];
      load_k=qkv_mem[index][35:18];load_v=qkv_mem[index][53:36];
    end
    @(negedge clk);load_valid=0;
    while(!start_ready) begin
      @(negedge clk);scale_wait_cycles=scale_wait_cycles+1;
    end
    if(!scales_ready) $fatal(1,"dynamic scale table was not complete");
    start=1;@(negedge clk);start=0;
    wait(done);repeat(2) @(posedge clk);
    if(tile_count!=176) $fatal(1,"missing packed-head attention tiles");
    if(busy) $fatal(1,"packed attention head remained busy");
    $display("tb_attention_head_packed_m8_h0: PASS values=4096 cycles=%0d scale_wait=%0d",
             cycle_count,scale_wait_cycles);
    $finish;
  end
  initial begin repeat(30000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    sources = [
        "attention_head_scratchpad_banked.sv",
        "attention_head_scratchpad_qk_combined_banked.sv",
        "unsigned_divider_iterative.sv",
        "attention_dynamic_vector_scale_tracker.sv",
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
        "attention_head_pipeline_packed_m8.sv",
    ]
    build = tmp_path / "tb_attention_head_packed_m8_h0"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_attention_head_packed_m8_h0",
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
    assert "tb_attention_head_packed_m8_h0: PASS values=4096" in run_result.stdout
    cycle_match = re.search(r"cycles=(\d+) scale_wait=(\d+)", run_result.stdout)
    assert cycle_match is not None
    assert int(cycle_match.group(1)) == 11232
    assert int(cycle_match.group(2)) == 26
