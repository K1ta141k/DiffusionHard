from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.attention_int8 import dynamic_qk_fixed_pv_attention_q12
from diffusion_accel.fixed_attention import (
    fixed_attention_q12,
    fixed_qkv_projection_q12,
    fixed_rotary_q12,
)
from diffusion_accel.fixed_mlp import _load_tensors
from diffusion_accel.fixed_norm import fixed_layer_norm_q12


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"
RUN_LONG = os.environ.get("DIFFUSION_ACCEL_RUN_LONG_RTL") == "1"


def _packed_hex(values: list[int], width: int) -> str:
    mask = (1 << width) - 1
    packed = sum(
        (value & mask) << (index * width)
        for index, value in enumerate(values)
    )
    return f"{packed:0{(len(values) * width + 3) // 4}x}"


@pytest.mark.skipif(
    not RUN_LONG,
    reason="set DIFFUSION_ACCEL_RUN_LONG_RTL=1 for the 12-head RTL run",
)
@pytest.mark.parametrize("packed_m8", [False, True], ids=["fixed18", "packed-m8"])
def test_connected_qkv_attention_all_twelve_heads_match_h0(
    tmp_path: Path,
    packed_m8: bool,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    if not golden_path.is_file() or not weights_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    goldens = _load_tensors(golden_path, ["folded.embedding"])
    weights = _load_tensors(
        weights_path,
        [
            "block_00.qkv.weight",
            "block_00.qkv.bias",
            "rotary.cos",
            "rotary.sin",
        ],
    )
    _, normalized_q12, _ = fixed_layer_norm_q12(goldens["folded.embedding"][0])
    fixed_qkv, _, qkv_details = fixed_qkv_projection_q12(
        normalized_q12,
        weights["block_00.qkv.weight"],
        weights["block_00.qkv.bias"],
    )
    _, fixed_attention_values, _ = fixed_attention_q12(
        fixed_qkv, weights["rotary.cos"].float(), weights["rotary.sin"].float()
    )
    _, _, _, _, rotary_details = fixed_rotary_q12(
        fixed_qkv, weights["rotary.cos"].float(), weights["rotary.sin"].float()
    )
    if packed_m8:
        expected_attention_q12, _ = dynamic_qk_fixed_pv_attention_q12(
            fixed_qkv,
            weights["rotary.cos"].float(),
            weights["rotary.sin"].float(),
        )
    else:
        expected_attention_q12 = fixed_attention_values
    weight_q = qkv_details["tensors"]["weight_int16"]
    multipliers = qkv_details["tensors"]["requant_multiplier_q28"]
    biases = qkv_details["tensors"]["bias_q12"]
    cosine_q15 = rotary_details["tensors"]["cosine_q15"]
    sine_q15 = rotary_details["tensors"]["sine_q15"]

    normalized_hex = tmp_path / "multihead_normalized_q12.hex"
    normalized_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(normalized_q12[group * 4 + token_lane, tile * 32 + lane])
                    for token_lane in range(4)
                    for lane in range(32)
                ],
                18,
            )
            for group in range(16)
            for tile in range(24)
        )
        + "\n",
        encoding="utf-8",
    )

    tile_rows: list[list[int | None]] = []
    for head in range(12):
        for kind in range(3):
            for channel_tile in range(11):
                rows: list[int | None] = []
                for lane in range(6):
                    channel = channel_tile * 6 + lane
                    rows.append(
                        kind * 768 + head * 64 + channel
                        if channel < 64
                        else None
                    )
                tile_rows.append(rows)

    weight_hex = tmp_path / "multihead_qkv_weights.hex"
    weight_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    0 if row is None else int(weight_q[row, input_tile * 32 + lane])
                    for row in rows
                    for lane in range(32)
                ],
                16,
            )
            for rows in tile_rows
            for input_tile in range(24)
        )
        + "\n",
        encoding="utf-8",
    )
    multiplier_hex = tmp_path / "multihead_qkv_multipliers.hex"
    multiplier_hex.write_text(
        "\n".join(
            _packed_hex(
                [0 if row is None else int(multipliers[row]) for row in rows], 24
            )
            for rows in tile_rows
        )
        + "\n",
        encoding="utf-8",
    )
    bias_hex = tmp_path / "multihead_qkv_biases.hex"
    bias_hex.write_text(
        "\n".join(
            _packed_hex(
                [0 if row is None else int(biases[row]) for row in rows], 18
            )
            for rows in tile_rows
        )
        + "\n",
        encoding="utf-8",
    )
    constants_hex = tmp_path / "multihead_rotary_constants.hex"
    constants_hex.write_text(
        "\n".join(
            _packed_hex(
                [int(cosine_q15[token, pair]), int(sine_q15[token, pair])], 16
            )
            for token in range(64)
            for pair in range(32)
        )
        + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "multihead_attention_canvas.hex"
    expected_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(expected_attention_q12[token, head * 64 + channel])
                    for channel in range(64)
                ],
                18,
            )
            for head in range(12)
            for token in range(64)
        )
        + "\n",
        encoding="utf-8",
    )

    dut_module = (
        "qkv_attention_multihead_canvas_pipeline_packed_m8"
        if packed_m8
        else "qkv_attention_multihead_canvas_pipeline"
    )
    packed_dut_ports = ""
    if packed_m8:
        packed_dut_ports = """
    .array_request_narrow_int8_mode(array_request_narrow),
    .array_request_narrow_activations(array_narrow_activations),
    .array_request_narrow_weights(array_narrow_weights),
    .array_response_narrow_int8_mode(array_response_narrow),
    .array_response_narrow_accumulators(array_narrow_accumulators),"""
        shared_mac = """
  mixed_precision_packed_m8_mac_tile_pipelined #(
    .N_LANES(6),.TAG_WIDTH(8)
  ) shared_mac(
    .clk(clk),.rst_n(rst_n),.valid_in(array_request_valid),
    .narrow_int8_mode(array_request_narrow),
    .clear_accumulators(array_request_clear),
    .last_k_tile(array_request_last),.tag_in(array_request_tag),
    .attention_activations_packed(array_activations),
    .attention_weights_packed(array_weights),
    .mlp_activations_packed(array_narrow_activations),
    .mlp_weights_packed(array_narrow_weights),
    .valid_out(array_response_valid),
    .narrow_int8_mode_out(array_response_narrow),
    .tag_out(array_response_tag),
    .attention_accumulators_packed(array_response_accumulators),
    .mlp_accumulators_packed(array_narrow_accumulators));
"""
    else:
        shared_mac = """
  mixed_precision_mac_tile_pipelined #(
    .M_LANES(4),.N_LANES(6),.STORAGE_WIDTH(18),.ACC_WIDTH(48),.TAG_WIDTH(8)
  ) shared_mac(
    .clk(clk),.rst_n(rst_n),.valid_in(array_request_valid),
    .narrow_int8_mode(1'b0),.clear_accumulators(array_request_clear),
    .last_k_tile(array_request_last),.tag_in(array_request_tag),
    .activations_packed(array_activations),.weights_packed(array_weights),
    .valid_out(array_response_valid),.tag_out(array_response_tag),
    .accumulators_packed(array_response_accumulators));
  assign array_response_narrow=1'b0;
  assign array_narrow_accumulators=0;
"""
    testbench = tmp_path / "tb_qkv_attention_multihead_h0_connected.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_qkv_attention_multihead_h0_connected;
  reg clk=0,rst_n=0,block_start=0,constant_load=0,canvas_read=0;
  reg [5:0] constant_token=0,canvas_read_token=0;
  reg [4:0] constant_pair=0,weight_input_tile=0;
  reg [3:0] canvas_read_head=0;
  reg signed [15:0] constant_cosine=0,constant_sine=0;
  reg normalized_data_valid=0;
  reg [2303:0] normalized_data=0;
  reg [15:0] tile_index=0;
  wire block_start_ready,metadata_ready,weight_ready,normalized_read_valid;
  wire [3:0] normalized_group,requested_head,requested_channel_tile;
  wire [4:0] normalized_input_tile;
  wire [1:0] requested_kind;
  wire [2:0] requested_valid_channels;
  wire [11:0] requested_global_row;
  wire canvas_data_valid;
  wire [1151:0] canvas_data;
  wire array_request_valid,array_request_clear,array_request_last;
  wire array_request_narrow,array_response_narrow;
  wire [7:0] array_request_tag,array_response_tag;
  wire [2303:0] array_activations;
  wire [3455:0] array_weights;
  wire [2047:0] array_narrow_activations;
  wire [1535:0] array_narrow_weights,array_narrow_accumulators;
  wire array_response_valid;
  wire [1151:0] array_response_accumulators;
  wire [3:0] active_head;
  wire busy,done;
  reg [2303:0] normalized_mem [0:383];
  reg [3071:0] weight_mem [0:9503];
  reg [143:0] multiplier_mem [0:395];
  reg [107:0] bias_mem [0:395];
  reg [31:0] constant_mem [0:2047];
  reg [1151:0] expected_mem [0:767];
  integer index,head,token,channel,cycles=0;

  always @* tile_index=(requested_head*3+requested_kind)*11+
                            requested_channel_tile;
  {dut_module} #(.HEADS(12)) dut(
    .clk(clk),.rst_n(rst_n),.block_start(block_start),
    .block_start_ready(block_start_ready),.metadata_valid(busy),
    .metadata_ready(metadata_ready),.metadata_head(requested_head),
    .metadata_kind(requested_kind),
    .metadata_channel_tile(requested_channel_tile),
    .metadata_multipliers_packed(multiplier_mem[tile_index]),
    .metadata_biases_q12_packed(bias_mem[tile_index]),
    .weight_tile_valid(busy),.weight_tile_ready(weight_ready),
    .weight_head(requested_head),.weight_kind(requested_kind),
    .weight_channel_tile(requested_channel_tile),
    .weight_input_tile(weight_input_tile),
    .weight_int16_packed(weight_mem[tile_index*24+weight_input_tile]),
    .requested_head(requested_head),.requested_kind(requested_kind),
    .requested_channel_tile(requested_channel_tile),
    .requested_valid_channels(requested_valid_channels),
    .requested_global_row(requested_global_row),
    .normalized_read_valid(normalized_read_valid),
    .normalized_read_group(normalized_group),
    .normalized_read_input_tile(normalized_input_tile),
    .normalized_read_data_valid(normalized_data_valid),
    .normalized_q12_packed(normalized_data),.constant_load_valid(constant_load),
    .constant_load_token(constant_token),.constant_load_pair(constant_pair),
    .constant_load_cosine_q15(constant_cosine),
    .constant_load_sine_q15(constant_sine),
    .canvas_read_valid(canvas_read),.canvas_read_head(canvas_read_head),
    .canvas_read_token(canvas_read_token),.canvas_read_data_valid(canvas_data_valid),
    .canvas_read_data_packed(canvas_data),
    .array_request_valid(array_request_valid),
    .array_request_clear(array_request_clear),
    .array_request_last(array_request_last),.array_request_tag(array_request_tag),
    .array_request_activations(array_activations),
    .array_request_weights(array_weights),
{packed_dut_ports}
    .array_response_valid(array_response_valid),
    .array_response_tag(array_response_tag),
    .array_response_accumulators(array_response_accumulators),
    .active_head(active_head),.busy(busy),.done(done));

{shared_mac}

  always #2 clk=~clk;
  always @(posedge clk) begin
    normalized_data_valid<=normalized_read_valid;
    if(normalized_read_valid)
      normalized_data<=normalized_mem[normalized_group*24+normalized_input_tile];
    if(weight_ready) begin
      if(weight_input_tile==23) weight_input_tile<=0;
      else weight_input_tile<=weight_input_tile+1'b1;
    end
    if(busy) cycles=cycles+1;
  end

  initial begin
    $readmemh("{normalized_hex}",normalized_mem);
    $readmemh("{weight_hex}",weight_mem);
    $readmemh("{multiplier_hex}",multiplier_mem);
    $readmemh("{bias_hex}",bias_mem);
    $readmemh("{constants_hex}",constant_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(index=0;index<2048;index=index+1) begin
      @(negedge clk);constant_load=1;
      constant_token=index/32;constant_pair=index%32;
      constant_cosine=constant_mem[index][15:0];
      constant_sine=constant_mem[index][31:16];
    end
    @(negedge clk);constant_load=0;block_start=1;
    @(negedge clk);block_start=0;
    wait(done);repeat(3) @(posedge clk);
    if(busy || active_head!=11) $fatal(1,"multihead path did not finish head 11");
    for(head=0;head<12;head=head+1)
      for(token=0;token<64;token=token+1) begin
        @(negedge clk);canvas_read=1;canvas_read_head=head;
        canvas_read_token=token;
        @(posedge clk);#1;
        if(!canvas_data_valid) $fatal(1,"canvas read valid missing");
        for(channel=0;channel<64;channel=channel+1)
          if($signed(canvas_data[channel*18 +: 18])!==
             $signed(expected_mem[head*64+token][channel*18 +: 18]))
            $fatal(1,"multihead mismatch head %0d token %0d channel %0d",
                   head,token,channel);
      end
    @(negedge clk);canvas_read=0;
    $display("tb_qkv_attention_multihead_h0_connected: PASS cycles=%0d",cycles);
    $finish;
  end
  initial begin repeat(650000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )

    build = tmp_path / "tb_qkv_attention_multihead_h0_connected"
    sources = [
        "int8_mac_tile_pipelined.sv",
        "mixed_precision_mac_tile_pipelined.sv",
        "fixed_requantize.sv",
        "fixed_requantize_vector_serial.sv",
        "qkv_weight_tile_buffer.sv",
        "qkv_projection_output_tile_scheduler.sv",
        "qkv_projection_output_tile_scheduler_streaming.sv",
        "qkv_head_tile_controller.sv",
        "qkv_head_projection_pipeline.sv",
        "qkv_head_output_router.sv",
        "qk_unrotated_scratchpad_paired_uram.sv",
        "rotary_constant_table_bram.sv",
        "rotary_qk_pair_serial.sv",
        "rotary_head_writeback_scheduler.sv",
        "qkv_head_staging_pipeline.sv",
        "attention_head_scratchpad_banked.sv",
        "attention_head_scratchpad_qk_combined_banked.sv",
        "attention_qk_group_scheduler.sv",
        "unsigned_divider_iterative.sv",
        "exp_neg_q16_lut_bram.sv",
        "attention_softmax_row_q16.sv",
        "attention_score_group_softmax_stream.sv",
        "attention_pv_group_scheduler.sv",
        "attention_group_pipeline.sv",
        "attention_head_pipeline.sv",
        "qkv_attention_head_pipeline.sv",
        "attention_canvas_scratchpad_banked.sv",
        "attention_canvas_grouped_scratchpad_banked.sv",
        "qkv_attention_multihead_canvas_pipeline.sv",
    ]
    if packed_m8:
        sources.extend(
            [
                "attention_dynamic_vector_scale_tracker.sv",
                "attention_dynamic_int8_quantizer_16.sv",
                "attention_dynamic_score_multiplier_q28.sv",
                "attention_dynamic_score_requantizer_q10.sv",
                "mixed_precision_token_pair_multiplier.sv",
                "mixed_precision_packed_m8_mac_tile_pipelined.sv",
                "attention_qk_group_pair_scheduler_packed_m8.sv",
                "attention_score_group_pair_buffer.sv",
                "attention_qk_group_pair_softmax_pipeline.sv",
                "attention_group_pair_pipeline_packed_m8.sv",
                "attention_head_pipeline_packed_m8.sv",
                "qkv_attention_head_pipeline_packed_m8.sv",
                "qkv_attention_multihead_canvas_pipeline_packed_m8.sv",
            ]
        )
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-s", "tb_qkv_attention_multihead_h0_connected",
            "-o", str(build), *(str(RTL / item) for item in sources),
            str(testbench),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        ["vvp", str(build)], cwd=ROOT, check=False, capture_output=True, text=True
    )
    print(run_result.stdout, end="")
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_qkv_attention_multihead_h0_connected: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    cycles = int(match.group(1))
    if packed_m8:
        assert 311_000 <= cycles <= 315_000
    else:
        assert 565_000 <= cycles <= 567_000
