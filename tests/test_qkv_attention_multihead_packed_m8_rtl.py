from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_packed_m8_multihead_controller_fills_two_head_canvas(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")

    testbench = tmp_path / "tb_qkv_attention_multihead_packed_m8.sv"
    testbench.write_text(
        """`timescale 1ns/1ps
module tb_qkv_attention_multihead_packed_m8;
  reg clk=0,rst_n=0,block_start=0,constant_load=0,canvas_read=0;
  reg [5:0] constant_token=0,canvas_read_token=0;
  reg [4:0] constant_pair=0,weight_input_tile=0;
  reg [3:0] canvas_read_head=0;
  reg normalized_data_valid=0,array_response_valid=0;
  reg array_response_narrow=0;
  reg [7:0] array_response_tag=0;
  wire block_start_ready,metadata_ready,weight_ready,normalized_read_valid;
  wire [3:0] normalized_group,requested_head,requested_channel_tile;
  wire [4:0] normalized_input_tile;
  wire [1:0] requested_kind;
  wire [2:0] requested_valid_channels;
  wire [11:0] requested_global_row;
  wire canvas_data_valid;
  wire [1151:0] canvas_data;
  wire array_request_valid,array_request_narrow;
  wire array_request_clear,array_request_last;
  wire [7:0] array_request_tag;
  wire [2303:0] array_activations;
  wire [3455:0] array_weights;
  wire [2047:0] array_narrow_activations;
  wire [1535:0] array_narrow_weights;
  wire [3:0] active_head;
  wire busy,done;
  integer index,channel,cycles=0,narrow_requests=0,wide_requests=0;
  reg saw_head_one=0;

  qkv_attention_multihead_canvas_pipeline_packed_m8 #(.HEADS(2)) dut(
    .clk(clk),.rst_n(rst_n),.block_start(block_start),
    .block_start_ready(block_start_ready),.metadata_valid(busy),
    .metadata_ready(metadata_ready),.metadata_head(requested_head),
    .metadata_kind(requested_kind),
    .metadata_channel_tile(requested_channel_tile),
    .metadata_multipliers_packed(144'b0),
    .metadata_biases_q12_packed(108'b0),
    .weight_tile_valid(busy),.weight_tile_ready(weight_ready),
    .weight_head(requested_head),.weight_kind(requested_kind),
    .weight_channel_tile(requested_channel_tile),
    .weight_input_tile(weight_input_tile),.weight_int16_packed(3072'b0),
    .requested_head(requested_head),.requested_kind(requested_kind),
    .requested_channel_tile(requested_channel_tile),
    .requested_valid_channels(requested_valid_channels),
    .requested_global_row(requested_global_row),
    .normalized_read_valid(normalized_read_valid),
    .normalized_read_group(normalized_group),
    .normalized_read_input_tile(normalized_input_tile),
    .normalized_read_data_valid(normalized_data_valid),
    .normalized_q12_packed(2304'b0),.constant_load_valid(constant_load),
    .constant_load_token(constant_token),.constant_load_pair(constant_pair),
    .constant_load_cosine_q15(16'sd32767),
    .constant_load_sine_q15(16'sd0),
    .canvas_read_valid(canvas_read),.canvas_read_head(canvas_read_head),
    .canvas_read_token(canvas_read_token),
    .canvas_read_data_valid(canvas_data_valid),
    .canvas_read_data_packed(canvas_data),
    .canvas_group_read_valid(1'b0),.canvas_group_read_head(4'b0),
    .canvas_group_read_group(4'b0),.canvas_group_read_data_valid(),
    .canvas_group_read_data_packed(),
    .array_request_valid(array_request_valid),
    .array_request_narrow_int8_mode(array_request_narrow),
    .array_request_clear(array_request_clear),
    .array_request_last(array_request_last),
    .array_request_tag(array_request_tag),
    .array_request_activations(array_activations),
    .array_request_weights(array_weights),
    .array_request_narrow_activations(array_narrow_activations),
    .array_request_narrow_weights(array_narrow_weights),
    .array_response_valid(array_response_valid),
    .array_response_narrow_int8_mode(array_response_narrow),
    .array_response_tag(array_response_tag),
    .array_response_accumulators(1152'b0),
    .array_response_narrow_accumulators(1536'b0),
    .active_head(active_head),.busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    normalized_data_valid<=normalized_read_valid;
    array_response_valid<=array_request_valid && array_request_last;
    array_response_narrow<=array_request_narrow;
    array_response_tag<=array_request_tag;
    if(weight_ready) begin
      if(weight_input_tile==23) weight_input_tile<=0;
      else weight_input_tile<=weight_input_tile+1'b1;
    end
    if(array_request_valid) begin
      if(array_request_narrow) narrow_requests=narrow_requests+1;
      else wide_requests=wide_requests+1;
    end
    if(busy) cycles=cycles+1;
    if(active_head==1 && busy) saw_head_one=1;
  end

  task check_canvas_zero;
    input [3:0] read_head;
    input [5:0] read_token;
    begin
      @(negedge clk);canvas_read=1;canvas_read_head=read_head;
      canvas_read_token=read_token;
      @(posedge clk);#1;
      if(!canvas_data_valid) $fatal(1,"canvas read valid missing");
      for(channel=0;channel<64;channel=channel+1)
        if($signed(canvas_data[channel*18 +: 18])!==0)
          $fatal(1,"packed canvas mismatch head %0d token %0d channel %0d",
                 read_head,read_token,channel);
      @(negedge clk);canvas_read=0;
    end
  endtask

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(index=0;index<2048;index=index+1) begin
      @(negedge clk);constant_load=1;
      constant_token=index/32;constant_pair=index%32;
    end
    @(negedge clk);constant_load=0;block_start=1;
    @(negedge clk);block_start=0;
    wait(done);repeat(3) @(posedge clk);
    if(!saw_head_one) $fatal(1,"packed second head never started");
    if(busy) $fatal(1,"packed multihead producer remained busy");
    if(narrow_requests==0 || wide_requests==0)
      $fatal(1,"packed controller did not exercise both array modes");
    check_canvas_zero(0,17);
    check_canvas_zero(1,63);
    $display("tb_qkv_attention_multihead_packed_m8: PASS cycles=%0d narrow=%0d wide=%0d",
             cycles,narrow_requests,wide_requests);
    $finish;
  end
  initial begin repeat(120000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
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
        "unsigned_divider_iterative.sv",
        "attention_dynamic_vector_scale_tracker.sv",
        "attention_dynamic_int8_quantizer_16.sv",
        "attention_dynamic_score_multiplier_q28.sv",
        "attention_dynamic_score_requantizer_q10.sv",
        "exp_neg_q16_lut_bram.sv",
        "attention_softmax_row_q16.sv",
        "attention_score_group_softmax_stream.sv",
        "attention_pv_group_scheduler.sv",
        "mixed_precision_token_pair_multiplier.sv",
        "mixed_precision_packed_m8_mac_tile_pipelined.sv",
        "attention_qk_group_pair_scheduler_packed_m8.sv",
        "attention_score_group_pair_buffer.sv",
        "attention_qk_group_pair_softmax_pipeline.sv",
        "attention_group_pair_pipeline_packed_m8.sv",
        "attention_head_pipeline_packed_m8.sv",
        "qkv_attention_head_pipeline_packed_m8.sv",
        "attention_canvas_scratchpad_banked.sv",
        "attention_canvas_grouped_scratchpad_banked.sv",
        "qkv_attention_multihead_canvas_pipeline_packed_m8.sv",
    ]
    build = tmp_path / "tb_qkv_attention_multihead_packed_m8"
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_qkv_attention_multihead_packed_m8",
            "-o",
            str(build),
            *(str(RTL / item) for item in sources),
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
    print(run_result.stdout, end="")
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_qkv_attention_multihead_packed_m8: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+) narrow=(\d+) wide=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) == 48_215
    assert int(match.group(2)) == 352
    assert int(match.group(3)) == 26_752
