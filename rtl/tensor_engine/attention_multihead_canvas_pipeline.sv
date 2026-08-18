`timescale 1ns/1ps

module attention_multihead_canvas_pipeline #(
    parameter integer DATA_WIDTH = 18,
    parameter integer ACC_WIDTH = 48,
    parameter integer HEADS = 12,
    parameter integer INTERNAL_MAC = 1,
    parameter LUT_FILE = "rtl/tensor_engine/exp_neg_q16_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire block_start,
    output wire block_start_ready,
    input  wire load_valid,
    output wire load_ready,
    input  wire [3:0] load_head,
    input  wire [5:0] load_token,
    input  wire [5:0] load_channel,
    input  wire signed [DATA_WIDTH-1:0] load_query_q12,
    input  wire signed [DATA_WIDTH-1:0] load_key_q12,
    input  wire signed [DATA_WIDTH-1:0] load_value_q12,
    output wire [3:0] expected_head,
    input  wire canvas_read_valid,
    input  wire [3:0] canvas_read_head,
    input  wire [5:0] canvas_read_token,
    output wire canvas_read_data_valid,
    output wire [64*DATA_WIDTH-1:0] canvas_read_data_packed,
    output wire array_request_valid,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [3:0] array_request_tag,
    output wire [4*32*DATA_WIDTH-1:0] array_request_activations,
    output wire [6*32*DATA_WIDTH-1:0] array_request_weights,
    input  wire array_response_valid,
    input  wire [3:0] array_response_tag,
    input  wire [4*6*ACC_WIDTH-1:0] array_response_accumulators,
    output wire busy,
    output wire done
);

    wire controller_load_enable;
    wire controller_head_start;
    wire head_start_ready;
    wire head_done;
    wire head_tile_valid;
    wire head_tile_ready;
    wire [3:0] head_tile_group;
    wire [3:0] head_output_tile;
    wire [2:0] head_valid_channels;
    wire [4*6*DATA_WIDTH-1:0] head_data;
    wire head_busy;
    wire canvas_tile_done;
    wire canvas_tile_ready;
    wire accepted_load = load_valid && load_ready;

    assign load_ready = controller_load_enable && head_start_ready
        && load_head == expected_head;
    assign head_tile_ready = canvas_tile_ready;

    attention_multihead_controller #(
        .HEADS(HEADS), .LOADS_PER_HEAD(4096)
    ) controller (
        .clk(clk), .rst_n(rst_n), .block_start(block_start),
        .block_start_ready(block_start_ready), .load_fire(accepted_load),
        .load_enable(controller_load_enable), .expected_head(expected_head),
        .head_start_ready(head_start_ready), .head_start(controller_head_start),
        .head_done(head_done), .canvas_idle(canvas_tile_ready),
        .busy(busy), .done(done)
    );

    attention_head_pipeline #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .INTERNAL_MAC(INTERNAL_MAC), .LUT_FILE(LUT_FILE)
    ) head_pipeline (
        .clk(clk), .rst_n(rst_n), .load_valid(accepted_load),
        .query_load_valid(1'b0), .key_load_valid(1'b0),
        .value_load_valid(1'b0),
        .load_token(load_token), .load_channel(load_channel),
        .load_query_q12(load_query_q12), .load_key_q12(load_key_q12),
        .load_value_q12(load_value_q12), .start(controller_head_start),
        .start_ready(head_start_ready), .attention_tile_valid(head_tile_valid),
        .attention_tile_ready(head_tile_ready),
        .attention_group(head_tile_group),
        .attention_output_tile(head_output_tile),
        .attention_valid_channels(head_valid_channels),
        .attention_q12_packed(head_data),
        .array_request_valid(array_request_valid),
        .array_request_clear(array_request_clear),
        .array_request_last(array_request_last),
        .array_request_tag(array_request_tag),
        .array_request_activations(array_request_activations),
        .array_request_weights(array_request_weights),
        .array_response_valid(array_response_valid),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators),
        .busy(head_busy), .done(head_done)
    );

    attention_canvas_scratchpad_banked #(
        .DATA_WIDTH(DATA_WIDTH)
    ) canvas (
        .clk(clk), .rst_n(rst_n), .tile_valid(head_tile_valid),
        .tile_ready(canvas_tile_ready), .tile_head(expected_head),
        .tile_group(head_tile_group), .tile_channel_tile(head_output_tile),
        .tile_valid_channels(head_valid_channels),
        .tile_data_packed(head_data), .tile_done(canvas_tile_done),
        .read_valid(canvas_read_valid), .read_head(canvas_read_head),
        .read_token(canvas_read_token),
        .read_data_valid(canvas_read_data_valid),
        .read_data_packed(canvas_read_data_packed)
    );

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && load_valid && controller_load_enable
            && load_head != expected_head)
            $error("QKV load head did not match the requested head");
`endif
    end

endmodule
