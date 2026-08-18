`timescale 1ns/1ps

module attention_head_pipeline #(
    parameter integer DATA_WIDTH = 18,
    parameter integer ACC_WIDTH = 48,
    parameter integer GROUP_WIDTH = 4,
    parameter integer INTERNAL_MAC = 1,
    parameter integer SEPARATE_VALUE_LOAD_ADDRESS = 0,
    parameter LUT_FILE = "rtl/tensor_engine/exp_neg_q16_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire load_valid,
    input  wire query_load_valid,
    input  wire key_load_valid,
    input  wire value_load_valid,
    input  wire [5:0] load_token,
    input  wire [5:0] load_channel,
    input  wire [5:0] value_load_token,
    input  wire [5:0] value_load_channel,
    input  wire signed [DATA_WIDTH-1:0] load_query_q12,
    input  wire signed [DATA_WIDTH-1:0] load_key_q12,
    input  wire signed [DATA_WIDTH-1:0] load_value_q12,
    input  wire start,
    output wire start_ready,
    output wire attention_tile_valid,
    input  wire attention_tile_ready,
    output wire [GROUP_WIDTH-1:0] attention_group,
    output wire [3:0] attention_output_tile,
    output wire [2:0] attention_valid_channels,
    output wire [4*6*DATA_WIDTH-1:0] attention_q12_packed,
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
    output reg  done
);

    localparam STATE_IDLE = 1'b0;
    localparam STATE_RUN = 1'b1;

    reg state;
    reg [GROUP_WIDTH-1:0] active_group;
    reg group_start_pending;

    wire query_read_valid;
    wire [5:0] query_read_token;
    wire [1:0] query_read_channel_block;
    wire query_data_valid;
    wire [16*DATA_WIDTH-1:0] query_data;
    wire key_read_valid;
    wire [5:0] key_read_token;
    wire [1:0] key_read_channel_block;
    wire key_data_valid;
    wire [16*DATA_WIDTH-1:0] key_data;
    wire value_read_valid;
    wire [1:0] value_read_key_block;
    wire [5:0] value_read_channel;
    wire value_data_valid;
    wire [16*DATA_WIDTH-1:0] value_data;
    wire group_start_ready;
    wire group_start = (state == STATE_RUN) && group_start_pending
        && group_start_ready;
    wire group_busy;
    wire group_done;

    assign start_ready = (state == STATE_IDLE);
    assign busy = (state != STATE_IDLE);

    attention_head_scratchpad_banked #(
        .DATA_WIDTH(DATA_WIDTH),
        .SEPARATE_VALUE_LOAD_ADDRESS(SEPARATE_VALUE_LOAD_ADDRESS)
    ) scratchpad (
        .clk(clk), .load_valid(load_valid),
        .query_load_valid(query_load_valid),
        .key_load_valid(key_load_valid),
        .value_load_valid(value_load_valid), .load_token(load_token),
        .load_channel(load_channel), .value_load_token(value_load_token),
        .value_load_channel(value_load_channel),
        .load_query_q12(load_query_q12),
        .load_key_q12(load_key_q12), .load_value_q12(load_value_q12),
        .query_read_valid(query_read_valid),
        .query_read_token(query_read_token),
        .query_read_channel_block(query_read_channel_block),
        .query_data_valid(query_data_valid), .query_data_packed(query_data),
        .key_read_valid(key_read_valid), .key_read_token(key_read_token),
        .key_read_channel_block(key_read_channel_block),
        .key_data_valid(key_data_valid), .key_data_packed(key_data),
        .value_read_valid(value_read_valid),
        .value_read_key_block(value_read_key_block),
        .value_read_channel(value_read_channel),
        .value_data_valid(value_data_valid), .value_data_packed(value_data)
    );

    attention_group_pipeline #(
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .GROUP_WIDTH(GROUP_WIDTH), .INTERNAL_MAC(INTERNAL_MAC),
        .LUT_FILE(LUT_FILE)
    ) group_pipeline (
        .clk(clk), .rst_n(rst_n), .start(group_start),
        .group_in(active_group), .start_ready(group_start_ready),
        .query_read_valid(query_read_valid),
        .query_read_token(query_read_token),
        .query_read_channel_block(query_read_channel_block),
        .query_data_valid(query_data_valid), .query_data_packed(query_data),
        .key_read_valid(key_read_valid), .key_read_token(key_read_token),
        .key_read_channel_block(key_read_channel_block),
        .key_data_valid(key_data_valid), .key_data_packed(key_data),
        .value_read_valid(value_read_valid),
        .value_read_key_block(value_read_key_block),
        .value_read_channel(value_read_channel),
        .value_data_valid(value_data_valid), .value_data_packed(value_data),
        .attention_tile_valid(attention_tile_valid),
        .attention_tile_ready(attention_tile_ready),
        .attention_group(attention_group),
        .attention_output_tile(attention_output_tile),
        .attention_valid_channels(attention_valid_channels),
        .attention_q12_packed(attention_q12_packed),
        .array_request_valid(array_request_valid),
        .array_request_clear(array_request_clear),
        .array_request_last(array_request_last),
        .array_request_tag(array_request_tag),
        .array_request_activations(array_request_activations),
        .array_request_weights(array_request_weights),
        .array_response_valid(array_response_valid),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators),
        .busy(group_busy), .done(group_done)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_group <= 0;
            group_start_pending <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && start) begin
                state <= STATE_RUN;
                active_group <= 0;
                group_start_pending <= 1'b1;
            end else if (group_start) begin
                group_start_pending <= 1'b0;
            end
            if (state == STATE_RUN && group_done) begin
                if (active_group == 15) begin
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else begin
                    active_group <= active_group + 1'b1;
                    group_start_pending <= 1'b1;
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && load_valid && state != STATE_IDLE)
            $error("attention head scratchpad load overlapped execution");
`endif
    end

endmodule
