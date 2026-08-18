`timescale 1ns/1ps

module qkv_attention_head_pipeline_packed_m8 #(
    parameter integer INTERNAL_MAC = 1,
    parameter integer ARRAY_BACKPRESSURE = 0,
    parameter LUT_FILE = "rtl/tensor_engine/exp_neg_q16_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output wire start_ready,
    input  wire [3:0] head_in,
    input  wire metadata_valid,
    output wire metadata_ready,
    output wire parameter_request_valid,
    input  wire [3:0] metadata_head,
    input  wire [1:0] metadata_kind,
    input  wire [3:0] metadata_channel_tile,
    input  wire [6*24-1:0] metadata_multipliers_packed,
    input  wire [6*18-1:0] metadata_biases_q12_packed,
    input  wire weight_tile_valid,
    output wire weight_tile_ready,
    input  wire [3:0] weight_head,
    input  wire [1:0] weight_kind,
    input  wire [3:0] weight_channel_tile,
    input  wire [4:0] weight_input_tile,
    input  wire [6*32*16-1:0] weight_int16_packed,
    output wire [3:0] requested_head,
    output wire [1:0] requested_kind,
    output wire [3:0] requested_channel_tile,
    output wire [2:0] requested_valid_channels,
    output wire [11:0] requested_global_row,
    output wire normalized_read_valid,
    output wire [3:0] normalized_read_group,
    output wire [4:0] normalized_read_input_tile,
    input  wire normalized_read_data_valid,
    input  wire [4*32*18-1:0] normalized_q12_packed,
    input  wire constant_load_valid,
    input  wire [5:0] constant_load_token,
    input  wire [4:0] constant_load_pair,
    input  wire signed [15:0] constant_load_cosine_q15,
    input  wire signed [15:0] constant_load_sine_q15,
    output wire attention_tile_valid,
    input  wire attention_tile_ready,
    output wire [3:0] attention_group,
    output wire [3:0] attention_output_tile,
    output wire [2:0] attention_valid_channels,
    output wire [4*6*18-1:0] attention_q12_packed,
    output wire array_request_valid,
    input  wire array_request_ready,
    output wire array_request_narrow_int8_mode,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [7:0] array_request_tag,
    output wire [4*32*18-1:0] array_request_activations,
    output wire [6*32*18-1:0] array_request_weights,
    output wire [8*32*8-1:0] array_request_narrow_activations,
    output wire [6*32*8-1:0] array_request_narrow_weights,
    input  wire array_response_valid,
    input  wire array_response_narrow_int8_mode,
    input  wire [7:0] array_response_tag,
    input  wire [4*6*48-1:0] array_response_accumulators,
    input  wire [8*6*32-1:0] array_response_narrow_accumulators,
    output wire staging_busy,
    output wire attention_busy,
    output wire busy,
    output reg  done
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_STAGING = 2'd1;
    localparam [1:0] STATE_ATTENTION = 2'd2;

    reg [1:0] state;
    reg attention_start_pending;
    wire staging_start_ready;
    wire staging_done;
    wire query_write_valid;
    wire key_write_valid;
    wire value_write_valid;
    wire [5:0] query_key_write_token;
    wire [5:0] query_key_write_channel;
    wire signed [17:0] query_write_q12;
    wire signed [17:0] key_write_q12;
    wire [5:0] value_write_token;
    wire [5:0] value_write_channel;
    wire signed [17:0] value_write_q12;
    wire staging_array_valid;
    wire staging_array_clear;
    wire staging_array_last;
    wire [7:0] staging_array_tag;
    wire [4*32*18-1:0] staging_array_activations;
    wire [6*32*18-1:0] staging_array_weights;
    wire staging_projection_busy;
    wire staging_rotary_busy;
    wire head_start_ready;
    wire head_scales_ready;
    wire head_done;
    wire head_array_valid;
    wire head_array_narrow;
    wire head_array_clear;
    wire head_array_last;
    wire [5:0] head_array_tag;
    wire [4*32*18-1:0] head_array_activations;
    wire [6*32*18-1:0] head_array_weights;
    wire [8*32*8-1:0] head_array_narrow_activations;
    wire [6*32*8-1:0] head_array_narrow_weights;
    wire head_start = state == STATE_ATTENTION
        && attention_start_pending && head_start_ready;
    wire staging_phase = state == STATE_STAGING;
    wire selected_array_ready = INTERNAL_MAC || !ARRAY_BACKPRESSURE
        || array_request_ready;

    assign start_ready = state == STATE_IDLE && staging_start_ready;
    assign busy = state != STATE_IDLE;
    assign array_request_valid = staging_phase
        ? staging_array_valid : head_array_valid;
    assign array_request_narrow_int8_mode = staging_phase
        ? 1'b0 : head_array_narrow;
    assign array_request_clear = staging_phase
        ? staging_array_clear : head_array_clear;
    assign array_request_last = staging_phase
        ? staging_array_last : head_array_last;
    assign array_request_tag = staging_phase
        ? staging_array_tag : {2'b0, head_array_tag};
    assign array_request_activations = staging_phase
        ? staging_array_activations : head_array_activations;
    assign array_request_weights = staging_phase
        ? staging_array_weights : head_array_weights;
    assign array_request_narrow_activations = head_array_narrow_activations;
    assign array_request_narrow_weights = head_array_narrow_weights;

    qkv_head_staging_pipeline #(
        .INTERNAL_MAC(INTERNAL_MAC),
        .ARRAY_BACKPRESSURE(ARRAY_BACKPRESSURE)
    ) staging (
        .clk(clk), .rst_n(rst_n), .start(start && start_ready),
        .start_ready(staging_start_ready), .head_in(head_in),
        .metadata_valid(metadata_valid), .metadata_ready(metadata_ready),
        .parameter_request_valid(parameter_request_valid),
        .metadata_head(metadata_head), .metadata_kind(metadata_kind),
        .metadata_channel_tile(metadata_channel_tile),
        .metadata_multipliers_packed(metadata_multipliers_packed),
        .metadata_biases_q12_packed(metadata_biases_q12_packed),
        .weight_tile_valid(weight_tile_valid),
        .weight_tile_ready(weight_tile_ready), .weight_head(weight_head),
        .weight_kind(weight_kind), .weight_channel_tile(weight_channel_tile),
        .weight_input_tile(weight_input_tile),
        .weight_int16_packed(weight_int16_packed),
        .requested_head(requested_head), .requested_kind(requested_kind),
        .requested_channel_tile(requested_channel_tile),
        .requested_valid_channels(requested_valid_channels),
        .requested_global_row(requested_global_row),
        .normalized_read_valid(normalized_read_valid),
        .normalized_read_group(normalized_read_group),
        .normalized_read_input_tile(normalized_read_input_tile),
        .normalized_read_data_valid(normalized_read_data_valid),
        .normalized_q12_packed(normalized_q12_packed),
        .constant_load_valid(constant_load_valid),
        .constant_load_token(constant_load_token),
        .constant_load_pair(constant_load_pair),
        .constant_load_cosine_q15(constant_load_cosine_q15),
        .constant_load_sine_q15(constant_load_sine_q15),
        .query_write_valid(query_write_valid),
        .key_write_valid(key_write_valid), .value_write_valid(value_write_valid),
        .query_key_write_token(query_key_write_token),
        .query_key_write_channel(query_key_write_channel),
        .query_write_q12(query_write_q12), .key_write_q12(key_write_q12),
        .value_write_token(value_write_token),
        .value_write_channel(value_write_channel),
        .value_write_q12(value_write_q12),
        .array_request_valid(staging_array_valid),
        .array_request_ready(staging_phase && selected_array_ready),
        .array_request_clear(staging_array_clear),
        .array_request_last(staging_array_last),
        .array_request_tag(staging_array_tag),
        .array_request_activations(staging_array_activations),
        .array_request_weights(staging_array_weights),
        .array_response_valid(staging_phase && array_response_valid
            && !array_response_narrow_int8_mode),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators),
        .projection_busy(staging_projection_busy),
        .rotary_busy(staging_rotary_busy), .busy(staging_busy),
        .done(staging_done)
    );

    attention_head_pipeline_packed_m8 #(
        .INTERNAL_MAC(INTERNAL_MAC),
        .ARRAY_BACKPRESSURE(ARRAY_BACKPRESSURE),
        .SEPARATE_VALUE_LOAD_ADDRESS(1),
        .LUT_FILE(LUT_FILE)
    ) head (
        .clk(clk), .rst_n(rst_n), .load_valid(1'b0),
        .query_load_valid(query_write_valid),
        .key_load_valid(key_write_valid), .value_load_valid(value_write_valid),
        .load_token(query_key_write_token),
        .load_channel(query_key_write_channel),
        .value_load_token(value_write_token),
        .value_load_channel(value_write_channel),
        .load_query_q12(query_write_q12), .load_key_q12(key_write_q12),
        .load_value_q12(value_write_q12), .start(head_start),
        .start_ready(head_start_ready), .scales_ready(head_scales_ready),
        .attention_tile_valid(attention_tile_valid),
        .attention_tile_ready(attention_tile_ready),
        .attention_group(attention_group),
        .attention_output_tile(attention_output_tile),
        .attention_valid_channels(attention_valid_channels),
        .attention_q12_packed(attention_q12_packed),
        .mac_request_valid(head_array_valid),
        .mac_request_ready(!staging_phase && selected_array_ready),
        .mac_request_narrow_int8_mode(head_array_narrow),
        .mac_request_clear(head_array_clear),
        .mac_request_last(head_array_last),
        .mac_request_tag(head_array_tag),
        .mac_request_attention_activations(head_array_activations),
        .mac_request_attention_weights(head_array_weights),
        .mac_request_narrow_activations(head_array_narrow_activations),
        .mac_request_narrow_weights(head_array_narrow_weights),
        .mac_response_valid(!staging_phase && array_response_valid),
        .mac_response_narrow_int8_mode(
            array_response_narrow_int8_mode
        ),
        .mac_response_tag(array_response_tag[5:0]),
        .mac_response_attention_accumulators(
            array_response_accumulators
        ),
        .mac_response_narrow_accumulators(
            array_response_narrow_accumulators
        ),
        .busy(attention_busy), .done(head_done)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            attention_start_pending <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && start && start_ready)
                state <= STATE_STAGING;
            if (state == STATE_STAGING && staging_done) begin
                state <= STATE_ATTENTION;
                attention_start_pending <= 1'b1;
            end else if (head_start) begin
                attention_start_pending <= 1'b0;
            end
            if (state == STATE_ATTENTION && head_done) begin
                state <= STATE_IDLE;
                done <= 1'b1;
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && staging_array_valid && head_array_valid)
            $error("QKV staging and packed attention requested the array together");
        if (rst_n && head_start && !head_scales_ready)
            $error("packed attention started before dynamic scales completed");
`endif
    end

endmodule
