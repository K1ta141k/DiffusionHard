`timescale 1ns/1ps

module qkv_attention_projection_block_pipeline_packed_m8 #(
    parameter integer HEADS = 12,
    parameter integer OUTPUT_TILES = 128,
    parameter LUT_FILE = "rtl/tensor_engine/exp_neg_q16_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire block_start,
    output wire block_start_ready,
    input  wire residual_load_valid,
    input  wire [3:0] residual_load_group,
    input  wire [6:0] residual_load_output_tile,
    input  wire [4*6*24-1:0] residual_load_q10_packed,
    input  wire residual_replay_read_valid,
    input  wire [3:0] residual_replay_read_group,
    input  wire [6:0] residual_replay_read_output_tile,
    output wire residual_replay_read_data_valid,
    output wire [4*6*24-1:0] residual_replay_read_q10_packed,
    input  wire qkv_metadata_valid,
    output wire qkv_metadata_ready,
    output wire qkv_parameter_request_valid,
    input  wire [3:0] qkv_metadata_head,
    input  wire [1:0] qkv_metadata_kind,
    input  wire [3:0] qkv_metadata_channel_tile,
    input  wire [6*24-1:0] qkv_metadata_multipliers_packed,
    input  wire [6*18-1:0] qkv_metadata_biases_q12_packed,
    input  wire qkv_weight_tile_valid,
    output wire qkv_weight_tile_ready,
    input  wire [3:0] qkv_weight_head,
    input  wire [1:0] qkv_weight_kind,
    input  wire [3:0] qkv_weight_channel_tile,
    input  wire [4:0] qkv_weight_input_tile,
    input  wire [6*32*16-1:0] qkv_weight_int16_packed,
    output wire [3:0] requested_qkv_head,
    output wire [1:0] requested_qkv_kind,
    output wire [3:0] requested_qkv_channel_tile,
    output wire [11:0] requested_qkv_global_row,
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
    input  wire projection_metadata_valid,
    output wire projection_metadata_ready,
    output wire projection_parameter_request_valid,
    input  wire [6:0] projection_metadata_output_tile,
    input  wire [6*24-1:0] projection_metadata_multipliers_packed,
    input  wire projection_weight_tile_valid,
    output wire projection_weight_tile_ready,
    input  wire [6:0] projection_weight_output_tile,
    input  wire [4:0] projection_weight_input_tile,
    input  wire [6*32*8-1:0] projection_weight_int8_packed,
    output wire [6:0] requested_projection_output_tile,
    output wire block_tile_valid,
    input  wire block_tile_ready,
    output wire [3:0] block_group,
    output wire [6:0] block_output_tile,
    output wire [4*6*24-1:0] block_q10_packed,
    output wire array_request_valid,
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
    output wire producer_busy,
    output wire projection_busy,
    output wire busy,
    output reg  done
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_PRODUCE = 2'd1;
    localparam [1:0] STATE_PROJECT = 2'd2;

    reg [1:0] state;
    reg projection_start_pending;
    wire producer_start_ready;
    wire producer_done;
    wire producer_array_valid;
    wire producer_array_narrow;
    wire producer_array_clear;
    wire producer_array_last;
    wire [7:0] producer_array_tag;
    wire [4*32*18-1:0] producer_array_activations;
    wire [6*32*18-1:0] producer_array_weights;
    wire [8*32*8-1:0] producer_array_narrow_activations;
    wire [6*32*8-1:0] producer_array_narrow_weights;
    wire projection_start_ready;
    wire projection_start = state == STATE_PROJECT
        && projection_start_pending && projection_start_ready;
    wire projection_done;
    wire projection_array_valid;
    wire projection_array_clear;
    wire projection_array_last;
    wire [7:0] projection_array_tag;
    wire [4*32*18-1:0] projection_array_activations;
    wire [6*32*18-1:0] projection_array_weights;
    wire canvas_read_valid;
    wire [3:0] canvas_read_head;
    wire [5:0] canvas_read_token;
    wire canvas_read_data_valid;
    wire [64*18-1:0] canvas_read_data;
    wire canvas_group_read_valid;
    wire [3:0] canvas_group_read_head;
    wire [3:0] canvas_group_read_group;
    wire canvas_group_read_data_valid;
    wire [4*64*18-1:0] canvas_group_read_data;
    wire [3:0] producer_active_head;
    wire producer_phase = state == STATE_PRODUCE;

    assign block_start_ready = state == STATE_IDLE && producer_start_ready;
    assign busy = state != STATE_IDLE;
    assign array_request_valid = producer_phase
        ? producer_array_valid : projection_array_valid;
    assign array_request_narrow_int8_mode = producer_phase
        ? producer_array_narrow : 1'b0;
    assign array_request_clear = producer_phase
        ? producer_array_clear : projection_array_clear;
    assign array_request_last = producer_phase
        ? producer_array_last : projection_array_last;
    assign array_request_tag = producer_phase
        ? producer_array_tag : projection_array_tag;
    assign array_request_activations = producer_phase
        ? producer_array_activations : projection_array_activations;
    assign array_request_weights = producer_phase
        ? producer_array_weights : projection_array_weights;
    assign array_request_narrow_activations =
        producer_array_narrow_activations;
    assign array_request_narrow_weights = producer_array_narrow_weights;

    qkv_attention_multihead_canvas_pipeline_packed_m8 #(
        .HEADS(HEADS), .ENABLE_GROUP_READ(1), .LUT_FILE(LUT_FILE)
    ) producer (
        .clk(clk), .rst_n(rst_n),
        .block_start(block_start && block_start_ready),
        .block_start_ready(producer_start_ready),
        .metadata_valid(qkv_metadata_valid),
        .metadata_ready(qkv_metadata_ready),
        .parameter_request_valid(qkv_parameter_request_valid),
        .metadata_head(qkv_metadata_head), .metadata_kind(qkv_metadata_kind),
        .metadata_channel_tile(qkv_metadata_channel_tile),
        .metadata_multipliers_packed(qkv_metadata_multipliers_packed),
        .metadata_biases_q12_packed(qkv_metadata_biases_q12_packed),
        .weight_tile_valid(qkv_weight_tile_valid),
        .weight_tile_ready(qkv_weight_tile_ready),
        .weight_head(qkv_weight_head), .weight_kind(qkv_weight_kind),
        .weight_channel_tile(qkv_weight_channel_tile),
        .weight_input_tile(qkv_weight_input_tile),
        .weight_int16_packed(qkv_weight_int16_packed),
        .requested_head(requested_qkv_head),
        .requested_kind(requested_qkv_kind),
        .requested_channel_tile(requested_qkv_channel_tile),
        .requested_valid_channels(),
        .requested_global_row(requested_qkv_global_row),
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
        .canvas_read_valid(canvas_read_valid),
        .canvas_read_head(canvas_read_head),
        .canvas_read_token(canvas_read_token),
        .canvas_read_data_valid(canvas_read_data_valid),
        .canvas_read_data_packed(canvas_read_data),
        .canvas_group_read_valid(canvas_group_read_valid),
        .canvas_group_read_head(canvas_group_read_head),
        .canvas_group_read_group(canvas_group_read_group),
        .canvas_group_read_data_valid(canvas_group_read_data_valid),
        .canvas_group_read_data_packed(canvas_group_read_data),
        .array_request_valid(producer_array_valid),
        .array_request_narrow_int8_mode(producer_array_narrow),
        .array_request_clear(producer_array_clear),
        .array_request_last(producer_array_last),
        .array_request_tag(producer_array_tag),
        .array_request_activations(producer_array_activations),
        .array_request_weights(producer_array_weights),
        .array_request_narrow_activations(
            producer_array_narrow_activations
        ),
        .array_request_narrow_weights(producer_array_narrow_weights),
        .array_response_valid(producer_phase && array_response_valid),
        .array_response_narrow_int8_mode(
            array_response_narrow_int8_mode
        ),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators),
        .array_response_narrow_accumulators(
            array_response_narrow_accumulators
        ),
        .active_head(producer_active_head), .busy(producer_busy),
        .done(producer_done)
    );

    attention_projection_block_pipeline #(
        .OUTPUT_TILES(OUTPUT_TILES), .INTERNAL_MAC(0),
        .ENABLE_RESIDUAL_REPLAY(1), .GROUPED_CANVAS(1)
    ) projection (
        .clk(clk), .rst_n(rst_n),
        .residual_load_valid(residual_load_valid),
        .residual_load_group(residual_load_group),
        .residual_load_output_tile(residual_load_output_tile),
        .residual_load_q10_packed(residual_load_q10_packed),
        .residual_replay_read_valid(residual_replay_read_valid),
        .residual_replay_read_group(residual_replay_read_group),
        .residual_replay_read_output_tile(
            residual_replay_read_output_tile
        ),
        .residual_replay_read_data_valid(residual_replay_read_data_valid),
        .residual_replay_read_q10_packed(
            residual_replay_read_q10_packed
        ),
        .block_start(projection_start),
        .block_start_ready(projection_start_ready),
        .metadata_valid(projection_metadata_valid),
        .metadata_ready(projection_metadata_ready),
        .parameter_request_valid(projection_parameter_request_valid),
        .metadata_output_tile(projection_metadata_output_tile),
        .metadata_multipliers_packed(projection_metadata_multipliers_packed),
        .weight_tile_valid(projection_weight_tile_valid),
        .weight_tile_ready(projection_weight_tile_ready),
        .weight_output_tile(projection_weight_output_tile),
        .weight_input_tile(projection_weight_input_tile),
        .weight_int8_packed(projection_weight_int8_packed),
        .requested_output_tile(requested_projection_output_tile),
        .canvas_read_valid(canvas_read_valid),
        .canvas_read_head(canvas_read_head),
        .canvas_read_token(canvas_read_token),
        .canvas_read_data_valid(canvas_read_data_valid),
        .canvas_read_data_packed(canvas_read_data),
        .canvas_group_read_valid(canvas_group_read_valid),
        .canvas_group_read_head(canvas_group_read_head),
        .canvas_group_read_group(canvas_group_read_group),
        .canvas_group_read_data_valid(canvas_group_read_data_valid),
        .canvas_group_read_data_packed(canvas_group_read_data),
        .block_tile_valid(block_tile_valid),
        .block_tile_ready(block_tile_ready), .block_group(block_group),
        .block_output_tile(block_output_tile),
        .block_q10_packed(block_q10_packed),
        .array_request_valid(projection_array_valid),
        .array_request_clear(projection_array_clear),
        .array_request_last(projection_array_last),
        .array_request_tag(projection_array_tag),
        .array_request_activations(projection_array_activations),
        .array_request_weights(projection_array_weights),
        .array_response_valid(!producer_phase && array_response_valid
            && !array_response_narrow_int8_mode),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators),
        .busy(projection_busy), .done(projection_done)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            projection_start_pending <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && block_start && block_start_ready)
                state <= STATE_PRODUCE;
            if (state == STATE_PRODUCE && producer_done) begin
                state <= STATE_PROJECT;
                projection_start_pending <= 1'b1;
            end else if (projection_start) begin
                projection_start_pending <= 1'b0;
            end
            if (state == STATE_PROJECT && projection_done) begin
                state <= STATE_IDLE;
                done <= 1'b1;
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && producer_array_valid && projection_array_valid)
            $error("packed producer and output projection requested together");
        if (rst_n && !producer_phase && projection_array_valid
            && array_request_narrow_int8_mode)
            $error("attention output projection selected narrow mode");
`endif
    end

endmodule
