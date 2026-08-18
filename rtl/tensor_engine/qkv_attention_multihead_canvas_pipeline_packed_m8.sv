`timescale 1ns/1ps

module qkv_attention_multihead_canvas_pipeline_packed_m8 #(
    parameter integer HEADS = 12,
    parameter integer ENABLE_GROUP_READ = 0,
    parameter integer ARRAY_BACKPRESSURE = 0,
    parameter LUT_FILE = "rtl/tensor_engine/exp_neg_q16_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire block_start,
    output wire block_start_ready,
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
    input  wire canvas_read_valid,
    input  wire [3:0] canvas_read_head,
    input  wire [5:0] canvas_read_token,
    output wire canvas_read_data_valid,
    output wire [64*18-1:0] canvas_read_data_packed,
    input  wire canvas_group_read_valid,
    input  wire [3:0] canvas_group_read_head,
    input  wire [3:0] canvas_group_read_group,
    output wire canvas_group_read_data_valid,
    output wire [4*64*18-1:0] canvas_group_read_data_packed,
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
    output wire [3:0] active_head,
    output wire busy,
    output reg  done
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_HEAD = 2'd1;
    localparam [1:0] STATE_DRAIN = 2'd2;

    reg [1:0] state;
    reg [3:0] head;
    reg head_start_pending;
    wire head_start_ready;
    wire head_start = state == STATE_HEAD && head_start_pending
        && head_start_ready;
    wire head_tile_valid;
    wire head_tile_ready;
    wire [3:0] head_tile_group;
    wire [3:0] head_output_tile;
    wire [2:0] head_valid_channels;
    wire [4*6*18-1:0] head_data;
    wire head_staging_busy;
    wire head_attention_busy;
    wire head_busy;
    wire head_done;
    wire canvas_tile_done;

    assign block_start_ready = state == STATE_IDLE && head_start_ready;
    assign active_head = head;
    assign busy = state != STATE_IDLE;

    qkv_attention_head_pipeline_packed_m8 #(
        .INTERNAL_MAC(0), .ARRAY_BACKPRESSURE(ARRAY_BACKPRESSURE),
        .LUT_FILE(LUT_FILE)
    ) head_pipeline (
        .clk(clk), .rst_n(rst_n), .start(head_start),
        .start_ready(head_start_ready), .head_in(head),
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
        .attention_tile_valid(head_tile_valid),
        .attention_tile_ready(head_tile_ready),
        .attention_group(head_tile_group),
        .attention_output_tile(head_output_tile),
        .attention_valid_channels(head_valid_channels),
        .attention_q12_packed(head_data),
        .array_request_valid(array_request_valid),
        .array_request_ready(array_request_ready),
        .array_request_narrow_int8_mode(
            array_request_narrow_int8_mode
        ),
        .array_request_clear(array_request_clear),
        .array_request_last(array_request_last),
        .array_request_tag(array_request_tag),
        .array_request_activations(array_request_activations),
        .array_request_weights(array_request_weights),
        .array_request_narrow_activations(
            array_request_narrow_activations
        ),
        .array_request_narrow_weights(array_request_narrow_weights),
        .array_response_valid(array_response_valid),
        .array_response_narrow_int8_mode(
            array_response_narrow_int8_mode
        ),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators),
        .array_response_narrow_accumulators(
            array_response_narrow_accumulators
        ),
        .staging_busy(head_staging_busy),
        .attention_busy(head_attention_busy), .busy(head_busy),
        .done(head_done)
    );

    attention_canvas_grouped_scratchpad_banked canvas (
        .clk(clk), .rst_n(rst_n), .tile_valid(head_tile_valid),
        .tile_ready(head_tile_ready), .tile_head(head),
        .tile_group(head_tile_group), .tile_channel_tile(head_output_tile),
        .tile_valid_channels(head_valid_channels),
        .tile_data_packed(head_data), .tile_done(canvas_tile_done),
        .read_valid(canvas_read_valid), .read_head(canvas_read_head),
        .read_token(canvas_read_token),
        .read_data_valid(canvas_read_data_valid),
        .read_data_packed(canvas_read_data_packed),
        .group_read_valid(ENABLE_GROUP_READ ? canvas_group_read_valid : 1'b0),
        .group_read_head(canvas_group_read_head),
        .group_read_group(canvas_group_read_group),
        .group_read_data_valid(canvas_group_read_data_valid),
        .group_read_data_packed(canvas_group_read_data_packed)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            head <= 0;
            head_start_pending <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && block_start && block_start_ready) begin
                state <= STATE_HEAD;
                head <= 0;
                head_start_pending <= 1'b1;
            end else if (head_start) begin
                head_start_pending <= 1'b0;
            end
            if (state == STATE_HEAD && head_done) begin
                if (head == HEADS-1) begin
                    state <= STATE_DRAIN;
                end else begin
                    head <= head + 1'b1;
                    head_start_pending <= 1'b1;
                end
            end else if (state == STATE_DRAIN && head_tile_ready) begin
                state <= STATE_IDLE;
                done <= 1'b1;
            end
        end
    end

    initial begin
        if (HEADS < 1 || HEADS > 12)
            $error("packed QKV attention canvas supports one through twelve heads");
    end

endmodule
