`timescale 1ns/1ps

module attention_block_shared_mac_pipeline #(
    parameter integer HEADS = 12,
    parameter integer PROJECTION_OUTPUT_TILES = 128,
    parameter LUT_FILE = "rtl/tensor_engine/exp_neg_q16_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output wire start_ready,
    input  wire qkv_load_valid,
    output wire qkv_load_ready,
    input  wire [3:0] qkv_load_head,
    input  wire [5:0] qkv_load_token,
    input  wire [5:0] qkv_load_channel,
    input  wire signed [17:0] qkv_load_query_q12,
    input  wire signed [17:0] qkv_load_key_q12,
    input  wire signed [17:0] qkv_load_value_q12,
    output wire [3:0] expected_head,
    input  wire residual_load_valid,
    input  wire [3:0] residual_load_group,
    input  wire [6:0] residual_load_output_tile,
    input  wire [4*6*24-1:0] residual_load_q10_packed,
    input  wire projection_metadata_valid,
    output wire projection_metadata_ready,
    input  wire [6:0] projection_metadata_output_tile,
    input  wire [6*24-1:0] projection_multipliers_packed,
    input  wire projection_weight_valid,
    output wire projection_weight_ready,
    input  wire [6:0] projection_weight_output_tile,
    input  wire [4:0] projection_weight_input_tile,
    input  wire [6*32*8-1:0] projection_weight_int8_packed,
    output wire [6:0] requested_projection_output_tile,
    output wire block_tile_valid,
    input  wire block_tile_ready,
    output wire [3:0] block_group,
    output wire [6:0] block_output_tile,
    output wire [4*6*24-1:0] block_q10_packed,
    output wire busy,
    output reg  done
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_ATTENTION = 2'd1;
    localparam [1:0] STATE_PROJECTION = 2'd2;

    reg [1:0] state;
    reg projection_start_pending;

    wire attention_start_ready;
    wire attention_busy;
    wire attention_done;
    wire projection_start_ready;
    wire projection_busy;
    wire projection_done;
    wire projection_start = (state == STATE_PROJECTION)
        && projection_start_pending && projection_start_ready;

    wire canvas_read_valid;
    wire [3:0] canvas_read_head;
    wire [5:0] canvas_read_token;
    wire canvas_read_data_valid;
    wire [64*18-1:0] canvas_read_data;

    wire attention_array_valid;
    wire attention_array_clear;
    wire attention_array_last;
    wire [3:0] attention_array_tag;
    wire [4*32*18-1:0] attention_array_activations;
    wire [6*32*18-1:0] attention_array_weights;
    wire projection_array_valid;
    wire projection_array_clear;
    wire projection_array_last;
    wire [7:0] projection_array_tag;
    wire [4*32*18-1:0] projection_array_activations;
    wire [6*32*18-1:0] projection_array_weights;

    wire select_projection = (state == STATE_PROJECTION);
    wire shared_array_valid = select_projection
        ? projection_array_valid : attention_array_valid;
    wire shared_array_clear = select_projection
        ? projection_array_clear : attention_array_clear;
    wire shared_array_last = select_projection
        ? projection_array_last : attention_array_last;
    wire [7:0] shared_array_tag = select_projection
        ? projection_array_tag : {4'b0, attention_array_tag};
    wire [4*32*18-1:0] shared_array_activations = select_projection
        ? projection_array_activations : attention_array_activations;
    wire [6*32*18-1:0] shared_array_weights = select_projection
        ? projection_array_weights : attention_array_weights;
    wire shared_array_response_valid;
    wire [7:0] shared_array_response_tag;
    wire [4*6*48-1:0] shared_array_response_accumulators;

    assign start_ready = (state == STATE_IDLE) && attention_start_ready;
    assign busy = (state != STATE_IDLE);

    attention_multihead_canvas_pipeline #(
        .HEADS(HEADS), .INTERNAL_MAC(0), .LUT_FILE(LUT_FILE)
    ) attention (
        .clk(clk), .rst_n(rst_n), .block_start(start && start_ready),
        .block_start_ready(attention_start_ready),
        .load_valid(qkv_load_valid), .load_ready(qkv_load_ready),
        .load_head(qkv_load_head), .load_token(qkv_load_token),
        .load_channel(qkv_load_channel),
        .load_query_q12(qkv_load_query_q12),
        .load_key_q12(qkv_load_key_q12),
        .load_value_q12(qkv_load_value_q12), .expected_head(expected_head),
        .canvas_read_valid(canvas_read_valid),
        .canvas_read_head(canvas_read_head),
        .canvas_read_token(canvas_read_token),
        .canvas_read_data_valid(canvas_read_data_valid),
        .canvas_read_data_packed(canvas_read_data),
        .array_request_valid(attention_array_valid),
        .array_request_clear(attention_array_clear),
        .array_request_last(attention_array_last),
        .array_request_tag(attention_array_tag),
        .array_request_activations(attention_array_activations),
        .array_request_weights(attention_array_weights),
        .array_response_valid(
            shared_array_response_valid && !select_projection
        ),
        .array_response_tag(shared_array_response_tag[3:0]),
        .array_response_accumulators(shared_array_response_accumulators),
        .busy(attention_busy), .done(attention_done)
    );

    attention_projection_block_pipeline #(
        .OUTPUT_TILES(PROJECTION_OUTPUT_TILES), .INTERNAL_MAC(0)
    ) projection (
        .clk(clk), .rst_n(rst_n),
        .residual_load_valid(residual_load_valid),
        .residual_load_group(residual_load_group),
        .residual_load_output_tile(residual_load_output_tile),
        .residual_load_q10_packed(residual_load_q10_packed),
        .block_start(projection_start),
        .block_start_ready(projection_start_ready),
        .metadata_valid(projection_metadata_valid),
        .metadata_ready(projection_metadata_ready),
        .metadata_output_tile(projection_metadata_output_tile),
        .metadata_multipliers_packed(projection_multipliers_packed),
        .weight_tile_valid(projection_weight_valid),
        .weight_tile_ready(projection_weight_ready),
        .weight_output_tile(projection_weight_output_tile),
        .weight_input_tile(projection_weight_input_tile),
        .weight_int8_packed(projection_weight_int8_packed),
        .requested_output_tile(requested_projection_output_tile),
        .canvas_read_valid(canvas_read_valid),
        .canvas_read_head(canvas_read_head),
        .canvas_read_token(canvas_read_token),
        .canvas_read_data_valid(canvas_read_data_valid),
        .canvas_read_data_packed(
            (canvas_read_head < HEADS) ? canvas_read_data : {64*18{1'b0}}
        ),
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
        .array_response_valid(
            shared_array_response_valid && select_projection
        ),
        .array_response_tag(shared_array_response_tag),
        .array_response_accumulators(shared_array_response_accumulators),
        .busy(projection_busy), .done(projection_done)
    );

    mixed_precision_mac_tile_pipelined #(
        .M_LANES(4), .N_LANES(6), .STORAGE_WIDTH(18),
        .ACC_WIDTH(48), .TAG_WIDTH(8)
    ) shared_array (
        .clk(clk), .rst_n(rst_n), .valid_in(shared_array_valid),
        .narrow_int8_mode(1'b0), .clear_accumulators(shared_array_clear),
        .last_k_tile(shared_array_last), .tag_in(shared_array_tag),
        .activations_packed(shared_array_activations),
        .weights_packed(shared_array_weights),
        .valid_out(shared_array_response_valid),
        .tag_out(shared_array_response_tag),
        .accumulators_packed(shared_array_response_accumulators)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            projection_start_pending <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && start && start_ready) begin
                state <= STATE_ATTENTION;
            end else if (state == STATE_ATTENTION && attention_done) begin
                state <= STATE_PROJECTION;
                projection_start_pending <= 1'b1;
            end else if (state == STATE_PROJECTION && projection_start) begin
                projection_start_pending <= 1'b0;
            end
            if (state == STATE_PROJECTION && projection_done) begin
                state <= STATE_IDLE;
                done <= 1'b1;
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && attention_array_valid && projection_array_valid)
            $error("attention and projection requested the shared array together");
`endif
    end

endmodule
