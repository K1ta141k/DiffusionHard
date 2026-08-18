`timescale 1ns/1ps

module ddit_block_pipeline #(
    parameter integer HEADS = 12,
    parameter integer ATTENTION_OUTPUT_TILES = 128,
    parameter integer TOKENS = 64,
    parameter integer DOWN_INPUT_SIZE = 3072,
    parameter integer DOWN_OUTPUT_SIZE = 768,
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10,
    parameter integer INTERNAL_NORM1 = 0,
    parameter integer PACKED_ATTENTION = 0,
    parameter integer PHYSICAL_N_LANES = 6,
    parameter integer MLP_M_LANES =
        (TOKENS >= 8 && TOKENS % 8 == 0) ? 8 : 4,
    parameter integer GROUP_WIDTH = ((TOKENS / 4) <= 1)
        ? 1 : $clog2(TOKENS / 4),
    parameter integer MLP_CLIENT_TAG_WIDTH =
        1 + OUTPUT_TILE_TAG_WIDTH + (((TOKENS / MLP_M_LANES) <= 1)
            ? 1 : $clog2(TOKENS / MLP_M_LANES)),
    parameter LUT_FILE = "rtl/tensor_engine/exp_neg_q16_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire block_start,
    output wire block_start_ready,
    output wire busy,
    output reg  done,

    input  wire residual_load_valid,
    input  wire [3:0] residual_load_group,
    input  wire [6:0] residual_load_output_tile,
    input  wire [4*6*24-1:0] residual_load_q10_packed,

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
    output wire attention_tile_valid,
    output wire [3:0] attention_tile_group,
    output wire [6:0] attention_tile_output_tile,
    output wire [4*6*24-1:0] attention_tile_q10_packed,

    input  wire [17:0] smoothing_reciprocal_q15,
    output wire [9:0] smoothing_reciprocal_channel,
    output wire [OUTPUT_TILE_TAG_WIDTH-1:0] requested_up_output_tile,
    output wire requested_up_bank,
    input  wire up_weight_stream_valid,
    output wire up_weight_stream_ready,
    input  wire [6*32*8-1:0] up_weight_stream_data,
    input  wire up_metadata_stream_valid,
    output wire up_metadata_stream_ready,
    input  wire [443:0] up_metadata_stream_data,
    output wire [OUTPUT_TILE_TAG_WIDTH-1:0] requested_down_output_tile,
    output wire requested_down_bank,
    input  wire down_weight_stream_valid,
    output wire down_weight_stream_ready,
    input  wire [6*32*8-1:0] down_weight_stream_data,
    input  wire down_metadata_stream_valid,
    output wire down_metadata_stream_ready,
    input  wire [1343:0] down_metadata_stream_data,

    output wire output_valid,
    output wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile,
    output wire [GROUP_WIDTH-1:0] output_group,
    output wire [4*6*24-1:0] outputs_packed,
    output wire attention_busy,
    output wire mlp_busy
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_NORM1 = 2'd1;
    localparam [1:0] STATE_ATTENTION = 2'd2;
    localparam [1:0] STATE_MLP = 2'd3;
    localparam integer MLP_ARRAY_TAG_WIDTH = MLP_CLIENT_TAG_WIDTH + 1;

    reg [1:0] state;
    reg attention_start_pending;
    reg mlp_start_pending;
    wire attention_start_ready;
    wire attention_done;
    wire mlp_start_ready;
    wire mlp_done;
    wire mlp_phase = state == STATE_MLP;

    wire residual_replay_read_valid;
    wire [3:0] residual_replay_read_group;
    wire [6:0] residual_replay_read_tile;
    wire residual_replay_data_valid;
    wire [4*6*24-1:0] residual_replay_data;
    wire norm1_start_ready,norm1_busy,norm1_done;
    wire norm1_canvas_read_valid;wire [3:0] norm1_canvas_read_group;
    wire [6:0] norm1_canvas_read_tile;wire norm1_canvas_data_valid;
    wire norm1_normalized_data_valid;wire [4*32*18-1:0] norm1_normalized_data;
    wire mlp_canvas_read_valid;wire [3:0] mlp_canvas_read_group;
    wire [6:0] mlp_canvas_read_tile;wire mlp_canvas_data_valid;

    wire attention_array_valid;
    wire attention_array_narrow;
    wire attention_array_clear;
    wire attention_array_last;
    wire [7:0] attention_array_tag;
    wire [4*32*18-1:0] attention_array_activations;
    wire [6*32*18-1:0] attention_array_weights;
    wire [8*32*8-1:0] attention_array_narrow_activations;
    wire [6*32*8-1:0] attention_array_narrow_weights;
    wire attention_array_response_valid;
    wire attention_array_response_narrow;
    wire [7:0] attention_array_response_tag;
    wire [4*6*48-1:0] attention_array_response_accumulators;
    wire [8*6*32-1:0] attention_array_response_narrow_accumulators;
    wire mlp_array_valid;
    wire mlp_array_clear;
    wire mlp_array_last;
    wire [MLP_ARRAY_TAG_WIDTH-1:0] mlp_array_tag;
    wire [8*32*8-1:0] mlp_array_activations;
    wire [6*32*8-1:0] mlp_array_weights;
    wire mlp_array_response_valid;
    wire [MLP_ARRAY_TAG_WIDTH-1:0] mlp_array_response_tag;
    wire [8*6*32-1:0] mlp_array_response_accumulators;

    assign block_start_ready = state == STATE_IDLE
        && (INTERNAL_NORM1 ? norm1_start_ready : attention_start_ready);
    assign busy = state != STATE_IDLE;
    assign residual_replay_read_valid = state == STATE_NORM1
        ? norm1_canvas_read_valid : mlp_canvas_read_valid;
    assign residual_replay_read_group = state == STATE_NORM1
        ? norm1_canvas_read_group : mlp_canvas_read_group;
    assign residual_replay_read_tile = state == STATE_NORM1
        ? norm1_canvas_read_tile : mlp_canvas_read_tile;
    assign norm1_canvas_data_valid = state == STATE_NORM1
        && residual_replay_data_valid;
    assign mlp_canvas_data_valid = state == STATE_MLP
        && residual_replay_data_valid;

    hidden_canvas_norm1_precompute norm1(
        .clk(clk),.rst_n(rst_n),
        .start(INTERNAL_NORM1 && state==STATE_IDLE && block_start
            && block_start_ready),.start_ready(norm1_start_ready),
        .canvas_read_valid(norm1_canvas_read_valid),
        .canvas_read_group(norm1_canvas_read_group),
        .canvas_read_output_tile(norm1_canvas_read_tile),
        .canvas_read_data_valid(norm1_canvas_data_valid),
        .canvas_read_q10_packed(residual_replay_data),
        .normalized_read_valid(normalized_read_valid),
        .normalized_read_group(normalized_read_group),
        .normalized_read_input_tile(normalized_read_input_tile),
        .normalized_read_data_valid(norm1_normalized_data_valid),
        .normalized_q12_packed(norm1_normalized_data),
        .busy(norm1_busy),.done(norm1_done));

    generate
        if (PACKED_ATTENTION) begin : packed_attention_path
            qkv_attention_projection_block_pipeline_packed_m8 #(
                .HEADS(HEADS), .OUTPUT_TILES(ATTENTION_OUTPUT_TILES),
                .LUT_FILE(LUT_FILE)
            ) attention (
                .clk(clk), .rst_n(rst_n),
                .block_start(state == STATE_ATTENTION
                    && attention_start_pending),
                .block_start_ready(attention_start_ready),
                .residual_load_valid(residual_load_valid),
                .residual_load_group(residual_load_group),
                .residual_load_output_tile(residual_load_output_tile),
                .residual_load_q10_packed(residual_load_q10_packed),
                .residual_replay_read_valid(residual_replay_read_valid),
                .residual_replay_read_group(residual_replay_read_group),
                .residual_replay_read_output_tile(residual_replay_read_tile),
                .residual_replay_read_data_valid(residual_replay_data_valid),
                .residual_replay_read_q10_packed(residual_replay_data),
                .qkv_metadata_valid(qkv_metadata_valid),
                .qkv_metadata_ready(qkv_metadata_ready),
                .qkv_parameter_request_valid(qkv_parameter_request_valid),
                .qkv_metadata_head(qkv_metadata_head),
                .qkv_metadata_kind(qkv_metadata_kind),
                .qkv_metadata_channel_tile(qkv_metadata_channel_tile),
                .qkv_metadata_multipliers_packed(
                    qkv_metadata_multipliers_packed
                ),
                .qkv_metadata_biases_q12_packed(
                    qkv_metadata_biases_q12_packed
                ),
                .qkv_weight_tile_valid(qkv_weight_tile_valid),
                .qkv_weight_tile_ready(qkv_weight_tile_ready),
                .qkv_weight_head(qkv_weight_head),
                .qkv_weight_kind(qkv_weight_kind),
                .qkv_weight_channel_tile(qkv_weight_channel_tile),
                .qkv_weight_input_tile(qkv_weight_input_tile),
                .qkv_weight_int16_packed(qkv_weight_int16_packed),
                .requested_qkv_head(requested_qkv_head),
                .requested_qkv_kind(requested_qkv_kind),
                .requested_qkv_channel_tile(requested_qkv_channel_tile),
                .requested_qkv_global_row(requested_qkv_global_row),
                .normalized_read_valid(normalized_read_valid),
                .normalized_read_group(normalized_read_group),
                .normalized_read_input_tile(normalized_read_input_tile),
                .normalized_read_data_valid(INTERNAL_NORM1
                    ? norm1_normalized_data_valid
                    : normalized_read_data_valid),
                .normalized_q12_packed(INTERNAL_NORM1
                    ? norm1_normalized_data : normalized_q12_packed),
                .constant_load_valid(constant_load_valid),
                .constant_load_token(constant_load_token),
                .constant_load_pair(constant_load_pair),
                .constant_load_cosine_q15(constant_load_cosine_q15),
                .constant_load_sine_q15(constant_load_sine_q15),
                .projection_metadata_valid(projection_metadata_valid),
                .projection_metadata_ready(projection_metadata_ready),
                .projection_parameter_request_valid(
                    projection_parameter_request_valid
                ),
                .projection_metadata_output_tile(
                    projection_metadata_output_tile
                ),
                .projection_metadata_multipliers_packed(
                    projection_metadata_multipliers_packed
                ),
                .projection_weight_tile_valid(
                    projection_weight_tile_valid
                ),
                .projection_weight_tile_ready(
                    projection_weight_tile_ready
                ),
                .projection_weight_output_tile(
                    projection_weight_output_tile
                ),
                .projection_weight_input_tile(projection_weight_input_tile),
                .projection_weight_int8_packed(
                    projection_weight_int8_packed
                ),
                .requested_projection_output_tile(
                    requested_projection_output_tile
                ),
                .block_tile_valid(attention_tile_valid),
                .block_tile_ready(1'b1),
                .block_group(attention_tile_group),
                .block_output_tile(attention_tile_output_tile),
                .block_q10_packed(attention_tile_q10_packed),
                .array_request_valid(attention_array_valid),
                .array_request_narrow_int8_mode(attention_array_narrow),
                .array_request_clear(attention_array_clear),
                .array_request_last(attention_array_last),
                .array_request_tag(attention_array_tag),
                .array_request_activations(attention_array_activations),
                .array_request_weights(attention_array_weights),
                .array_request_narrow_activations(
                    attention_array_narrow_activations
                ),
                .array_request_narrow_weights(
                    attention_array_narrow_weights
                ),
                .array_response_valid(attention_array_response_valid),
                .array_response_narrow_int8_mode(
                    attention_array_response_narrow
                ),
                .array_response_tag(attention_array_response_tag),
                .array_response_accumulators(
                    attention_array_response_accumulators
                ),
                .array_response_narrow_accumulators(
                    attention_array_response_narrow_accumulators
                ),
                .producer_busy(), .projection_busy(),
                .busy(attention_busy), .done(attention_done)
            );
        end else begin : fixed_attention_path
    qkv_attention_projection_block_pipeline #(
        .HEADS(HEADS), .OUTPUT_TILES(ATTENTION_OUTPUT_TILES),
        .LUT_FILE(LUT_FILE)
    ) attention (
        .clk(clk), .rst_n(rst_n),
        .block_start(state == STATE_ATTENTION && attention_start_pending),
        .block_start_ready(attention_start_ready),
        .residual_load_valid(residual_load_valid),
        .residual_load_group(residual_load_group),
        .residual_load_output_tile(residual_load_output_tile),
        .residual_load_q10_packed(residual_load_q10_packed),
        .residual_replay_read_valid(residual_replay_read_valid),
        .residual_replay_read_group(residual_replay_read_group),
        .residual_replay_read_output_tile(residual_replay_read_tile),
        .residual_replay_read_data_valid(residual_replay_data_valid),
        .residual_replay_read_q10_packed(residual_replay_data),
        .qkv_metadata_valid(qkv_metadata_valid),
        .qkv_metadata_ready(qkv_metadata_ready),
        .qkv_parameter_request_valid(qkv_parameter_request_valid),
        .qkv_metadata_head(qkv_metadata_head),
        .qkv_metadata_kind(qkv_metadata_kind),
        .qkv_metadata_channel_tile(qkv_metadata_channel_tile),
        .qkv_metadata_multipliers_packed(qkv_metadata_multipliers_packed),
        .qkv_metadata_biases_q12_packed(qkv_metadata_biases_q12_packed),
        .qkv_weight_tile_valid(qkv_weight_tile_valid),
        .qkv_weight_tile_ready(qkv_weight_tile_ready),
        .qkv_weight_head(qkv_weight_head), .qkv_weight_kind(qkv_weight_kind),
        .qkv_weight_channel_tile(qkv_weight_channel_tile),
        .qkv_weight_input_tile(qkv_weight_input_tile),
        .qkv_weight_int16_packed(qkv_weight_int16_packed),
        .requested_qkv_head(requested_qkv_head),
        .requested_qkv_kind(requested_qkv_kind),
        .requested_qkv_channel_tile(requested_qkv_channel_tile),
        .requested_qkv_global_row(requested_qkv_global_row),
        .normalized_read_valid(normalized_read_valid),
        .normalized_read_group(normalized_read_group),
        .normalized_read_input_tile(normalized_read_input_tile),
        .normalized_read_data_valid(INTERNAL_NORM1
            ? norm1_normalized_data_valid : normalized_read_data_valid),
        .normalized_q12_packed(INTERNAL_NORM1
            ? norm1_normalized_data : normalized_q12_packed),
        .constant_load_valid(constant_load_valid),
        .constant_load_token(constant_load_token),
        .constant_load_pair(constant_load_pair),
        .constant_load_cosine_q15(constant_load_cosine_q15),
        .constant_load_sine_q15(constant_load_sine_q15),
        .projection_metadata_valid(projection_metadata_valid),
        .projection_metadata_ready(projection_metadata_ready),
        .projection_parameter_request_valid(
            projection_parameter_request_valid
        ),
        .projection_metadata_output_tile(projection_metadata_output_tile),
        .projection_metadata_multipliers_packed(
            projection_metadata_multipliers_packed
        ), .projection_weight_tile_valid(projection_weight_tile_valid),
        .projection_weight_tile_ready(projection_weight_tile_ready),
        .projection_weight_output_tile(projection_weight_output_tile),
        .projection_weight_input_tile(projection_weight_input_tile),
        .projection_weight_int8_packed(projection_weight_int8_packed),
        .requested_projection_output_tile(requested_projection_output_tile),
        .block_tile_valid(attention_tile_valid), .block_tile_ready(1'b1),
        .block_group(attention_tile_group),
        .block_output_tile(attention_tile_output_tile),
        .block_q10_packed(attention_tile_q10_packed),
        .array_request_valid(attention_array_valid),
        .array_request_clear(attention_array_clear),
        .array_request_last(attention_array_last),
        .array_request_tag(attention_array_tag),
        .array_request_activations(attention_array_activations),
        .array_request_weights(attention_array_weights),
        .array_response_valid(attention_array_response_valid),
        .array_response_tag(attention_array_response_tag),
        .array_response_accumulators(attention_array_response_accumulators),
        .producer_busy(), .projection_busy(), .busy(attention_busy),
        .done(attention_done)
    );
            assign attention_array_narrow = 1'b0;
            assign attention_array_narrow_activations = 0;
            assign attention_array_narrow_weights = 0;
        end
    endgenerate

    hidden_canvas_automatic_mlp_block #(
        .TOKENS(TOKENS), .DOWN_INPUT_SIZE(DOWN_INPUT_SIZE),
        .DOWN_OUTPUT_SIZE(DOWN_OUTPUT_SIZE),
        .MLP_M_LANES(MLP_M_LANES),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .GROUP_WIDTH(GROUP_WIDTH),
        .CLIENT_TAG_WIDTH(MLP_CLIENT_TAG_WIDTH)
    ) mlp (
        .clk(clk), .rst_n(rst_n),
        .block_start(mlp_phase && mlp_start_pending),
        .block_start_ready(mlp_start_ready),
        .smoothing_reciprocal_q15(smoothing_reciprocal_q15),
        .smoothing_reciprocal_channel(smoothing_reciprocal_channel),
        .busy(mlp_busy), .done(mlp_done),
        .canvas_read_valid(mlp_canvas_read_valid),
        .canvas_read_group(mlp_canvas_read_group),
        .canvas_read_output_tile(mlp_canvas_read_tile),
        .canvas_read_data_valid(mlp_canvas_data_valid),
        .canvas_read_q10_packed(residual_replay_data),
        .requested_up_output_tile(requested_up_output_tile),
        .requested_up_bank(requested_up_bank),
        .up_weight_stream_valid(up_weight_stream_valid),
        .up_weight_stream_ready(up_weight_stream_ready),
        .up_weight_stream_data(up_weight_stream_data),
        .up_metadata_stream_valid(up_metadata_stream_valid),
        .up_metadata_stream_ready(up_metadata_stream_ready),
        .up_metadata_stream_data(up_metadata_stream_data),
        .requested_down_output_tile(requested_down_output_tile),
        .requested_down_bank(requested_down_bank),
        .down_weight_stream_valid(down_weight_stream_valid),
        .down_weight_stream_ready(down_weight_stream_ready),
        .down_weight_stream_data(down_weight_stream_data),
        .down_metadata_stream_valid(down_metadata_stream_valid),
        .down_metadata_stream_ready(down_metadata_stream_ready),
        .down_metadata_stream_data(down_metadata_stream_data),
        .output_valid(output_valid), .output_tile(output_tile),
        .output_group(output_group), .outputs_packed(outputs_packed),
        .array_request_valid(mlp_array_valid),
        .array_request_clear(mlp_array_clear),
        .array_request_last(mlp_array_last),
        .array_request_tag(mlp_array_tag),
        .array_request_activations(mlp_array_activations),
        .array_request_weights(mlp_array_weights),
        .array_response_valid(mlp_array_response_valid),
        .array_response_tag(mlp_array_response_tag),
        .array_response_accumulators(mlp_array_response_accumulators)
    );

    ddit_block_shared_mac_packed_m8 #(
        .MLP_TAG_WIDTH(MLP_ARRAY_TAG_WIDTH),
        .ATTENTION_PACKED(PACKED_ATTENTION),
        .PHYSICAL_N_LANES(PHYSICAL_N_LANES)
    ) shared_mac (
        .clk(clk), .rst_n(rst_n), .mlp_phase(mlp_phase),
        .attention_request_valid(attention_array_valid),
        .attention_request_narrow_int8_mode(attention_array_narrow),
        .attention_request_clear(attention_array_clear),
        .attention_request_last(attention_array_last),
        .attention_request_tag(attention_array_tag),
        .attention_request_activations(attention_array_activations),
        .attention_request_weights(attention_array_weights),
        .attention_request_narrow_activations(
            attention_array_narrow_activations
        ),
        .attention_request_narrow_weights(
            attention_array_narrow_weights
        ),
        .attention_response_valid(attention_array_response_valid),
        .attention_response_narrow_int8_mode(
            attention_array_response_narrow
        ),
        .attention_response_tag(attention_array_response_tag),
        .attention_response_accumulators(
            attention_array_response_accumulators
        ),
        .attention_response_narrow_accumulators(
            attention_array_response_narrow_accumulators
        ), .mlp_request_valid(mlp_array_valid),
        .mlp_request_clear(mlp_array_clear),
        .mlp_request_last(mlp_array_last), .mlp_request_tag(mlp_array_tag),
        .mlp_request_activations(mlp_array_activations),
        .mlp_request_weights(mlp_array_weights),
        .mlp_response_valid(mlp_array_response_valid),
        .mlp_response_tag(mlp_array_response_tag),
        .mlp_response_accumulators(mlp_array_response_accumulators)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            attention_start_pending <= 1'b0;
            mlp_start_pending <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && block_start && block_start_ready) begin
                if (INTERNAL_NORM1)
                    state <= STATE_NORM1;
                else begin
                    state <= STATE_ATTENTION;
                    attention_start_pending <= 1'b1;
                end
            end
            if (state == STATE_NORM1 && norm1_done) begin
                state <= STATE_ATTENTION;
                attention_start_pending <= 1'b1;
            end else if (state == STATE_ATTENTION && attention_start_pending
                         && attention_start_ready) begin
                attention_start_pending <= 1'b0;
            end
            if (state == STATE_ATTENTION && attention_done) begin
                state <= STATE_MLP;
                mlp_start_pending <= 1'b1;
            end else if (state == STATE_MLP && mlp_start_pending
                         && mlp_start_ready) begin
                mlp_start_pending <= 1'b0;
            end
            if (state == STATE_MLP && mlp_done) begin
                state <= STATE_IDLE;
                done <= 1'b1;
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && attention_array_valid && mlp_array_valid)
            $error("DDiT attention and MLP array phases overlapped");
`endif
    end

    initial begin
        if (MLP_M_LANES != 4 && MLP_M_LANES != 8)
            $error("DDiT block supports four or eight MLP token lanes");
        if (TOKENS % MLP_M_LANES != 0)
            $error("DDiT token count must align to the MLP token lanes");
    end

endmodule
