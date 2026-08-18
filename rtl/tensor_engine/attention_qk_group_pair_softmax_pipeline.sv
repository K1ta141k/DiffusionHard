`timescale 1ns/1ps

module attention_qk_group_pair_softmax_pipeline #(
    parameter integer N_LANES = 6,
    parameter integer INTERNAL_MAC = 1,
    parameter integer ARRAY_BACKPRESSURE = 0,
    parameter integer TAG_WIDTH = 6,
    parameter LUT_FILE = "rtl/tensor_engine/exp_neg_q16_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire scale_load_valid,
    input  wire [5:0] scale_load_token,
    input  wire [17:0] scale_load_query_maximum,
    input  wire [17:0] scale_load_key_maximum,
    input  wire [23:0] scale_load_query_multiplier_q17,
    input  wire [23:0] scale_load_key_multiplier_q17,
    input  wire start,
    input  wire [2:0] group_pair_in,
    output wire start_ready,
    output wire query_read_valid,
    output wire [5:0] query_read_token,
    output wire [1:0] query_read_channel_block,
    input  wire query_data_valid,
    input  wire [16*18-1:0] query_data_q12_packed,
    output wire key_read_valid,
    output wire [5:0] key_read_token,
    output wire [1:0] key_read_channel_block,
    input  wire key_data_valid,
    input  wire [16*18-1:0] key_data_q12_packed,
    output wire probability_group_valid,
    input  wire probability_group_ready,
    output wire [3:0] probability_group,
    output wire [4*64*16-1:0] probabilities_q16_packed,
    output wire mac_request_valid,
    input  wire mac_request_ready,
    output wire mac_request_clear,
    output wire mac_request_last,
    output wire [TAG_WIDTH-1:0] mac_request_tag,
    output wire [8*32*8-1:0] mac_request_activations_int8,
    output wire [N_LANES*32*8-1:0] mac_request_weights_int8,
    input  wire mac_response_valid,
    input  wire [TAG_WIDTH-1:0] mac_response_tag,
    input  wire [8*N_LANES*32-1:0] mac_response_accumulators,
    output wire busy,
    output reg  done
);

    wire pair_score_valid;
    wire pair_score_ready;
    wire [2:0] pair_score_group;
    wire [5:0] pair_score_key_tile;
    wire [2:0] pair_score_valid_keys;
    wire [8*N_LANES*18-1:0] pair_scores;
    wire qk_busy;
    wire qk_done;
    wire score_tile_valid;
    wire score_tile_ready;
    wire [3:0] score_group;
    wire [5:0] score_key_tile;
    wire [2:0] score_valid_keys;
    wire [4*N_LANES*18-1:0] scores;
    wire pair_buffer_busy;
    wire pair_buffer_done;
    wire softmax_busy;
    wire softmax_done;
    reg active;
    reg probability_half;

    assign start_ready = !active;
    assign busy = active || qk_busy || pair_buffer_busy || softmax_busy;

    attention_qk_group_pair_scheduler_packed_m8 #(
        .N_LANES(N_LANES), .INTERNAL_MAC(INTERNAL_MAC),
        .ARRAY_BACKPRESSURE(ARRAY_BACKPRESSURE),
        .TAG_WIDTH(TAG_WIDTH)
    ) qk_scheduler (
        .clk(clk), .rst_n(rst_n),
        .scale_load_valid(scale_load_valid),
        .scale_load_token(scale_load_token),
        .scale_load_query_maximum(scale_load_query_maximum),
        .scale_load_key_maximum(scale_load_key_maximum),
        .scale_load_query_multiplier_q17(scale_load_query_multiplier_q17),
        .scale_load_key_multiplier_q17(scale_load_key_multiplier_q17),
        .start(start && start_ready), .group_pair_in(group_pair_in),
        .start_ready(), .query_read_valid(query_read_valid),
        .query_read_token(query_read_token),
        .query_read_channel_block(query_read_channel_block),
        .query_data_valid(query_data_valid),
        .query_data_q12_packed(query_data_q12_packed),
        .key_read_valid(key_read_valid), .key_read_token(key_read_token),
        .key_read_channel_block(key_read_channel_block),
        .key_data_valid(key_data_valid),
        .key_data_q12_packed(key_data_q12_packed),
        .score_pair_valid(pair_score_valid),
        .score_pair_ready(pair_score_ready),
        .score_group_pair(pair_score_group),
        .score_key_tile(pair_score_key_tile),
        .score_valid_keys(pair_score_valid_keys),
        .scores_q10_packed(pair_scores),
        .mac_request_valid(mac_request_valid),
        .mac_request_ready(mac_request_ready),
        .mac_request_clear(mac_request_clear),
        .mac_request_last(mac_request_last),
        .mac_request_tag(mac_request_tag),
        .mac_request_activations_int8(mac_request_activations_int8),
        .mac_request_weights_int8(mac_request_weights_int8),
        .mac_response_valid(mac_response_valid),
        .mac_response_tag(mac_response_tag),
        .mac_response_accumulators(mac_response_accumulators),
        .busy(qk_busy), .done(qk_done)
    );

    attention_score_group_pair_buffer #(
        .N_LANES(N_LANES)
    ) pair_buffer (
        .clk(clk), .rst_n(rst_n), .pair_tile_valid(pair_score_valid),
        .pair_tile_ready(pair_score_ready), .pair_group(pair_score_group),
        .pair_key_tile(pair_score_key_tile),
        .pair_valid_keys(pair_score_valid_keys),
        .pair_scores_q10_packed(pair_scores),
        .score_tile_valid(score_tile_valid),
        .score_tile_ready(score_tile_ready), .score_group(score_group),
        .score_key_tile(score_key_tile),
        .score_valid_keys(score_valid_keys), .scores_q10_packed(scores),
        .busy(pair_buffer_busy), .done(pair_buffer_done)
    );

    attention_score_group_softmax_stream #(
        .N_LANES(N_LANES), .LUT_FILE(LUT_FILE)
    ) softmax (
        .clk(clk), .rst_n(rst_n), .score_tile_valid(score_tile_valid),
        .score_tile_ready(score_tile_ready), .score_tile_group(score_group),
        .score_key_tile(score_key_tile[3:0]),
        .score_valid_keys(score_valid_keys),
        .scores_q10_packed(scores),
        .probability_group_valid(probability_group_valid),
        .probability_group_ready(probability_group_ready),
        .probability_group(probability_group),
        .probabilities_q16_packed(probabilities_q16_packed),
        .busy(softmax_busy), .done(softmax_done)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            active <= 1'b0;
            probability_half <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start && start_ready) begin
                active <= 1'b1;
                probability_half <= 1'b0;
            end
            if (active && probability_group_valid && probability_group_ready) begin
                if (probability_half) begin
                    active <= 1'b0;
                    done <= 1'b1;
                end else begin
                    probability_half <= 1'b1;
                end
            end
        end
    end

endmodule
