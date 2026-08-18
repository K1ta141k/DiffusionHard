`timescale 1ns/1ps

module attention_group_pipeline #(
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 18,
    parameter integer ACC_WIDTH = 48,
    parameter integer GROUP_WIDTH = 4,
    parameter integer INTERNAL_MAC = 1,
    parameter LUT_FILE = "rtl/tensor_engine/exp_neg_q16_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [GROUP_WIDTH-1:0] group_in,
    output wire start_ready,
    output wire query_read_valid,
    output wire [5:0] query_read_token,
    output wire [1:0] query_read_channel_block,
    input  wire query_data_valid,
    input  wire [16*DATA_WIDTH-1:0] query_data_packed,
    output wire key_read_valid,
    output wire [5:0] key_read_token,
    output wire [1:0] key_read_channel_block,
    input  wire key_data_valid,
    input  wire [16*DATA_WIDTH-1:0] key_data_packed,
    output wire value_read_valid,
    output wire [1:0] value_read_key_block,
    output wire [5:0] value_read_channel,
    input  wire value_data_valid,
    input  wire [16*DATA_WIDTH-1:0] value_data_packed,
    output wire attention_tile_valid,
    input  wire attention_tile_ready,
    output wire [GROUP_WIDTH-1:0] attention_group,
    output wire [3:0] attention_output_tile,
    output wire [2:0] attention_valid_channels,
    output wire [M_LANES*N_LANES*DATA_WIDTH-1:0]
        attention_q12_packed,
    output wire array_request_valid,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [3:0] array_request_tag,
    output wire [M_LANES*32*DATA_WIDTH-1:0] array_request_activations,
    output wire [N_LANES*32*DATA_WIDTH-1:0] array_request_weights,
    input  wire array_response_valid,
    input  wire [3:0] array_response_tag,
    input  wire [M_LANES*N_LANES*ACC_WIDTH-1:0]
        array_response_accumulators,
    output wire busy,
    output wire done
);

    wire score_valid;
    wire score_ready;
    wire [GROUP_WIDTH-1:0] score_group;
    wire [3:0] score_key_tile;
    wire [2:0] score_valid_keys;
    wire [M_LANES*N_LANES*DATA_WIDTH-1:0] scores_q10;
    wire qk_busy;
    wire qk_done;

    wire probability_group_valid;
    wire probability_group_ready;
    wire [GROUP_WIDTH-1:0] probability_group;
    wire [M_LANES*64*16-1:0] probabilities_q16;
    wire softmax_busy;
    wire softmax_done;
    wire pv_start_ready;
    wire pv_busy;

    wire qk_mac_valid;
    wire qk_mac_clear;
    wire qk_mac_last;
    wire [3:0] qk_mac_tag;
    wire [M_LANES*32*DATA_WIDTH-1:0] qk_mac_activations;
    wire [N_LANES*32*DATA_WIDTH-1:0] qk_mac_weights;
    wire pv_mac_valid;
    wire pv_mac_clear;
    wire pv_mac_last;
    wire [3:0] pv_mac_tag;
    wire [M_LANES*32*DATA_WIDTH-1:0] pv_mac_activations;
    wire [N_LANES*32*DATA_WIDTH-1:0] pv_mac_weights;

    wire shared_mac_valid = qk_mac_valid || pv_mac_valid;
    wire shared_mac_clear = qk_mac_valid ? qk_mac_clear : pv_mac_clear;
    wire shared_mac_last = qk_mac_valid ? qk_mac_last : pv_mac_last;
    wire [3:0] shared_mac_tag = qk_mac_valid ? qk_mac_tag : pv_mac_tag;
    wire [M_LANES*32*DATA_WIDTH-1:0] shared_mac_activations =
        qk_mac_valid ? qk_mac_activations : pv_mac_activations;
    wire [N_LANES*32*DATA_WIDTH-1:0] shared_mac_weights =
        qk_mac_valid ? qk_mac_weights : pv_mac_weights;
    wire internal_mac_output_valid;
    wire [3:0] internal_mac_output_tag;
    wire [M_LANES*N_LANES*ACC_WIDTH-1:0] internal_mac_accumulators;
    wire shared_mac_output_valid = INTERNAL_MAC
        ? internal_mac_output_valid : array_response_valid;
    wire [3:0] shared_mac_output_tag = INTERNAL_MAC
        ? internal_mac_output_tag : array_response_tag;
    wire [M_LANES*N_LANES*ACC_WIDTH-1:0] shared_mac_accumulators = INTERNAL_MAC
        ? internal_mac_accumulators : array_response_accumulators;

    assign probability_group_ready = pv_start_ready;
    assign array_request_valid = shared_mac_valid;
    assign array_request_clear = shared_mac_clear;
    assign array_request_last = shared_mac_last;
    assign array_request_tag = shared_mac_tag;
    assign array_request_activations = shared_mac_activations;
    assign array_request_weights = shared_mac_weights;
    assign start_ready = !busy;
    assign busy = qk_busy || softmax_busy || pv_busy
        || probability_group_valid;

    attention_qk_group_scheduler #(
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .GROUP_WIDTH(GROUP_WIDTH),
        .INTERNAL_MAC(0)
    ) qk_scheduler (
        .clk(clk), .rst_n(rst_n), .start(start), .group_in(group_in),
        .start_ready(),
        .query_read_valid(query_read_valid),
        .query_read_token(query_read_token),
        .query_read_channel_block(query_read_channel_block),
        .query_data_valid(query_data_valid),
        .query_data_packed(query_data_packed),
        .key_read_valid(key_read_valid), .key_read_token(key_read_token),
        .key_read_channel_block(key_read_channel_block),
        .key_data_valid(key_data_valid), .key_data_packed(key_data_packed),
        .score_valid(score_valid), .score_ready(score_ready),
        .score_group(score_group), .score_key_tile(score_key_tile),
        .score_valid_keys(score_valid_keys), .scores_q10_packed(scores_q10),
        .mac_request_valid(qk_mac_valid),
        .mac_request_clear(qk_mac_clear), .mac_request_last(qk_mac_last),
        .mac_request_tag(qk_mac_tag),
        .mac_request_activations(qk_mac_activations),
        .mac_request_weights(qk_mac_weights),
        .mac_response_valid(shared_mac_output_valid),
        .mac_response_tag(shared_mac_output_tag),
        .mac_response_accumulators(shared_mac_accumulators),
        .busy(qk_busy), .done(qk_done)
    );

    attention_score_group_softmax_stream #(
        .M_LANES(M_LANES), .N_LANES(N_LANES),
        .SCORE_WIDTH(DATA_WIDTH), .GROUP_WIDTH(GROUP_WIDTH),
        .LUT_FILE(LUT_FILE)
    ) group_softmax (
        .clk(clk), .rst_n(rst_n), .score_tile_valid(score_valid),
        .score_tile_ready(score_ready), .score_tile_group(score_group),
        .score_key_tile(score_key_tile), .score_valid_keys(score_valid_keys),
        .scores_q10_packed(scores_q10),
        .probability_group_valid(probability_group_valid),
        .probability_group_ready(probability_group_ready),
        .probability_group(probability_group),
        .probabilities_q16_packed(probabilities_q16),
        .busy(softmax_busy), .done(softmax_done)
    );

    attention_pv_group_scheduler #(
        .M_LANES(M_LANES), .N_LANES(N_LANES),
        .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .GROUP_WIDTH(GROUP_WIDTH), .INTERNAL_MAC(0)
    ) pv_scheduler (
        .clk(clk), .rst_n(rst_n),
        .start(probability_group_valid && probability_group_ready),
        .group_in(probability_group),
        .probabilities_q16_packed(probabilities_q16),
        .start_ready(pv_start_ready),
        .value_read_valid(value_read_valid),
        .value_read_key_block(value_read_key_block),
        .value_read_channel(value_read_channel),
        .value_data_valid(value_data_valid), .value_data_packed(value_data_packed),
        .attention_tile_valid(attention_tile_valid),
        .attention_tile_ready(attention_tile_ready),
        .attention_group(attention_group),
        .attention_output_tile(attention_output_tile),
        .attention_valid_channels(attention_valid_channels),
        .attention_q12_packed(attention_q12_packed),
        .mac_request_valid(pv_mac_valid),
        .mac_request_clear(pv_mac_clear), .mac_request_last(pv_mac_last),
        .mac_request_tag(pv_mac_tag),
        .mac_request_activations(pv_mac_activations),
        .mac_request_weights(pv_mac_weights),
        .mac_response_valid(shared_mac_output_valid),
        .mac_response_tag(shared_mac_output_tag),
        .mac_response_accumulators(shared_mac_accumulators),
        .busy(pv_busy), .done(done)
    );

    generate
        if (INTERNAL_MAC) begin : internal_array
            mixed_precision_mac_tile_pipelined #(
                .M_LANES(M_LANES), .N_LANES(N_LANES),
                .STORAGE_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .TAG_WIDTH(4)
            ) shared_mac (
                .clk(clk), .rst_n(rst_n), .valid_in(shared_mac_valid),
                .narrow_int8_mode(1'b0), .clear_accumulators(shared_mac_clear),
                .last_k_tile(shared_mac_last), .tag_in(shared_mac_tag),
                .activations_packed(shared_mac_activations),
                .weights_packed(shared_mac_weights),
                .valid_out(internal_mac_output_valid),
                .tag_out(internal_mac_output_tag),
                .accumulators_packed(internal_mac_accumulators)
            );
        end else begin : no_internal_array
            assign internal_mac_output_valid = 1'b0;
            assign internal_mac_output_tag = 4'b0;
            assign internal_mac_accumulators = {M_LANES*N_LANES*ACC_WIDTH{1'b0}};
        end
    endgenerate

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && qk_mac_valid && pv_mac_valid)
            $error("QK and probability-times-V requested the shared MAC together");
        if (rst_n && start && !start_ready)
            $error("attention group start arrived while busy");
`endif
    end

endmodule
