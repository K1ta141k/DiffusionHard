`timescale 1ns/1ps

module attention_group_pair_pipeline_packed_m8 #(
    parameter integer N_LANES = 6,
    parameter integer INTERNAL_MAC = 1,
    parameter integer ARRAY_BACKPRESSURE = 0,
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
    output wire value_read_valid,
    output wire [1:0] value_read_key_block,
    output wire [5:0] value_read_channel,
    input  wire value_data_valid,
    input  wire [16*18-1:0] value_data_q12_packed,
    output wire attention_tile_valid,
    input  wire attention_tile_ready,
    output wire [3:0] attention_group,
    output wire [3:0] attention_output_tile,
    output wire [2:0] attention_valid_channels,
    output wire [4*N_LANES*18-1:0] attention_q12_packed,
    output wire mac_request_valid,
    input  wire mac_request_ready,
    output wire mac_request_narrow_int8_mode,
    output wire mac_request_clear,
    output wire mac_request_last,
    output wire [5:0] mac_request_tag,
    output wire [4*32*18-1:0] mac_request_attention_activations,
    output wire [N_LANES*32*18-1:0] mac_request_attention_weights,
    output wire [8*32*8-1:0] mac_request_narrow_activations,
    output wire [N_LANES*32*8-1:0] mac_request_narrow_weights,
    input  wire mac_response_valid,
    input  wire mac_response_narrow_int8_mode,
    input  wire [5:0] mac_response_tag,
    input  wire [4*N_LANES*48-1:0]
        mac_response_attention_accumulators,
    input  wire [8*N_LANES*32-1:0]
        mac_response_narrow_accumulators,
    output wire busy,
    output reg  done
);

    wire probability_group_valid;
    wire probability_group_ready;
    wire [3:0] probability_group;
    wire [4*64*16-1:0] probabilities_q16;
    wire qk_softmax_busy;
    wire qk_softmax_done;
    wire qk_mac_valid;
    wire qk_mac_clear;
    wire qk_mac_last;
    wire [5:0] qk_mac_tag;
    wire [8*32*8-1:0] qk_mac_activations;
    wire [N_LANES*32*8-1:0] qk_mac_weights;
    wire qk_mac_response_valid;
    wire [5:0] qk_mac_response_tag;
    wire [8*N_LANES*32-1:0] qk_mac_response_accumulators;

    wire pv_start_ready;
    wire pv_busy;
    wire pv_done;
    wire pv_mac_valid;
    wire pv_mac_clear;
    wire pv_mac_last;
    wire [3:0] pv_mac_tag;
    wire [4*32*18-1:0] pv_mac_activations;
    wire [N_LANES*32*18-1:0] pv_mac_weights;
    wire pv_mac_response_valid;
    wire [3:0] pv_mac_response_tag;
    wire [4*N_LANES*48-1:0] pv_mac_response_accumulators;

    wire shared_mac_valid = qk_mac_valid || pv_mac_valid;
    wire shared_mac_ready = INTERNAL_MAC || !ARRAY_BACKPRESSURE
        || mac_request_ready;
    wire shared_mac_narrow = qk_mac_valid;
    wire shared_mac_clear = qk_mac_valid ? qk_mac_clear : pv_mac_clear;
    wire shared_mac_last = qk_mac_valid ? qk_mac_last : pv_mac_last;
    wire [5:0] shared_mac_tag = qk_mac_valid
        ? qk_mac_tag : {2'b0, pv_mac_tag};
    wire internal_array_valid;
    wire internal_array_narrow;
    wire [5:0] internal_array_tag;
    wire [4*N_LANES*48-1:0] internal_array_attention_accumulators;
    wire [8*N_LANES*32-1:0] internal_array_narrow_accumulators;
    wire selected_response_valid = INTERNAL_MAC
        ? internal_array_valid : mac_response_valid;
    wire selected_response_narrow = INTERNAL_MAC
        ? internal_array_narrow : mac_response_narrow_int8_mode;
    wire [5:0] selected_response_tag = INTERNAL_MAC
        ? internal_array_tag : mac_response_tag;
    wire [4*N_LANES*48-1:0] selected_attention_accumulators = INTERNAL_MAC
        ? internal_array_attention_accumulators
        : mac_response_attention_accumulators;
    wire [8*N_LANES*32-1:0] selected_narrow_accumulators = INTERNAL_MAC
        ? internal_array_narrow_accumulators
        : mac_response_narrow_accumulators;
    reg array_mode_hold;
    reg [8*32*8-1:0] qk_activations_hold;
    reg [N_LANES*32*8-1:0] qk_weights_hold;
    reg [4*32*18-1:0] pv_activations_hold;
    reg [N_LANES*32*18-1:0] pv_weights_hold;
    wire array_mode_input = shared_mac_valid
        ? shared_mac_narrow : array_mode_hold;
    wire [8*32*8-1:0] array_qk_activations = qk_mac_valid
        ? qk_mac_activations : qk_activations_hold;
    wire [N_LANES*32*8-1:0] array_qk_weights = qk_mac_valid
        ? qk_mac_weights : qk_weights_hold;
    wire [4*32*18-1:0] array_pv_activations = pv_mac_valid
        ? pv_mac_activations : pv_activations_hold;
    wire [N_LANES*32*18-1:0] array_pv_weights = pv_mac_valid
        ? pv_mac_weights : pv_weights_hold;
    reg active;
    reg pv_half;

    assign start_ready = !active;
    assign busy = active || qk_softmax_busy || pv_busy;
    assign probability_group_ready = pv_start_ready;
    assign qk_mac_response_valid = selected_response_valid
        && selected_response_narrow;
    assign qk_mac_response_tag = selected_response_tag;
    assign qk_mac_response_accumulators = selected_narrow_accumulators;
    assign pv_mac_response_valid = selected_response_valid
        && !selected_response_narrow;
    assign pv_mac_response_tag = selected_response_tag[3:0];
    assign pv_mac_response_accumulators = selected_attention_accumulators;
    assign mac_request_valid = shared_mac_valid;
    assign mac_request_narrow_int8_mode = array_mode_input;
    assign mac_request_clear = shared_mac_clear;
    assign mac_request_last = shared_mac_last;
    assign mac_request_tag = shared_mac_tag;
    assign mac_request_attention_activations = array_pv_activations;
    assign mac_request_attention_weights = array_pv_weights;
    assign mac_request_narrow_activations = array_qk_activations;
    assign mac_request_narrow_weights = array_qk_weights;

    attention_qk_group_pair_softmax_pipeline #(
        .N_LANES(N_LANES), .INTERNAL_MAC(0),
        .ARRAY_BACKPRESSURE(ARRAY_BACKPRESSURE), .LUT_FILE(LUT_FILE)
    ) qk_softmax (
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
        .probability_group_valid(probability_group_valid),
        .probability_group_ready(probability_group_ready),
        .probability_group(probability_group),
        .probabilities_q16_packed(probabilities_q16),
        .mac_request_valid(qk_mac_valid),
        .mac_request_ready(shared_mac_ready),
        .mac_request_clear(qk_mac_clear),
        .mac_request_last(qk_mac_last), .mac_request_tag(qk_mac_tag),
        .mac_request_activations_int8(qk_mac_activations),
        .mac_request_weights_int8(qk_mac_weights),
        .mac_response_valid(qk_mac_response_valid),
        .mac_response_tag(qk_mac_response_tag),
        .mac_response_accumulators(qk_mac_response_accumulators),
        .busy(qk_softmax_busy), .done(qk_softmax_done)
    );

    attention_pv_group_scheduler #(
        .N_LANES(N_LANES), .INTERNAL_MAC(0),
        .ARRAY_BACKPRESSURE(ARRAY_BACKPRESSURE)
    ) pv_scheduler (
        .clk(clk), .rst_n(rst_n),
        .start(probability_group_valid && probability_group_ready),
        .group_in(probability_group),
        .probabilities_q16_packed(probabilities_q16),
        .start_ready(pv_start_ready), .value_read_valid(value_read_valid),
        .value_read_key_block(value_read_key_block),
        .value_read_channel(value_read_channel),
        .value_data_valid(value_data_valid),
        .value_data_packed(value_data_q12_packed),
        .attention_tile_valid(attention_tile_valid),
        .attention_tile_ready(attention_tile_ready),
        .attention_group(attention_group),
        .attention_output_tile(attention_output_tile),
        .attention_valid_channels(attention_valid_channels),
        .attention_q12_packed(attention_q12_packed),
        .mac_request_valid(pv_mac_valid),
        .mac_request_ready(shared_mac_ready),
        .mac_request_clear(pv_mac_clear), .mac_request_last(pv_mac_last),
        .mac_request_tag(pv_mac_tag),
        .mac_request_activations(pv_mac_activations),
        .mac_request_weights(pv_mac_weights),
        .mac_response_valid(pv_mac_response_valid),
        .mac_response_tag(pv_mac_response_tag),
        .mac_response_accumulators(pv_mac_response_accumulators),
        .busy(pv_busy), .done(pv_done)
    );

    generate
        if (INTERNAL_MAC) begin : internal_mac
            mixed_precision_packed_m8_mac_tile_pipelined #(
                .N_LANES(N_LANES), .TAG_WIDTH(6)
            ) shared_array (
                .clk(clk), .rst_n(rst_n), .valid_in(shared_mac_valid),
                .narrow_int8_mode(array_mode_input),
                .clear_accumulators(shared_mac_clear),
                .last_k_tile(shared_mac_last), .tag_in(shared_mac_tag),
                .attention_activations_packed(array_pv_activations),
                .attention_weights_packed(array_pv_weights),
                .mlp_activations_packed(array_qk_activations),
                .mlp_weights_packed(array_qk_weights),
                .valid_out(internal_array_valid),
                .narrow_int8_mode_out(internal_array_narrow),
                .tag_out(internal_array_tag),
                .attention_accumulators_packed(
                    internal_array_attention_accumulators
                ),
                .mlp_accumulators_packed(
                    internal_array_narrow_accumulators
                )
            );
        end else begin : external_mac
            assign internal_array_valid = 1'b0;
            assign internal_array_narrow = 1'b0;
            assign internal_array_tag = 6'b0;
            assign internal_array_attention_accumulators = 0;
            assign internal_array_narrow_accumulators = 0;
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            active <= 1'b0;
            pv_half <= 1'b0;
            array_mode_hold <= 1'b0;
            qk_activations_hold <= 0;
            qk_weights_hold <= 0;
            pv_activations_hold <= 0;
            pv_weights_hold <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (qk_mac_valid) begin
                array_mode_hold <= 1'b1;
                qk_activations_hold <= qk_mac_activations;
                qk_weights_hold <= qk_mac_weights;
            end else if (pv_mac_valid) begin
                array_mode_hold <= 1'b0;
                pv_activations_hold <= pv_mac_activations;
                pv_weights_hold <= pv_mac_weights;
            end
            if (start && start_ready) begin
                active <= 1'b1;
                pv_half <= 1'b0;
            end
            if (active && pv_done) begin
                if (pv_half) begin
                    active <= 1'b0;
                    done <= 1'b1;
                end else begin
                    pv_half <= 1'b1;
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && qk_mac_valid && pv_mac_valid)
            $error("packed QK and fixed18 PV requested the shared array together");
`endif
    end

endmodule
