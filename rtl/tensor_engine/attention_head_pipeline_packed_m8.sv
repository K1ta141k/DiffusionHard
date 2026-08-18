`timescale 1ns/1ps

module attention_head_pipeline_packed_m8 #(
    parameter integer DATA_WIDTH = 18,
    parameter integer GROUP_WIDTH = 4,
    parameter integer INTERNAL_MAC = 1,
    parameter integer ARRAY_BACKPRESSURE = 0,
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
    output wire scales_ready,
    output wire attention_tile_valid,
    input  wire attention_tile_ready,
    output wire [GROUP_WIDTH-1:0] attention_group,
    output wire [3:0] attention_output_tile,
    output wire [2:0] attention_valid_channels,
    output wire [4*6*DATA_WIDTH-1:0] attention_q12_packed,
    output wire mac_request_valid,
    input  wire mac_request_ready,
    output wire mac_request_narrow_int8_mode,
    output wire mac_request_clear,
    output wire mac_request_last,
    output wire [5:0] mac_request_tag,
    output wire [4*32*18-1:0] mac_request_attention_activations,
    output wire [6*32*18-1:0] mac_request_attention_weights,
    output wire [8*32*8-1:0] mac_request_narrow_activations,
    output wire [6*32*8-1:0] mac_request_narrow_weights,
    input  wire mac_response_valid,
    input  wire mac_response_narrow_int8_mode,
    input  wire [5:0] mac_response_tag,
    input  wire [4*6*48-1:0] mac_response_attention_accumulators,
    input  wire [8*6*32-1:0] mac_response_narrow_accumulators,
    output wire busy,
    output reg  done
);

    localparam STATE_IDLE = 1'b0;
    localparam STATE_RUN = 1'b1;

    reg state;
    reg [2:0] active_pair;
    reg pair_start_pending;
    reg scale_table_complete;

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

    wire scale_input_valid = load_valid
        || (query_load_valid && key_load_valid);
    wire scale_input_ready;
    wire scale_valid;
    wire [5:0] scale_token;
    wire [17:0] scale_query_maximum;
    wire [17:0] scale_key_maximum;
    wire [23:0] scale_query_multiplier_q17;
    wire [23:0] scale_key_multiplier_q17;

    wire pair_start_ready;
    wire pair_start = state == STATE_RUN && pair_start_pending
        && pair_start_ready;
    wire pair_busy;
    wire pair_done;

    assign start_ready = state == STATE_IDLE && scale_table_complete;
    assign scales_ready = scale_table_complete;
    assign busy = state != STATE_IDLE;

    attention_head_scratchpad_qk_combined_banked #(
        .DATA_WIDTH(DATA_WIDTH),
        .SEPARATE_VALUE_LOAD_ADDRESS(SEPARATE_VALUE_LOAD_ADDRESS)
    ) scratchpad (
        .clk(clk), .load_valid(load_valid),
        .query_load_valid(query_load_valid),
        .key_load_valid(key_load_valid),
        .value_load_valid(value_load_valid), .load_token(load_token),
        .load_channel(load_channel), .value_load_token(value_load_token),
        .value_load_channel(value_load_channel),
        .load_query_q12(load_query_q12), .load_key_q12(load_key_q12),
        .load_value_q12(load_value_q12),
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

    attention_dynamic_vector_scale_tracker scale_tracker (
        .clk(clk), .rst_n(rst_n), .valid_in(scale_input_valid),
        .ready_in(scale_input_ready), .token_in(load_token),
        .channel_in(load_channel), .query_q12_in(load_query_q12),
        .key_q12_in(load_key_q12), .scale_valid(scale_valid),
        .scale_token(scale_token), .query_maximum(scale_query_maximum),
        .key_maximum(scale_key_maximum),
        .query_multiplier_q17(scale_query_multiplier_q17),
        .key_multiplier_q17(scale_key_multiplier_q17)
    );

    attention_group_pair_pipeline_packed_m8 #(
        .INTERNAL_MAC(INTERNAL_MAC),
        .ARRAY_BACKPRESSURE(ARRAY_BACKPRESSURE), .LUT_FILE(LUT_FILE)
    ) pair_pipeline (
        .clk(clk), .rst_n(rst_n), .scale_load_valid(scale_valid),
        .scale_load_token(scale_token),
        .scale_load_query_maximum(scale_query_maximum),
        .scale_load_key_maximum(scale_key_maximum),
        .scale_load_query_multiplier_q17(scale_query_multiplier_q17),
        .scale_load_key_multiplier_q17(scale_key_multiplier_q17),
        .start(pair_start), .group_pair_in(active_pair),
        .start_ready(pair_start_ready),
        .query_read_valid(query_read_valid),
        .query_read_token(query_read_token),
        .query_read_channel_block(query_read_channel_block),
        .query_data_valid(query_data_valid),
        .query_data_q12_packed(query_data), .key_read_valid(key_read_valid),
        .key_read_token(key_read_token),
        .key_read_channel_block(key_read_channel_block),
        .key_data_valid(key_data_valid), .key_data_q12_packed(key_data),
        .value_read_valid(value_read_valid),
        .value_read_key_block(value_read_key_block),
        .value_read_channel(value_read_channel),
        .value_data_valid(value_data_valid),
        .value_data_q12_packed(value_data),
        .attention_tile_valid(attention_tile_valid),
        .attention_tile_ready(attention_tile_ready),
        .attention_group(attention_group),
        .attention_output_tile(attention_output_tile),
        .attention_valid_channels(attention_valid_channels),
        .attention_q12_packed(attention_q12_packed),
        .mac_request_valid(mac_request_valid),
        .mac_request_ready(mac_request_ready),
        .mac_request_narrow_int8_mode(mac_request_narrow_int8_mode),
        .mac_request_clear(mac_request_clear),
        .mac_request_last(mac_request_last),
        .mac_request_tag(mac_request_tag),
        .mac_request_attention_activations(
            mac_request_attention_activations
        ),
        .mac_request_attention_weights(mac_request_attention_weights),
        .mac_request_narrow_activations(mac_request_narrow_activations),
        .mac_request_narrow_weights(mac_request_narrow_weights),
        .mac_response_valid(mac_response_valid),
        .mac_response_narrow_int8_mode(mac_response_narrow_int8_mode),
        .mac_response_tag(mac_response_tag),
        .mac_response_attention_accumulators(
            mac_response_attention_accumulators
        ),
        .mac_response_narrow_accumulators(
            mac_response_narrow_accumulators
        ),
        .busy(pair_busy), .done(pair_done)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_pair <= 0;
            pair_start_pending <= 1'b0;
            scale_table_complete <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (scale_valid && scale_token == 63)
                scale_table_complete <= 1'b1;
            if (state == STATE_IDLE && start && start_ready) begin
                state <= STATE_RUN;
                active_pair <= 0;
                pair_start_pending <= 1'b1;
                scale_table_complete <= 1'b0;
            end else if (pair_start) begin
                pair_start_pending <= 1'b0;
            end
            if (state == STATE_RUN && pair_done) begin
                if (active_pair == 7) begin
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else begin
                    active_pair <= active_pair + 1'b1;
                    pair_start_pending <= 1'b1;
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && scale_input_valid && !scale_input_ready)
            $error("packed attention scale tracker input overflow");
        if (rst_n && query_load_valid != key_load_valid)
            $error("packed attention requires aligned query and key writes");
        if (rst_n && (load_valid || query_load_valid || key_load_valid
            || value_load_valid) && state != STATE_IDLE)
            $error("packed attention scratchpad load overlapped execution");
`endif
    end

endmodule
