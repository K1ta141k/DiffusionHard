`timescale 1ns/1ps

module qkv_head_staging_pipeline #(
    parameter integer KINDS = 3,
    parameter integer CHANNEL_TILES = 11,
    parameter integer LAST_TILE_VALID_CHANNELS = 4,
    parameter integer INTERNAL_MAC = 1,
    parameter integer ARRAY_BACKPRESSURE = 0
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
    output wire query_write_valid,
    output wire key_write_valid,
    output wire value_write_valid,
    output wire [5:0] query_key_write_token,
    output wire [5:0] query_key_write_channel,
    output wire signed [17:0] query_write_q12,
    output wire signed [17:0] key_write_q12,
    output wire [5:0] value_write_token,
    output wire [5:0] value_write_channel,
    output wire signed [17:0] value_write_q12,
    output wire array_request_valid,
    input  wire array_request_ready,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [7:0] array_request_tag,
    output wire [4*32*18-1:0] array_request_activations,
    output wire [6*32*18-1:0] array_request_weights,
    input  wire array_response_valid,
    input  wire [7:0] array_response_tag,
    input  wire [4*6*48-1:0] array_response_accumulators,
    output wire projection_busy,
    output wire rotary_busy,
    output wire busy,
    output reg  done
);

    wire projection_start_ready;
    wire projection_done;
    wire qkv_tile_valid;
    wire qkv_tile_ready;
    wire [3:0] qkv_group;
    wire [3:0] qkv_head;
    wire [1:0] qkv_kind;
    wire [3:0] qkv_channel_tile;
    wire [2:0] qkv_valid_channels;
    wire [4*6*18-1:0] qkv_q12_packed;
    wire router_query_load;
    wire router_key_load;
    wire router_value_load;
    wire [3:0] router_load_head;
    wire [5:0] router_load_token;
    wire [5:0] router_load_channel;
    wire signed [17:0] router_load_q12;
    wire router_done;
    wire [3:0] router_done_head;
    wire [1:0] router_done_kind;
    wire [3:0] router_done_group;
    wire [3:0] router_done_channel_tile;
    wire router_busy;
    wire rotary_start_ready;
    wire rotary_start = router_done && router_done_kind == 1
        && router_done_group == 15
        && router_done_channel_tile == CHANNEL_TILES-1;
    wire qk_read_valid;
    wire [5:0] qk_read_token;
    wire [4:0] qk_read_pair;
    wire qk_read_data_valid;
    wire [5:0] qk_read_token_out;
    wire [4:0] qk_read_pair_out;
    wire signed [17:0] query_first_q12;
    wire signed [17:0] query_second_q12;
    wire signed [17:0] key_first_q12;
    wire signed [17:0] key_second_q12;
    wire constant_read_valid;
    wire [5:0] constant_read_token;
    wire [4:0] constant_read_pair;
    wire constant_read_data_valid;
    wire [5:0] constant_read_token_out;
    wire [4:0] constant_read_pair_out;
    wire signed [15:0] cosine_q15;
    wire signed [15:0] sine_q15;
    wire rotary_done;
    reg active;
    reg projection_finished;
    reg rotary_finished;
    reg value_finished;

    assign start_ready = !active && projection_start_ready;
    assign busy = active;
    assign value_write_valid = router_value_load;
    assign value_write_token = router_load_token;
    assign value_write_channel = router_load_channel;
    assign value_write_q12 = router_load_q12;

    qkv_head_projection_pipeline #(
        .KINDS(KINDS), .CHANNEL_TILES(CHANNEL_TILES),
        .LAST_TILE_VALID_CHANNELS(LAST_TILE_VALID_CHANNELS),
        .INTERNAL_MAC(INTERNAL_MAC),
        .ARRAY_BACKPRESSURE(ARRAY_BACKPRESSURE)
    ) projection (
        .clk(clk), .rst_n(rst_n), .start(start && start_ready),
        .start_ready(projection_start_ready), .head_in(head_in),
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
        .qkv_tile_valid(qkv_tile_valid), .qkv_tile_ready(qkv_tile_ready),
        .qkv_group(qkv_group), .qkv_head(qkv_head), .qkv_kind(qkv_kind),
        .qkv_channel_tile(qkv_channel_tile),
        .qkv_valid_channels(qkv_valid_channels),
        .qkv_q12_packed(qkv_q12_packed),
        .array_request_valid(array_request_valid),
        .array_request_ready(array_request_ready),
        .array_request_clear(array_request_clear),
        .array_request_last(array_request_last),
        .array_request_tag(array_request_tag),
        .array_request_activations(array_request_activations),
        .array_request_weights(array_request_weights),
        .array_response_valid(array_response_valid),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators),
        .busy(projection_busy), .done(projection_done)
    );

    qkv_head_output_router router (
        .clk(clk), .rst_n(rst_n), .tile_valid(qkv_tile_valid),
        .tile_ready(qkv_tile_ready), .tile_head(qkv_head),
        .tile_kind(qkv_kind), .tile_group(qkv_group),
        .tile_channel_tile(qkv_channel_tile),
        .tile_valid_channels(qkv_valid_channels),
        .tile_q12_packed(qkv_q12_packed),
        .query_load_valid(router_query_load),
        .key_load_valid(router_key_load),
        .value_load_valid(router_value_load), .load_head(router_load_head),
        .load_token(router_load_token), .load_channel(router_load_channel),
        .load_q12(router_load_q12),
        .tile_done(router_done),
        .done_head(router_done_head), .done_kind(router_done_kind),
        .done_group(router_done_group),
        .done_channel_tile(router_done_channel_tile), .busy(router_busy)
    );

    qk_unrotated_scratchpad_paired_uram unrotated (
        .clk(clk), .query_load_valid(router_query_load),
        .key_load_valid(router_key_load), .load_token(router_load_token),
        .load_channel(router_load_channel), .load_query_q12(router_load_q12),
        .load_key_q12(router_load_q12),
        .read_valid(qk_read_valid),
        .read_token(qk_read_token), .read_pair(qk_read_pair),
        .read_data_valid(qk_read_data_valid),
        .read_token_out(qk_read_token_out), .read_pair_out(qk_read_pair_out),
        .query_first_q12(query_first_q12),
        .query_second_q12(query_second_q12), .key_first_q12(key_first_q12),
        .key_second_q12(key_second_q12)
    );

    rotary_constant_table_bram constants (
        .clk(clk), .load_valid(constant_load_valid),
        .load_token(constant_load_token), .load_pair(constant_load_pair),
        .load_cosine_q15(constant_load_cosine_q15),
        .load_sine_q15(constant_load_sine_q15),
        .read_valid(constant_read_valid), .read_token(constant_read_token),
        .read_pair(constant_read_pair),
        .read_data_valid(constant_read_data_valid),
        .read_token_out(constant_read_token_out),
        .read_pair_out(constant_read_pair_out), .cosine_q15(cosine_q15),
        .sine_q15(sine_q15)
    );

    rotary_head_writeback_scheduler rotary (
        .clk(clk), .rst_n(rst_n), .start(rotary_start),
        .start_ready(rotary_start_ready), .head_in(router_done_head),
        .qk_read_valid(qk_read_valid), .qk_read_token(qk_read_token),
        .qk_read_pair(qk_read_pair), .qk_read_data_valid(qk_read_data_valid),
        .qk_read_token_out(qk_read_token_out),
        .qk_read_pair_out(qk_read_pair_out), .query_first_q12(query_first_q12),
        .query_second_q12(query_second_q12), .key_first_q12(key_first_q12),
        .key_second_q12(key_second_q12),
        .constant_read_valid(constant_read_valid),
        .constant_read_token(constant_read_token),
        .constant_read_pair(constant_read_pair),
        .constant_read_data_valid(constant_read_data_valid),
        .constant_read_token_out(constant_read_token_out),
        .constant_read_pair_out(constant_read_pair_out),
        .cosine_q15(cosine_q15), .sine_q15(sine_q15),
        .query_write_valid(query_write_valid),
        .key_write_valid(key_write_valid), .write_token(query_key_write_token),
        .write_channel(query_key_write_channel),
        .write_query_q12(query_write_q12), .write_key_q12(key_write_q12),
        .busy(rotary_busy), .done(rotary_done)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            active <= 1'b0;
            projection_finished <= 1'b0;
            rotary_finished <= 1'b0;
            value_finished <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start && start_ready) begin
                active <= 1'b1;
                projection_finished <= 1'b0;
                rotary_finished <= 1'b0;
                value_finished <= 1'b0;
            end
            if (projection_done)
                projection_finished <= 1'b1;
            if (rotary_done)
                rotary_finished <= 1'b1;
            if (router_done && router_done_kind == 2
                && router_done_group == 15
                && router_done_channel_tile == CHANNEL_TILES-1)
                value_finished <= 1'b1;
            if (active && projection_finished && rotary_finished
                && value_finished) begin
                active <= 1'b0;
                done <= 1'b1;
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && rotary_start && !rotary_start_ready)
            $error("rotary scheduler was not ready at the Q/K boundary");
        if (rst_n && router_done && router_done_head != head_in)
            $error("QKV staging router completed the wrong head");
`endif
    end

endmodule
