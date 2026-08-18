`timescale 1ns/1ps

module qkv_head_projection_pipeline #(
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
    output wire qkv_tile_valid,
    input  wire qkv_tile_ready,
    output wire [3:0] qkv_group,
    output wire [3:0] qkv_head,
    output wire [1:0] qkv_kind,
    output wire [3:0] qkv_channel_tile,
    output wire [2:0] qkv_valid_channels,
    output wire [4*6*18-1:0] qkv_q12_packed,
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
    output wire busy,
    output wire done
);

    wire controller_metadata_enable;
    wire controller_tile_start;
    wire scheduler_start_ready;
    wire scheduler_done;
    wire scheduler_busy;
    wire [8:0] scheduler_output_tile;
    reg [6*24-1:0] active_multipliers;
    reg [6*18-1:0] active_biases;
    wire metadata_matches = metadata_head == requested_head
        && metadata_kind == requested_kind
        && metadata_channel_tile == requested_channel_tile;
    wire weight_matches = weight_head == requested_head
        && weight_kind == requested_kind
        && weight_channel_tile == requested_channel_tile;
    wire accepted_metadata = metadata_valid && metadata_ready;
    wire scheduler_weight_ready;

    assign metadata_ready = controller_metadata_enable && metadata_matches;
    assign parameter_request_valid = controller_metadata_enable;
    assign weight_tile_ready = scheduler_weight_ready && weight_matches;
    assign qkv_head = requested_head;
    assign qkv_kind = scheduler_output_tile[5:4];
    assign qkv_channel_tile = scheduler_output_tile[3:0];
    assign qkv_valid_channels = requested_valid_channels;

    qkv_head_tile_controller #(
        .KINDS(KINDS), .CHANNEL_TILES(CHANNEL_TILES),
        .LAST_TILE_VALID_CHANNELS(LAST_TILE_VALID_CHANNELS)
    ) controller (
        .clk(clk), .rst_n(rst_n), .start(start), .start_ready(start_ready),
        .head_in(head_in), .metadata_enable(controller_metadata_enable),
        .metadata_fire(accepted_metadata),
        .tile_start_ready(scheduler_start_ready),
        .tile_start(controller_tile_start), .tile_done(scheduler_done),
        .active_head(requested_head), .active_kind(requested_kind),
        .active_channel_tile(requested_channel_tile),
        .active_valid_channels(requested_valid_channels),
        .active_global_row(requested_global_row), .busy(busy), .done(done)
    );

    qkv_projection_output_tile_scheduler_streaming #(
        .INTERNAL_MAC(INTERNAL_MAC),
        .ARRAY_BACKPRESSURE(ARRAY_BACKPRESSURE)
    ) scheduler (
        .clk(clk), .rst_n(rst_n), .start(controller_tile_start),
        .start_ready(scheduler_start_ready),
        .output_tile_in({3'b0, requested_kind, requested_channel_tile}),
        .multipliers_packed(active_multipliers),
        .biases_q12_packed(active_biases),
        .weight_tile_valid(weight_tile_valid && weight_tile_ready),
        .weight_tile_ready(scheduler_weight_ready),
        .weight_input_tile(weight_input_tile),
        .weight_int16_packed(weight_int16_packed),
        .normalized_read_valid(normalized_read_valid),
        .normalized_read_group(normalized_read_group),
        .normalized_read_input_tile(normalized_read_input_tile),
        .normalized_read_data_valid(normalized_read_data_valid),
        .normalized_q12_packed(normalized_q12_packed),
        .qkv_tile_valid(qkv_tile_valid), .qkv_tile_ready(qkv_tile_ready),
        .qkv_group(qkv_group), .qkv_output_tile(scheduler_output_tile),
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
        .busy(scheduler_busy), .done(scheduler_done)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            active_multipliers <= 0;
            active_biases <= 0;
        end else if (accepted_metadata) begin
            active_multipliers <= metadata_multipliers_packed;
            active_biases <= metadata_biases_q12_packed;
        end
    end

endmodule
