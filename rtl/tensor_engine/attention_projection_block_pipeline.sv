`timescale 1ns/1ps

module attention_projection_block_pipeline #(
    parameter integer OUTPUT_TILES = 128,
    parameter integer INTERNAL_MAC = 1,
    parameter integer ENABLE_RESIDUAL_REPLAY = 0,
    parameter integer GROUPED_CANVAS = 0
) (
    input  wire clk,
    input  wire rst_n,
    input  wire residual_load_valid,
    input  wire [3:0] residual_load_group,
    input  wire [6:0] residual_load_output_tile,
    input  wire [4*6*24-1:0] residual_load_q10_packed,
    input  wire residual_replay_read_valid,
    input  wire [3:0] residual_replay_read_group,
    input  wire [6:0] residual_replay_read_output_tile,
    output wire residual_replay_read_data_valid,
    output wire [4*6*24-1:0] residual_replay_read_q10_packed,
    input  wire block_start,
    output wire block_start_ready,
    input  wire metadata_valid,
    output wire metadata_ready,
    output wire parameter_request_valid,
    input  wire [6:0] metadata_output_tile,
    input  wire [6*24-1:0] metadata_multipliers_packed,
    input  wire weight_tile_valid,
    output wire weight_tile_ready,
    input  wire [6:0] weight_output_tile,
    input  wire [4:0] weight_input_tile,
    input  wire [6*32*8-1:0] weight_int8_packed,
    output wire [6:0] requested_output_tile,
    output wire canvas_read_valid,
    output wire [3:0] canvas_read_head,
    output wire [5:0] canvas_read_token,
    input  wire canvas_read_data_valid,
    input  wire [64*18-1:0] canvas_read_data_packed,
    output wire canvas_group_read_valid,
    output wire [3:0] canvas_group_read_head,
    output wire [3:0] canvas_group_read_group,
    input  wire canvas_group_read_data_valid,
    input  wire [4*64*18-1:0] canvas_group_read_data_packed,
    output wire block_tile_valid,
    input  wire block_tile_ready,
    output wire [3:0] block_group,
    output wire [6:0] block_output_tile,
    output wire [4*6*24-1:0] block_q10_packed,
    output wire array_request_valid,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [7:0] array_request_tag,
    output wire [4*32*18-1:0] array_request_activations,
    output wire [6*32*18-1:0] array_request_weights,
    input  wire array_response_valid,
    input  wire [7:0] array_response_tag,
    input  wire [4*6*48-1:0] array_response_accumulators,
    output wire busy,
    output reg  done
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_METADATA = 2'd1;
    localparam [1:0] STATE_START = 2'd2;
    localparam [1:0] STATE_RUN = 2'd3;

    reg [1:0] state;
    reg [6:0] active_output_tile;
    reg [6*24-1:0] active_multipliers;
    wire output_tile_start_ready;
    wire output_tile_start = (state == STATE_START)
        && output_tile_start_ready;
    wire output_tile_weight_ready;
    wire output_tile_busy;
    wire output_tile_done;

    assign block_start_ready = (state == STATE_IDLE);
    assign metadata_ready = (state == STATE_METADATA)
        && metadata_output_tile == active_output_tile;
    assign parameter_request_valid = state == STATE_METADATA;
    assign weight_tile_ready = output_tile_weight_ready
        && weight_output_tile == active_output_tile;
    assign requested_output_tile = active_output_tile;
    assign busy = (state != STATE_IDLE);

    attention_projection_residual_output_tile #(
        .INTERNAL_MAC(INTERNAL_MAC),
        .ENABLE_RESIDUAL_REPLAY(ENABLE_RESIDUAL_REPLAY),
        .GROUPED_CANVAS(GROUPED_CANVAS)
    ) output_tile_pipeline (
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
        .start(output_tile_start), .start_ready(output_tile_start_ready),
        .output_tile_in(active_output_tile),
        .multipliers_packed(active_multipliers),
        .weight_tile_valid(weight_tile_valid && weight_tile_ready),
        .weight_tile_ready(output_tile_weight_ready),
        .weight_input_tile(weight_input_tile),
        .weight_int8_packed(weight_int8_packed),
        .canvas_read_valid(canvas_read_valid),
        .canvas_read_head(canvas_read_head),
        .canvas_read_token(canvas_read_token),
        .canvas_read_data_valid(canvas_read_data_valid),
        .canvas_read_data_packed(canvas_read_data_packed),
        .canvas_group_read_valid(canvas_group_read_valid),
        .canvas_group_read_head(canvas_group_read_head),
        .canvas_group_read_group(canvas_group_read_group),
        .canvas_group_read_data_valid(canvas_group_read_data_valid),
        .canvas_group_read_data_packed(canvas_group_read_data_packed),
        .block_tile_valid(block_tile_valid),
        .block_tile_ready(block_tile_ready), .block_group(block_group),
        .block_output_tile(block_output_tile),
        .block_q10_packed(block_q10_packed),
        .array_request_valid(array_request_valid),
        .array_request_clear(array_request_clear),
        .array_request_last(array_request_last),
        .array_request_tag(array_request_tag),
        .array_request_activations(array_request_activations),
        .array_request_weights(array_request_weights),
        .array_response_valid(array_response_valid),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators),
        .busy(output_tile_busy), .done(output_tile_done)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_output_tile <= 0;
            active_multipliers <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && block_start) begin
                active_output_tile <= 0;
                state <= STATE_METADATA;
            end else if (state == STATE_METADATA
                         && metadata_valid && metadata_ready) begin
                active_multipliers <= metadata_multipliers_packed;
                state <= STATE_START;
            end else if (state == STATE_START && output_tile_start) begin
                state <= STATE_RUN;
            end else if (state == STATE_RUN && output_tile_done) begin
                if (active_output_tile == OUTPUT_TILES-1) begin
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else begin
                    active_output_tile <= active_output_tile + 1'b1;
                    state <= STATE_METADATA;
                end
            end
        end
    end

    initial begin
        if (OUTPUT_TILES < 1 || OUTPUT_TILES > 128)
            $error("attention projection supports one through 128 output tiles");
    end

endmodule
