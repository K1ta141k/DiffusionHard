`timescale 1ns/1ps

module attention_projection_residual_output_tile #(
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
    input  wire start,
    output wire start_ready,
    input  wire [6:0] output_tile_in,
    input  wire [6*24-1:0] multipliers_packed,
    input  wire weight_tile_valid,
    output wire weight_tile_ready,
    input  wire [4:0] weight_input_tile,
    input  wire [6*32*8-1:0] weight_int8_packed,
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
    output reg  [3:0] block_group,
    output reg  [6:0] block_output_tile,
    output reg  [4*6*24-1:0] block_q10_packed,
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
    output wire done
);

    wire scheduler_start_ready;
    wire scheduler_busy;
    wire scheduler_done;
    wire projection_valid;
    wire projection_ready = !block_tile_valid || block_tile_ready;
    wire [3:0] projection_group;
    wire [6:0] projection_output_tile;
    wire [2:0] projection_valid_channels;
    wire [4*6*24-1:0] projection_data;
    wire canvas_data_valid;
    wire [4*6*24-1:0] canvas_data;
    wire replay_read_valid = ENABLE_RESIDUAL_REPLAY
        ? residual_replay_read_valid : 1'b0;
    wire residual_canvas_read_valid = scheduler_busy || replay_read_valid;
    wire [3:0] residual_canvas_read_group = scheduler_busy
        ? projection_group : residual_replay_read_group;
    wire [6:0] residual_canvas_read_output_tile = scheduler_busy
        ? projection_output_tile : residual_replay_read_output_tile;
    wire feedback_valid = block_tile_valid && block_tile_ready;
    wire canvas_load_valid = feedback_valid || residual_load_valid;
    wire [3:0] canvas_load_group = feedback_valid
        ? block_group : residual_load_group;
    wire [6:0] canvas_load_output_tile = feedback_valid
        ? block_output_tile : residual_load_output_tile;
    wire [4*6*24-1:0] canvas_load_data = feedback_valid
        ? block_q10_packed : residual_load_q10_packed;
    reg output_valid;
    reg signed [24:0] sum;
    integer lane;

    assign block_tile_valid = output_valid;
    assign start_ready = scheduler_start_ready && !output_valid;
    assign busy = scheduler_busy || output_valid;
    assign done = block_tile_valid && block_tile_ready && block_group == 15;
    assign residual_replay_read_data_valid = ENABLE_RESIDUAL_REPLAY
        && !scheduler_busy && canvas_data_valid;
    assign residual_replay_read_q10_packed = canvas_data;

    attention_residual_canvas_uram residual_canvas (
        .clk(clk), .load_valid(canvas_load_valid),
        .load_group(canvas_load_group),
        .load_output_tile(canvas_load_output_tile),
        .load_data_packed(canvas_load_data),
        .read_valid(residual_canvas_read_valid),
        .read_group(residual_canvas_read_group),
        .read_output_tile(residual_canvas_read_output_tile),
        .read_data_valid(canvas_data_valid),
        .read_data_packed(canvas_data)
    );

    attention_projection_output_tile_scheduler #(
        .INTERNAL_MAC(INTERNAL_MAC),.GROUPED_CANVAS(GROUPED_CANVAS)
    ) projection (
        .clk(clk), .rst_n(rst_n), .start(start && start_ready),
        .start_ready(scheduler_start_ready),
        .output_tile_in(output_tile_in), .multipliers_packed(multipliers_packed),
        .weight_tile_valid(weight_tile_valid),
        .weight_tile_ready(weight_tile_ready),
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
        .projection_tile_valid(projection_valid),
        .projection_tile_ready(projection_ready),
        .projection_group(projection_group),
        .projection_output_tile(projection_output_tile),
        .projection_valid_channels(projection_valid_channels),
        .projection_q10_packed(projection_data),
        .array_request_valid(array_request_valid),
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
            output_valid <= 1'b0;
            block_group <= 0;
            block_output_tile <= 0;
            block_q10_packed <= 0;
        end else if (projection_ready) begin
            output_valid <= projection_valid;
            if (projection_valid) begin
                block_group <= projection_group;
                block_output_tile <= projection_output_tile;
                for (lane = 0; lane < 24; lane = lane + 1) begin
                    sum = $signed(projection_data[lane*24 +: 24])
                        + $signed(canvas_data[lane*24 +: 24]);
                    if (sum > 25'sd8388607)
                        block_q10_packed[lane*24 +: 24] <= 24'sh7fffff;
                    else if (sum < -25'sd8388608)
                        block_q10_packed[lane*24 +: 24] <= 24'sh800000;
                    else
                        block_q10_packed[lane*24 +: 24] <= sum[23:0];
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && projection_valid && !canvas_data_valid)
            $error("attention residual data was not prefetched");
        if (rst_n && ENABLE_RESIDUAL_REPLAY && scheduler_busy
            && residual_replay_read_valid)
            $error("attention residual replay overlapped projection");
`endif
    end

endmodule
