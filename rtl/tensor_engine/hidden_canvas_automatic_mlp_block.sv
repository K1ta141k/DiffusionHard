`timescale 1ns/1ps

module hidden_canvas_automatic_mlp_block #(
    parameter integer TOKENS = 64,
    parameter integer DOWN_INPUT_SIZE = 3072,
    parameter integer DOWN_OUTPUT_SIZE = 768,
    parameter integer MLP_M_LANES = 4,
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10,
    parameter integer GROUP_WIDTH = ((TOKENS / 4) <= 1)
        ? 1 : $clog2(TOKENS / 4),
    parameter integer DOWN_K_TILE_WIDTH = ((DOWN_INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(DOWN_INPUT_SIZE / 32),
    parameter integer CLIENT_TAG_WIDTH = 1 + OUTPUT_TILE_TAG_WIDTH
        + (((TOKENS / MLP_M_LANES) <= 1)
            ? 1 : $clog2(TOKENS / MLP_M_LANES))
) (
    input  wire clk,
    input  wire rst_n,
    input  wire block_start,
    output wire block_start_ready,
    input  wire [17:0] smoothing_reciprocal_q15,
    output wire [9:0] smoothing_reciprocal_channel,
    output wire busy,
    output wire done,

    output wire canvas_read_valid,
    output wire [3:0] canvas_read_group,
    output wire [6:0] canvas_read_output_tile,
    input  wire canvas_read_data_valid,
    input  wire [4*6*24-1:0] canvas_read_q10_packed,

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

    output wire array_request_valid,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [CLIENT_TAG_WIDTH:0] array_request_tag,
    output wire [MLP_M_LANES*32*8-1:0] array_request_activations,
    output wire [6*32*8-1:0] array_request_weights,
    input  wire array_response_valid,
    input  wire [CLIENT_TAG_WIDTH:0] array_response_tag,
    input  wire [MLP_M_LANES*6*32-1:0] array_response_accumulators
);

    localparam integer FRONTEND_GROUPS = TOKENS / 4;
    localparam integer UP_OUTPUT_TILES = DOWN_INPUT_SIZE / 6;
    localparam integer DOWN_OUTPUT_TILES = DOWN_OUTPUT_SIZE / 6;

    wire frontend_start;
    wire [3:0] frontend_group;
    wire frontend_start_ready;
    wire frontend_busy;
    wire frontend_done;
    wire frontend_canvas_read_valid;
    wire [3:0] frontend_canvas_read_group;
    wire [6:0] frontend_canvas_read_tile;
    wire residual_canvas_read_valid;
    wire [GROUP_WIDTH-1:0] residual_canvas_read_group;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] residual_canvas_read_tile;
    wire residual_loader_busy;

    wire up_load_enable;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] up_load_tile;
    wire up_load_bank;
    wire up_loader_command_ready;
    wire up_load_done;
    wire up_weight_load_valid;
    wire up_weight_load_bank;
    wire [4:0] up_weight_load_k_tile;
    wire [6*32*8-1:0] up_weight_load_data;
    wire up_weight_load_ready;
    wire up_metadata_load_valid;
    wire up_metadata_load_bank;
    wire [443:0] up_metadata_load_data;
    wire up_metadata_load_ready;
    wire up_start;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] up_start_tile;
    wire up_start_bank;
    wire up_start_ready;
    wire up_tile_done;
    wire up_all_activations_done;

    wire down_load_enable;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] down_load_tile;
    wire down_load_bank;
    wire down_tile_command_ready;
    wire down_residual_command_ready;
    wire down_command_fire = down_load_enable
        && down_tile_command_ready && down_residual_command_ready;
    wire down_tile_loader_done;
    wire down_residual_loader_done;
    reg down_tile_complete;
    reg down_residual_complete;
    wire down_load_done = (down_tile_complete || down_tile_loader_done)
        && (down_residual_complete || down_residual_loader_done);
    wire down_weight_load_valid;
    wire down_weight_load_bank;
    wire [DOWN_K_TILE_WIDTH-1:0] down_weight_load_k_tile;
    wire [6*32*8-1:0] down_weight_load_data;
    wire down_weight_load_ready;
    wire down_metadata_load_valid;
    wire down_metadata_load_bank;
    wire [1343:0] down_metadata_load_data;
    wire down_metadata_load_ready;
    wire down_residual_load_valid;
    wire [GROUP_WIDTH-1:0] down_residual_load_group;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] down_residual_load_tile;
    wire [4*6*24-1:0] down_residual_load_data;
    wire down_residual_load_ready;
    wire down_start;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] down_start_tile;
    wire down_start_bank;
    wire down_start_ready;
    wire down_done;

    assign requested_up_output_tile = up_load_tile;
    assign requested_up_bank = up_load_bank;
    assign requested_down_output_tile = down_load_tile;
    assign requested_down_bank = down_load_bank;
    assign canvas_read_valid = frontend_busy
        ? frontend_canvas_read_valid : residual_canvas_read_valid;
    assign canvas_read_group = frontend_busy
        ? frontend_canvas_read_group
        : {{(4-GROUP_WIDTH){1'b0}}, residual_canvas_read_group};
    assign canvas_read_output_tile = frontend_busy
        ? frontend_canvas_read_tile : residual_canvas_read_tile[6:0];

    mlp_block_controller #(
        .FRONTEND_GROUPS(FRONTEND_GROUPS),
        .UP_OUTPUT_TILES(UP_OUTPUT_TILES),
        .DOWN_OUTPUT_TILES(DOWN_OUTPUT_TILES),
        .TILE_WIDTH(OUTPUT_TILE_TAG_WIDTH)
    ) controller (
        .clk(clk), .rst_n(rst_n), .block_start(block_start),
        .block_start_ready(block_start_ready),
        .frontend_start(frontend_start), .frontend_group(frontend_group),
        .frontend_start_ready(frontend_start_ready),
        .frontend_done(frontend_done), .up_load_enable(up_load_enable),
        .up_load_tile(up_load_tile), .up_load_bank(up_load_bank),
        .up_load_done(up_load_done), .up_start(up_start),
        .up_start_tile(up_start_tile), .up_start_bank(up_start_bank),
        .up_start_ready(up_start_ready), .up_tile_done(up_tile_done),
        .up_all_activations_done(up_all_activations_done),
        .down_load_enable(down_load_enable),
        .down_load_tile(down_load_tile), .down_load_bank(down_load_bank),
        .down_load_done(down_load_done), .down_start(down_start),
        .down_start_tile(down_start_tile),
        .down_start_bank(down_start_bank),
        .down_start_ready(down_start_ready), .down_tile_done(down_done),
        .busy(busy), .done(done)
    );

    mlp_tile_load_sequencer #(
        .INPUT_SIZE(768), .METADATA_WIDTH(444)
    ) up_loader (
        .clk(clk), .rst_n(rst_n), .command_valid(up_load_enable),
        .command_bank(up_load_bank), .command_ready(up_loader_command_ready),
        .weight_stream_valid(up_weight_stream_valid),
        .weight_stream_ready(up_weight_stream_ready),
        .weight_stream_data(up_weight_stream_data),
        .metadata_stream_valid(up_metadata_stream_valid),
        .metadata_stream_ready(up_metadata_stream_ready),
        .metadata_stream_data(up_metadata_stream_data),
        .weight_load_valid(up_weight_load_valid),
        .weight_load_bank(up_weight_load_bank),
        .weight_load_k_tile(up_weight_load_k_tile),
        .weight_load_data(up_weight_load_data),
        .weight_load_ready(up_weight_load_ready),
        .metadata_load_valid(up_metadata_load_valid),
        .metadata_load_bank(up_metadata_load_bank),
        .metadata_load_data(up_metadata_load_data),
        .metadata_load_ready(up_metadata_load_ready),
        .busy(), .done(up_load_done)
    );

    mlp_tile_load_sequencer #(
        .INPUT_SIZE(DOWN_INPUT_SIZE), .METADATA_WIDTH(1344),
        .K_TILE_WIDTH(DOWN_K_TILE_WIDTH)
    ) down_loader (
        .clk(clk), .rst_n(rst_n), .command_valid(down_command_fire),
        .command_bank(down_load_bank),
        .command_ready(down_tile_command_ready),
        .weight_stream_valid(down_weight_stream_valid),
        .weight_stream_ready(down_weight_stream_ready),
        .weight_stream_data(down_weight_stream_data),
        .metadata_stream_valid(down_metadata_stream_valid),
        .metadata_stream_ready(down_metadata_stream_ready),
        .metadata_stream_data(down_metadata_stream_data),
        .weight_load_valid(down_weight_load_valid),
        .weight_load_bank(down_weight_load_bank),
        .weight_load_k_tile(down_weight_load_k_tile),
        .weight_load_data(down_weight_load_data),
        .weight_load_ready(down_weight_load_ready),
        .metadata_load_valid(down_metadata_load_valid),
        .metadata_load_bank(down_metadata_load_bank),
        .metadata_load_data(down_metadata_load_data),
        .metadata_load_ready(down_metadata_load_ready),
        .busy(), .done(down_tile_loader_done)
    );

    hidden_canvas_residual_load_sequencer #(
        .TOKEN_GROUPS(FRONTEND_GROUPS),
        .OUTPUT_TILE_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .GROUP_WIDTH(GROUP_WIDTH)
    ) residual_loader (
        .clk(clk), .rst_n(rst_n), .command_valid(down_command_fire),
        .command_output_tile(down_load_tile),
        .command_ready(down_residual_command_ready),
        .canvas_read_valid(residual_canvas_read_valid),
        .canvas_read_group(residual_canvas_read_group),
        .canvas_read_output_tile(residual_canvas_read_tile),
        .canvas_read_data_valid(
            canvas_read_data_valid && residual_loader_busy
        ), .canvas_read_data(canvas_read_q10_packed),
        .residual_load_valid(down_residual_load_valid),
        .residual_load_group(down_residual_load_group),
        .residual_load_output_tile(down_residual_load_tile),
        .residual_load_data(down_residual_load_data),
        .residual_load_ready(down_residual_load_ready),
        .busy(residual_loader_busy), .done(down_residual_loader_done)
    );

    generate
    if (MLP_M_LANES == 4) begin : narrow_datapath
    hidden_canvas_mlp_shared_pipeline #(
        .TOKENS(TOKENS), .DOWN_INPUT_SIZE(DOWN_INPUT_SIZE),
        .DOWN_OUTPUT_SIZE(DOWN_OUTPUT_SIZE),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .DOWN_SYNC_ACTIVATION_MEMORY(1),
        .GROUP_WIDTH(GROUP_WIDTH),
        .DOWN_K_TILE_WIDTH(DOWN_K_TILE_WIDTH),
        .CLIENT_TAG_WIDTH(CLIENT_TAG_WIDTH)
    ) datapath (
        .clk(clk), .rst_n(rst_n), .frontend_start(frontend_start),
        .frontend_group(frontend_group),
        .smoothing_reciprocal_q15(smoothing_reciprocal_q15),
        .smoothing_reciprocal_channel(smoothing_reciprocal_channel),
        .frontend_start_ready(frontend_start_ready),
        .frontend_busy(frontend_busy), .frontend_done(frontend_done),
        .canvas_read_valid(frontend_canvas_read_valid),
        .canvas_read_group(frontend_canvas_read_group),
        .canvas_read_output_tile(frontend_canvas_read_tile),
        .canvas_read_data_valid(canvas_read_data_valid && frontend_busy),
        .canvas_read_q10_packed(canvas_read_q10_packed),
        .up_weight_load_valid(up_weight_load_valid),
        .up_weight_load_bank(up_weight_load_bank),
        .up_weight_load_k_tile(up_weight_load_k_tile),
        .up_weight_load_data(up_weight_load_data),
        .up_weight_load_ready(up_weight_load_ready),
        .up_metadata_load_valid(up_metadata_load_valid),
        .up_metadata_load_bank(up_metadata_load_bank),
        .up_metadata_output_factors(up_metadata_load_data[107:0]),
        .up_metadata_biases(up_metadata_load_data[299:108]),
        .up_metadata_interstage_multipliers(
            up_metadata_load_data[443:300]
        ), .up_metadata_load_ready(up_metadata_load_ready),
        .up_start(up_start), .up_start_bank(up_start_bank),
        .up_start_output_tile(up_start_tile),
        .up_start_ready(up_start_ready), .up_busy(),
        .up_tile_done(up_tile_done),
        .up_all_activations_done(up_all_activations_done),
        .down_weight_load_valid(down_weight_load_valid),
        .down_weight_load_bank(down_weight_load_bank),
        .down_weight_load_k_tile(down_weight_load_k_tile),
        .down_weight_load_data(down_weight_load_data),
        .down_weight_load_ready(down_weight_load_ready),
        .down_metadata_load_valid(down_metadata_load_valid),
        .down_metadata_load_bank(down_metadata_load_bank),
        .down_metadata_multipliers(down_metadata_load_data[575:0]),
        .down_metadata_biases(down_metadata_load_data[1343:576]),
        .down_metadata_load_ready(down_metadata_load_ready),
        .down_residual_load_valid(down_residual_load_valid),
        .down_residual_load_group(down_residual_load_group),
        .down_residual_load_output_tile(down_residual_load_tile),
        .down_residual_load_data(down_residual_load_data),
        .down_residual_load_ready(down_residual_load_ready),
        .down_start(down_start), .down_start_bank(down_start_bank),
        .down_start_output_tile(down_start_tile),
        .down_start_ready(down_start_ready), .down_busy(),
        .output_valid(output_valid), .output_bank(),
        .output_tile(output_tile), .output_group(output_group),
        .outputs_packed(outputs_packed), .down_done(down_done),
        .array_request_valid(array_request_valid),
        .array_request_clear(array_request_clear),
        .array_request_last(array_request_last),
        .array_request_tag(array_request_tag),
        .array_request_activations(array_request_activations),
        .array_request_weights(array_request_weights),
        .array_response_valid(array_response_valid),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators)
    );
    end else begin : wide_datapath
    hidden_canvas_mlp_wide_shared_pipeline #(
        .TOKENS(TOKENS), .DOWN_INPUT_SIZE(DOWN_INPUT_SIZE),
        .DOWN_OUTPUT_SIZE(DOWN_OUTPUT_SIZE),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .DOWN_K_TILE_WIDTH(DOWN_K_TILE_WIDTH),
        .CLIENT_TAG_WIDTH(CLIENT_TAG_WIDTH)
    ) datapath (
        .clk(clk), .rst_n(rst_n), .frontend_start(frontend_start),
        .frontend_group(frontend_group),
        .smoothing_reciprocal_q15(smoothing_reciprocal_q15),
        .smoothing_reciprocal_channel(smoothing_reciprocal_channel),
        .frontend_start_ready(frontend_start_ready),
        .frontend_busy(frontend_busy), .frontend_done(frontend_done),
        .canvas_read_valid(frontend_canvas_read_valid),
        .canvas_read_group(frontend_canvas_read_group),
        .canvas_read_output_tile(frontend_canvas_read_tile),
        .canvas_read_data_valid(canvas_read_data_valid && frontend_busy),
        .canvas_read_q10_packed(canvas_read_q10_packed),
        .up_weight_load_valid(up_weight_load_valid),
        .up_weight_load_bank(up_weight_load_bank),
        .up_weight_load_k_tile(up_weight_load_k_tile),
        .up_weight_load_data(up_weight_load_data),
        .up_weight_load_ready(up_weight_load_ready),
        .up_metadata_load_valid(up_metadata_load_valid),
        .up_metadata_load_bank(up_metadata_load_bank),
        .up_metadata_output_factors(up_metadata_load_data[107:0]),
        .up_metadata_biases(up_metadata_load_data[299:108]),
        .up_metadata_interstage_multipliers(up_metadata_load_data[443:300]),
        .up_metadata_load_ready(up_metadata_load_ready),
        .up_start(up_start), .up_start_bank(up_start_bank),
        .up_start_output_tile(up_start_tile),
        .up_start_ready(up_start_ready), .up_busy(),
        .up_tile_done(up_tile_done),
        .up_all_activations_done(up_all_activations_done),
        .down_weight_load_valid(down_weight_load_valid),
        .down_weight_load_bank(down_weight_load_bank),
        .down_weight_load_k_tile(down_weight_load_k_tile),
        .down_weight_load_data(down_weight_load_data),
        .down_weight_load_ready(down_weight_load_ready),
        .down_metadata_load_valid(down_metadata_load_valid),
        .down_metadata_load_bank(down_metadata_load_bank),
        .down_metadata_multipliers(down_metadata_load_data[575:0]),
        .down_metadata_biases(down_metadata_load_data[1343:576]),
        .down_metadata_load_ready(down_metadata_load_ready),
        .down_residual_load_valid(down_residual_load_valid),
        .down_residual_load_group(down_residual_load_group),
        .down_residual_load_output_tile(down_residual_load_tile),
        .down_residual_load_data(down_residual_load_data),
        .down_residual_load_ready(down_residual_load_ready),
        .down_start(down_start), .down_start_bank(down_start_bank),
        .down_start_output_tile(down_start_tile),
        .down_start_ready(down_start_ready), .down_busy(),
        .output_valid(output_valid), .output_tile(output_tile),
        .output_group(output_group), .outputs_packed(outputs_packed),
        .down_done(down_done), .array_request_valid(array_request_valid),
        .array_request_clear(array_request_clear),
        .array_request_last(array_request_last),
        .array_request_tag(array_request_tag),
        .array_request_activations(array_request_activations),
        .array_request_weights(array_request_weights),
        .array_response_valid(array_response_valid),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators)
    );
    end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            down_tile_complete <= 1'b0;
            down_residual_complete <= 1'b0;
        end else if (down_command_fire) begin
            down_tile_complete <= 1'b0;
            down_residual_complete <= 1'b0;
        end else if (down_load_done) begin
            down_tile_complete <= 1'b0;
            down_residual_complete <= 1'b0;
        end else begin
            if (down_tile_loader_done)
                down_tile_complete <= 1'b1;
            if (down_residual_loader_done)
                down_residual_complete <= 1'b1;
        end
    end

    initial begin
        if (TOKENS < 4 || TOKENS % 4 != 0 || TOKENS > 64)
            $error("automatic MLP block supports 4 through 64 tokens");
        if (DOWN_INPUT_SIZE % 6 != 0 || DOWN_OUTPUT_SIZE % 6 != 0)
            $error("automatic MLP dimensions must align to six lanes");
        if (MLP_M_LANES != 4 && MLP_M_LANES != 8)
            $error("automatic MLP supports four or eight physical token lanes");
    end

endmodule
