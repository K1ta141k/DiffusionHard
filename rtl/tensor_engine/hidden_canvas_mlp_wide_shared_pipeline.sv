`timescale 1ns/1ps

module hidden_canvas_mlp_wide_shared_pipeline #(
    parameter integer TOKENS = 64,
    parameter integer DOWN_INPUT_SIZE = 3072,
    parameter integer DOWN_OUTPUT_SIZE = 768,
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10,
    parameter integer DOWN_K_TILE_WIDTH = ((DOWN_INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(DOWN_INPUT_SIZE / 32),
    parameter integer CLIENT_TAG_WIDTH =
        1 + OUTPUT_TILE_TAG_WIDTH + (((TOKENS / 8) <= 1)
            ? 1 : $clog2(TOKENS / 8))
) (
    input  wire clk,
    input  wire rst_n,
    input  wire frontend_start,
    input  wire [3:0] frontend_group,
    input  wire [17:0] smoothing_reciprocal_q15,
    output wire [9:0] smoothing_reciprocal_channel,
    output wire frontend_start_ready,
    output wire frontend_busy,
    output wire frontend_done,
    output wire canvas_read_valid,
    output wire [3:0] canvas_read_group,
    output wire [6:0] canvas_read_output_tile,
    input  wire canvas_read_data_valid,
    input  wire [4*6*24-1:0] canvas_read_q10_packed,
    input  wire up_weight_load_valid,
    input  wire up_weight_load_bank,
    input  wire [4:0] up_weight_load_k_tile,
    input  wire [6*32*8-1:0] up_weight_load_data,
    output wire up_weight_load_ready,
    input  wire up_metadata_load_valid,
    input  wire up_metadata_load_bank,
    input  wire [6*18-1:0] up_metadata_output_factors,
    input  wire [6*32-1:0] up_metadata_biases,
    input  wire [6*24-1:0] up_metadata_interstage_multipliers,
    output wire up_metadata_load_ready,
    input  wire up_start,
    input  wire up_start_bank,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] up_start_output_tile,
    output wire up_start_ready,
    output wire up_busy,
    output wire up_tile_done,
    output wire up_all_activations_done,
    input  wire down_weight_load_valid,
    input  wire down_weight_load_bank,
    input  wire [DOWN_K_TILE_WIDTH-1:0] down_weight_load_k_tile,
    input  wire [6*32*8-1:0] down_weight_load_data,
    output wire down_weight_load_ready,
    input  wire down_metadata_load_valid,
    input  wire down_metadata_load_bank,
    input  wire [4*6*24-1:0] down_metadata_multipliers,
    input  wire [4*6*32-1:0] down_metadata_biases,
    output wire down_metadata_load_ready,
    input  wire down_residual_load_valid,
    input  wire [3:0] down_residual_load_group,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] down_residual_load_output_tile,
    input  wire [4*6*24-1:0] down_residual_load_data,
    output wire down_residual_load_ready,
    input  wire down_start,
    input  wire down_start_bank,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] down_start_output_tile,
    output wire down_start_ready,
    output wire down_busy,
    output wire output_valid,
    output wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile,
    output wire [3:0] output_group,
    output wire [4*6*24-1:0] outputs_packed,
    output wire down_done,
    output wire array_request_valid,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [CLIENT_TAG_WIDTH:0] array_request_tag,
    output wire [8*32*8-1:0] array_request_activations,
    output wire [6*32*8-1:0] array_request_weights,
    input  wire array_response_valid,
    input  wire [CLIENT_TAG_WIDTH:0] array_response_tag,
    input  wire [8*6*32-1:0] array_response_accumulators
);

    wire narrow_token_factor_valid;
    wire [3:0] narrow_token_factor_group;
    wire [4*16-1:0] narrow_token_factors;
    wire narrow_activation_valid;
    wire [3:0] narrow_activation_group;
    wire [4:0] narrow_activation_k_tile;
    wire [4*32*8-1:0] narrow_activation_data;
    wire wide_token_factor_valid;
    wire [2:0] wide_token_factor_group;
    wire [8*16-1:0] wide_token_factors;
    wire wide_activation_valid;
    wire [2:0] wide_activation_group;
    wire [4:0] wide_activation_k_tile;
    wire [8*32*8-1:0] wide_activation_data;
    wire token_factor_ready;
    wire wide_residual_valid;
    wire [2:0] wide_residual_group;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] wide_residual_output_tile;
    wire [8*6*24-1:0] wide_residual_data;
    wire wide_output_valid;
    wire [2:0] wide_output_group;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] wide_output_tile;
    wire [8*6*24-1:0] wide_outputs;
    wire wide_down_done;

    hidden_canvas_mlp_frontend frontend (
        .clk(clk), .rst_n(rst_n), .start(frontend_start),
        .group_in(frontend_group),
        .smoothing_reciprocal_q15(smoothing_reciprocal_q15),
        .smoothing_reciprocal_channel(smoothing_reciprocal_channel),
        .start_ready(frontend_start_ready),
        .canvas_read_valid(canvas_read_valid),
        .canvas_read_group(canvas_read_group),
        .canvas_read_output_tile(canvas_read_output_tile),
        .canvas_read_data_valid(canvas_read_data_valid),
        .canvas_read_q10_packed(canvas_read_q10_packed),
        .token_factor_valid(narrow_token_factor_valid),
        .token_factor_group(narrow_token_factor_group),
        .token_factors_packed(narrow_token_factors),
        .activation_load_valid(narrow_activation_valid),
        .activation_load_group(narrow_activation_group),
        .activation_load_k_tile(narrow_activation_k_tile),
        .activation_load_data(narrow_activation_data),
        .busy(frontend_busy), .done(frontend_done)
    );

    mlp_token_pair_input_adapter input_adapter (
        .clk(clk), .rst_n(rst_n),
        .activation_valid_in(narrow_activation_valid),
        .activation_group_in(narrow_activation_group),
        .activation_k_tile_in(narrow_activation_k_tile),
        .activation_data_in(narrow_activation_data),
        .token_factor_valid_in(narrow_token_factor_valid),
        .token_factor_group_in(narrow_token_factor_group),
        .token_factors_in(narrow_token_factors),
        .activation_valid_out(wide_activation_valid),
        .activation_group_out(wide_activation_group),
        .activation_k_tile_out(wide_activation_k_tile),
        .activation_data_out(wide_activation_data),
        .token_factor_valid_out(wide_token_factor_valid),
        .token_factor_group_out(wide_token_factor_group),
        .token_factors_out(wide_token_factors)
    );

    mlp_token_pair_residual_adapter #(
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH)
    ) residual_adapter (
        .clk(clk), .rst_n(rst_n), .valid_in(down_residual_load_valid),
        .group_in(down_residual_load_group),
        .output_tile_in(down_residual_load_output_tile),
        .data_in(down_residual_load_data), .valid_out(wide_residual_valid),
        .group_out(wide_residual_group),
        .output_tile_out(wide_residual_output_tile),
        .data_out(wide_residual_data)
    );

    mlp_shared_up_down_pipeline #(
        .TOKENS(TOKENS), .UP_INPUT_SIZE(768),
        .DOWN_INPUT_SIZE(DOWN_INPUT_SIZE),
        .DOWN_OUTPUT_SIZE(DOWN_OUTPUT_SIZE), .M_LANES(8), .N_LANES(6),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .UP_POSTPROCESS_PARALLEL4(1),
        .DOWN_SYNC_ACTIVATION_MEMORY(1), .GROUP_WIDTH(3),
        .DOWN_K_TILE_WIDTH(DOWN_K_TILE_WIDTH),
        .CLIENT_TAG_WIDTH(CLIENT_TAG_WIDTH)
    ) mlp (
        .clk(clk), .rst_n(rst_n),
        .up_activation_load_valid(wide_activation_valid),
        .up_activation_load_group(wide_activation_group),
        .up_activation_load_k_tile(wide_activation_k_tile),
        .up_activation_load_data(wide_activation_data),
        .up_weight_load_valid(up_weight_load_valid),
        .up_weight_load_bank(up_weight_load_bank),
        .up_weight_load_k_tile(up_weight_load_k_tile),
        .up_weight_load_data(up_weight_load_data),
        .up_weight_load_ready(up_weight_load_ready),
        .up_metadata_load_valid(up_metadata_load_valid),
        .up_metadata_load_bank(up_metadata_load_bank),
        .up_metadata_output_factors(up_metadata_output_factors),
        .up_metadata_biases(up_metadata_biases),
        .up_metadata_interstage_multipliers(up_metadata_interstage_multipliers),
        .up_metadata_load_ready(up_metadata_load_ready),
        .up_token_factor_load_valid(wide_token_factor_valid),
        .up_token_factor_load_group(wide_token_factor_group),
        .up_token_factor_load_factors(wide_token_factors),
        .up_token_factor_load_ready(token_factor_ready),
        .up_start(up_start), .up_start_bank(up_start_bank),
        .up_start_output_tile(up_start_output_tile),
        .up_start_ready(up_start_ready), .up_busy(up_busy),
        .up_tile_done(up_tile_done),
        .up_all_activations_done(up_all_activations_done),
        .down_weight_load_valid(down_weight_load_valid),
        .down_weight_load_bank(down_weight_load_bank),
        .down_weight_load_k_tile(down_weight_load_k_tile),
        .down_weight_load_data(down_weight_load_data),
        .down_weight_load_ready(down_weight_load_ready),
        .down_metadata_load_valid(down_metadata_load_valid),
        .down_metadata_load_bank(down_metadata_load_bank),
        .down_metadata_multipliers({
            down_metadata_multipliers, down_metadata_multipliers
        }),
        .down_metadata_biases({down_metadata_biases, down_metadata_biases}),
        .down_metadata_load_ready(down_metadata_load_ready),
        .down_residual_load_valid(wide_residual_valid),
        .down_residual_load_group(wide_residual_group),
        .down_residual_load_output_tile(wide_residual_output_tile),
        .down_residual_load_data(wide_residual_data),
        .down_residual_load_ready(down_residual_load_ready),
        .down_start(down_start), .down_start_bank(down_start_bank),
        .down_start_output_tile(down_start_output_tile),
        .down_start_ready(down_start_ready), .down_busy(down_busy),
        .output_valid(wide_output_valid), .output_bank(),
        .output_tile(wide_output_tile), .output_group(wide_output_group),
        .outputs_packed(wide_outputs), .down_done(wide_down_done),
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

    mlp_token_pair_output_serializer #(
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH)
    ) output_serializer (
        .clk(clk), .rst_n(rst_n), .valid_in(wide_output_valid),
        .done_in(wide_down_done), .group_in(wide_output_group),
        .output_tile_in(wide_output_tile), .data_in(wide_outputs),
        .valid_out(output_valid), .done_out(down_done),
        .group_out(output_group), .output_tile_out(output_tile),
        .data_out(outputs_packed)
    );

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && wide_token_factor_valid && !token_factor_ready)
            $error("wide MLP token-factor store was not ready");
`endif
    end

endmodule
