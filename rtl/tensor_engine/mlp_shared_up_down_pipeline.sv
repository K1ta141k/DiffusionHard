`timescale 1ns/1ps

module mlp_shared_up_down_pipeline #(
    parameter integer TOKENS = 64,
    parameter integer UP_INPUT_SIZE = 768,
    parameter integer DOWN_INPUT_SIZE = 3072,
    parameter integer DOWN_OUTPUT_SIZE = 768,
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer TOKEN_FACTOR_WIDTH = 16,
    parameter integer OUTPUT_FACTOR_WIDTH = 18,
    parameter integer OUTPUT_WIDTH = 24,
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10,
    parameter integer UP_POSTPROCESS_PARALLEL4 = 0,
    parameter integer DOWN_SYNC_ACTIVATION_MEMORY = 0,
    parameter integer GROUP_WIDTH = ((TOKENS / M_LANES) <= 1)
        ? 1 : $clog2(TOKENS / M_LANES),
    parameter integer UP_K_TILE_WIDTH = ((UP_INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(UP_INPUT_SIZE / 32),
    parameter integer DOWN_K_TILE_WIDTH = ((DOWN_INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(DOWN_INPUT_SIZE / 32),
    parameter integer CLIENT_TAG_WIDTH = 1 + OUTPUT_TILE_TAG_WIDTH + GROUP_WIDTH
) (
    input  wire clk,
    input  wire rst_n,
    input  wire up_activation_load_valid,
    input  wire [GROUP_WIDTH-1:0] up_activation_load_group,
    input  wire [UP_K_TILE_WIDTH-1:0] up_activation_load_k_tile,
    input  wire [M_LANES*32*DATA_WIDTH-1:0] up_activation_load_data,
    input  wire up_weight_load_valid,
    input  wire up_weight_load_bank,
    input  wire [UP_K_TILE_WIDTH-1:0] up_weight_load_k_tile,
    input  wire [N_LANES*32*DATA_WIDTH-1:0] up_weight_load_data,
    output wire up_weight_load_ready,
    input  wire up_metadata_load_valid,
    input  wire up_metadata_load_bank,
    input  wire [N_LANES*OUTPUT_FACTOR_WIDTH-1:0]
        up_metadata_output_factors,
    input  wire [N_LANES*ACC_WIDTH-1:0] up_metadata_biases,
    input  wire [N_LANES*MULTIPLIER_WIDTH-1:0]
        up_metadata_interstage_multipliers,
    output wire up_metadata_load_ready,
    input  wire up_token_factor_load_valid,
    input  wire [GROUP_WIDTH-1:0] up_token_factor_load_group,
    input  wire [M_LANES*TOKEN_FACTOR_WIDTH-1:0]
        up_token_factor_load_factors,
    output wire up_token_factor_load_ready,
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
    input  wire [N_LANES*32*DATA_WIDTH-1:0] down_weight_load_data,
    output wire down_weight_load_ready,
    input  wire down_metadata_load_valid,
    input  wire down_metadata_load_bank,
    input  wire [M_LANES*N_LANES*MULTIPLIER_WIDTH-1:0]
        down_metadata_multipliers,
    input  wire [M_LANES*N_LANES*ACC_WIDTH-1:0] down_metadata_biases,
    output wire down_metadata_load_ready,
    input  wire down_residual_load_valid,
    input  wire [GROUP_WIDTH-1:0] down_residual_load_group,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0]
        down_residual_load_output_tile,
    input  wire [M_LANES*N_LANES*OUTPUT_WIDTH-1:0]
        down_residual_load_data,
    output wire down_residual_load_ready,
    input  wire down_start,
    input  wire down_start_bank,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] down_start_output_tile,
    output wire down_start_ready,
    output wire down_busy,
    output wire output_valid,
    output wire output_bank,
    output wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile,
    output wire [GROUP_WIDTH-1:0] output_group,
    output wire [M_LANES*N_LANES*OUTPUT_WIDTH-1:0] outputs_packed,
    output wire down_done,
    output wire array_request_valid,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [CLIENT_TAG_WIDTH:0] array_request_tag,
    output wire [M_LANES*32*DATA_WIDTH-1:0] array_request_activations,
    output wire [N_LANES*32*DATA_WIDTH-1:0] array_request_weights,
    input  wire array_response_valid,
    input  wire [CLIENT_TAG_WIDTH:0] array_response_tag,
    input  wire [M_LANES*N_LANES*ACC_WIDTH-1:0]
        array_response_accumulators
);

    wire down_activation_load_valid;
    wire [GROUP_WIDTH-1:0] down_activation_load_group;
    wire [DOWN_K_TILE_WIDTH-1:0] down_activation_load_k_tile;
    wire [M_LANES*32*DATA_WIDTH-1:0] down_activation_load_data;
    wire up_array_valid;
    wire up_array_clear;
    wire up_array_last;
    wire [CLIENT_TAG_WIDTH-1:0] up_array_tag;
    wire [M_LANES*32*DATA_WIDTH-1:0] up_array_activations;
    wire [N_LANES*32*DATA_WIDTH-1:0] up_array_weights;
    wire down_array_valid;
    wire down_array_clear;
    wire down_array_last;
    wire [CLIENT_TAG_WIDTH-1:0] down_array_tag;
    wire [M_LANES*32*DATA_WIDTH-1:0] down_array_activations;
    wire [N_LANES*32*DATA_WIDTH-1:0] down_array_weights;
    wire response_owner = array_response_tag[CLIENT_TAG_WIDTH];

    assign array_request_valid = up_array_valid || down_array_valid;
    assign array_request_clear = up_array_valid
        ? up_array_clear : down_array_clear;
    assign array_request_last = up_array_valid
        ? up_array_last : down_array_last;
    assign array_request_tag = up_array_valid
        ? {1'b0, up_array_tag} : {1'b1, down_array_tag};
    assign array_request_activations = up_array_valid
        ? up_array_activations : down_array_activations;
    assign array_request_weights = up_array_valid
        ? up_array_weights : down_array_weights;

    mlp_up_to_down_activation_pipeline #(
        .TOKENS(TOKENS), .UP_INPUT_SIZE(UP_INPUT_SIZE),
        .DOWN_INPUT_SIZE(DOWN_INPUT_SIZE), .M_LANES(M_LANES),
        .N_LANES(N_LANES), .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH), .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .TOKEN_FACTOR_WIDTH(TOKEN_FACTOR_WIDTH),
        .OUTPUT_FACTOR_WIDTH(OUTPUT_FACTOR_WIDTH),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH), .INTERNAL_MAC(0),
        .UP_POSTPROCESS_PARALLEL4(UP_POSTPROCESS_PARALLEL4),
        .GROUP_WIDTH(GROUP_WIDTH), .UP_K_TILE_WIDTH(UP_K_TILE_WIDTH),
        .DOWN_K_TILE_WIDTH(DOWN_K_TILE_WIDTH)
    ) up (
        .clk(clk), .rst_n(rst_n),
        .activation_load_valid(up_activation_load_valid),
        .activation_load_group(up_activation_load_group),
        .activation_load_k_tile(up_activation_load_k_tile),
        .activation_load_data(up_activation_load_data),
        .weight_load_valid(up_weight_load_valid),
        .weight_load_bank(up_weight_load_bank),
        .weight_load_k_tile(up_weight_load_k_tile),
        .weight_load_data(up_weight_load_data),
        .weight_load_ready(up_weight_load_ready),
        .metadata_load_valid(up_metadata_load_valid),
        .metadata_load_bank(up_metadata_load_bank),
        .metadata_load_output_factors(up_metadata_output_factors),
        .metadata_load_biases(up_metadata_biases),
        .metadata_load_interstage_multipliers(
            up_metadata_interstage_multipliers
        ), .metadata_load_ready(up_metadata_load_ready),
        .token_factor_load_valid(up_token_factor_load_valid),
        .token_factor_load_group(up_token_factor_load_group),
        .token_factor_load_factors(up_token_factor_load_factors),
        .token_factor_load_ready(up_token_factor_load_ready),
        .start(up_start), .start_bank(up_start_bank),
        .start_output_tile(up_start_output_tile),
        .start_ready(up_start_ready), .busy(up_busy),
        .down_activation_load_valid(down_activation_load_valid),
        .down_activation_load_group(down_activation_load_group),
        .down_activation_load_k_tile(down_activation_load_k_tile),
        .down_activation_load_data(down_activation_load_data),
        .array_request_valid(up_array_valid),
        .array_request_clear(up_array_clear),
        .array_request_last(up_array_last), .array_request_tag(up_array_tag),
        .array_request_activations(up_array_activations),
        .array_request_weights(up_array_weights),
        .array_response_valid(array_response_valid && !response_owner),
        .array_response_tag(array_response_tag[CLIENT_TAG_WIDTH-1:0]),
        .array_response_accumulators(array_response_accumulators),
        .up_done(up_tile_done), .done(up_all_activations_done)
    );

    mlp_down_pingpong_pipeline #(
        .TOKENS(TOKENS), .INPUT_SIZE(DOWN_INPUT_SIZE),
        .OUTPUT_SIZE(DOWN_OUTPUT_SIZE), .M_LANES(M_LANES),
        .N_LANES(N_LANES), .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH), .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH), .INTERNAL_MAC(0),
        .SYNC_ACTIVATION_MEMORY(DOWN_SYNC_ACTIVATION_MEMORY),
        .GROUP_WIDTH(GROUP_WIDTH), .K_TILE_WIDTH(DOWN_K_TILE_WIDTH)
    ) down (
        .clk(clk), .rst_n(rst_n),
        .activation_load_valid(down_activation_load_valid),
        .activation_load_group(down_activation_load_group),
        .activation_load_k_tile(down_activation_load_k_tile),
        .activation_load_data(down_activation_load_data),
        .weight_load_valid(down_weight_load_valid),
        .weight_load_bank(down_weight_load_bank),
        .weight_load_k_tile(down_weight_load_k_tile),
        .weight_load_data(down_weight_load_data),
        .weight_load_ready(down_weight_load_ready),
        .metadata_load_valid(down_metadata_load_valid),
        .metadata_load_bank(down_metadata_load_bank),
        .metadata_load_multipliers(down_metadata_multipliers),
        .metadata_load_biases(down_metadata_biases),
        .metadata_load_ready(down_metadata_load_ready),
        .residual_load_valid(down_residual_load_valid),
        .residual_load_group(down_residual_load_group),
        .residual_load_output_tile(down_residual_load_output_tile),
        .residual_load_data(down_residual_load_data),
        .residual_load_ready(down_residual_load_ready),
        .start(down_start), .start_bank(down_start_bank),
        .start_output_tile(down_start_output_tile),
        .start_ready(down_start_ready), .busy(down_busy),
        .valid_out(output_valid), .bank_out(output_bank),
        .output_tile_out(output_tile), .group_out(output_group),
        .outputs_packed(outputs_packed),
        .array_request_valid(down_array_valid),
        .array_request_clear(down_array_clear),
        .array_request_last(down_array_last),
        .array_request_tag(down_array_tag),
        .array_request_activations(down_array_activations),
        .array_request_weights(down_array_weights),
        .array_response_valid(array_response_valid && response_owner),
        .array_response_tag(array_response_tag[CLIENT_TAG_WIDTH-1:0]),
        .array_response_accumulators(array_response_accumulators),
        .done(down_done)
    );

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && up_array_valid && down_array_valid)
            $error("MLP up and down requested the shared array together");
`endif
    end

endmodule
