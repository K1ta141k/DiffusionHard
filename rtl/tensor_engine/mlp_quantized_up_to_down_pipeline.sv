`timescale 1ns/1ps

module mlp_quantized_up_to_down_pipeline #(
    parameter integer TOKENS = 64,
    parameter integer UP_INPUT_SIZE = 768,
    parameter integer DOWN_INPUT_SIZE = 3072,
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer ACC_WIDTH = 32,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer TOKEN_FACTOR_WIDTH = 16,
    parameter integer OUTPUT_FACTOR_WIDTH = 18,
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10,
    parameter integer INTERNAL_MAC = 1,
    parameter integer GROUP_WIDTH = ((TOKENS / M_LANES) <= 1)
        ? 1 : $clog2(TOKENS / M_LANES),
    parameter integer UP_K_TILE_WIDTH = ((UP_INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(UP_INPUT_SIZE / 32),
    parameter integer DOWN_K_TILE_WIDTH = ((DOWN_INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(DOWN_INPUT_SIZE / 32)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire quantizer_start,
    input  wire [GROUP_WIDTH-1:0] quantizer_group,
    output wire quantizer_start_ready,
    input  wire quantizer_start_pass2,
    output wire quantizer_pass2_ready,
    input  wire normalized_input_valid,
    output wire normalized_input_ready,
    input  wire [M_LANES*18-1:0] normalized_q12_packed,
    input  wire [17:0] smoothing_reciprocal_q15,
    output wire quantizer_busy,
    output wire quantizer_done,
    input  wire weight_load_valid,
    input  wire weight_load_bank,
    input  wire [UP_K_TILE_WIDTH-1:0] weight_load_k_tile,
    input  wire [N_LANES*32*8-1:0] weight_load_data,
    output wire weight_load_ready,
    input  wire metadata_load_valid,
    input  wire metadata_load_bank,
    input  wire [N_LANES*OUTPUT_FACTOR_WIDTH-1:0]
        metadata_load_output_factors,
    input  wire [N_LANES*ACC_WIDTH-1:0] metadata_load_biases,
    input  wire [N_LANES*MULTIPLIER_WIDTH-1:0]
        metadata_load_interstage_multipliers,
    output wire metadata_load_ready,
    input  wire start,
    input  wire start_bank,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] start_output_tile,
    output wire start_ready,
    output wire busy,
    output wire down_activation_load_valid,
    output wire [GROUP_WIDTH-1:0] down_activation_load_group,
    output wire [DOWN_K_TILE_WIDTH-1:0] down_activation_load_k_tile,
    output wire [M_LANES*32*8-1:0] down_activation_load_data,
    output wire array_request_valid,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [OUTPUT_TILE_TAG_WIDTH+GROUP_WIDTH:0] array_request_tag,
    output wire [M_LANES*32*8-1:0] array_request_activations,
    output wire [N_LANES*32*8-1:0] array_request_weights,
    input  wire array_response_valid,
    input  wire [OUTPUT_TILE_TAG_WIDTH+GROUP_WIDTH:0] array_response_tag,
    input  wire [M_LANES*N_LANES*ACC_WIDTH-1:0]
        array_response_accumulators,
    output wire up_done,
    output wire done
);

    wire quantized_activation_valid;
    wire [GROUP_WIDTH-1:0] quantized_activation_group;
    wire [UP_K_TILE_WIDTH-1:0] quantized_activation_k_tile;
    wire [M_LANES*32*8-1:0] quantized_activation_data;
    wire token_factor_valid;
    wire [GROUP_WIDTH-1:0] token_factor_group;
    wire [M_LANES*TOKEN_FACTOR_WIDTH-1:0] token_factors;
    wire token_factor_load_ready;

    mlp_up_activation_quantizer #(
        .INPUT_SIZE(UP_INPUT_SIZE),
        .M_LANES(M_LANES),
        .TOKEN_FACTOR_WIDTH(TOKEN_FACTOR_WIDTH),
        .GROUP_WIDTH(GROUP_WIDTH),
        .K_TILE_WIDTH(UP_K_TILE_WIDTH)
    ) activation_quantizer (
        .clk(clk),
        .rst_n(rst_n),
        .start(quantizer_start),
        .group_in(quantizer_group),
        .start_ready(quantizer_start_ready),
        .start_pass2(quantizer_start_pass2),
        .pass2_ready(quantizer_pass2_ready),
        .input_valid(normalized_input_valid),
        .input_ready(normalized_input_ready),
        .normalized_q12_packed(normalized_q12_packed),
        .smoothing_reciprocal_q15(smoothing_reciprocal_q15),
        .token_factor_valid(token_factor_valid),
        .token_factor_group(token_factor_group),
        .token_factors_packed(token_factors),
        .activation_load_valid(quantized_activation_valid),
        .activation_load_group(quantized_activation_group),
        .activation_load_k_tile(quantized_activation_k_tile),
        .activation_load_data(quantized_activation_data),
        .busy(quantizer_busy),
        .done(quantizer_done)
    );

    mlp_up_to_down_activation_pipeline #(
        .TOKENS(TOKENS),
        .UP_INPUT_SIZE(UP_INPUT_SIZE),
        .DOWN_INPUT_SIZE(DOWN_INPUT_SIZE),
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .ACC_WIDTH(ACC_WIDTH),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .TOKEN_FACTOR_WIDTH(TOKEN_FACTOR_WIDTH),
        .OUTPUT_FACTOR_WIDTH(OUTPUT_FACTOR_WIDTH),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .INTERNAL_MAC(INTERNAL_MAC),
        .GROUP_WIDTH(GROUP_WIDTH),
        .UP_K_TILE_WIDTH(UP_K_TILE_WIDTH),
        .DOWN_K_TILE_WIDTH(DOWN_K_TILE_WIDTH)
    ) up_to_down (
        .clk(clk),
        .rst_n(rst_n),
        .activation_load_valid(quantized_activation_valid),
        .activation_load_group(quantized_activation_group),
        .activation_load_k_tile(quantized_activation_k_tile),
        .activation_load_data(quantized_activation_data),
        .weight_load_valid(weight_load_valid),
        .weight_load_bank(weight_load_bank),
        .weight_load_k_tile(weight_load_k_tile),
        .weight_load_data(weight_load_data),
        .weight_load_ready(weight_load_ready),
        .metadata_load_valid(metadata_load_valid),
        .metadata_load_bank(metadata_load_bank),
        .metadata_load_output_factors(metadata_load_output_factors),
        .metadata_load_biases(metadata_load_biases),
        .metadata_load_interstage_multipliers(
            metadata_load_interstage_multipliers
        ),
        .metadata_load_ready(metadata_load_ready),
        .token_factor_load_valid(token_factor_valid),
        .token_factor_load_group(token_factor_group),
        .token_factor_load_factors(token_factors),
        .token_factor_load_ready(token_factor_load_ready),
        .start(start),
        .start_bank(start_bank),
        .start_output_tile(start_output_tile),
        .start_ready(start_ready),
        .busy(busy),
        .down_activation_load_valid(down_activation_load_valid),
        .down_activation_load_group(down_activation_load_group),
        .down_activation_load_k_tile(down_activation_load_k_tile),
        .down_activation_load_data(down_activation_load_data),
        .array_request_valid(array_request_valid),
        .array_request_clear(array_request_clear),
        .array_request_last(array_request_last),
        .array_request_tag(array_request_tag),
        .array_request_activations(array_request_activations),
        .array_request_weights(array_request_weights),
        .array_response_valid(array_response_valid),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators),
        .up_done(up_done),
        .done(done)
    );

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && token_factor_valid && !token_factor_load_ready)
            $error("runtime token-factor sink was not ready");
`endif
    end

endmodule
