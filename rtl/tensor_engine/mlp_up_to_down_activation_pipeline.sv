`timescale 1ns/1ps

module mlp_up_to_down_activation_pipeline #(
    parameter integer TOKENS = 64,
    parameter integer UP_INPUT_SIZE = 768,
    parameter integer DOWN_INPUT_SIZE = 3072,
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer TOKEN_FACTOR_WIDTH = 16,
    parameter integer OUTPUT_FACTOR_WIDTH = 18,
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10,
    parameter integer INTERNAL_MAC = 1,
    parameter integer UP_POSTPROCESS_PARALLEL4 = 0,
    parameter integer GROUP_WIDTH = ((TOKENS / M_LANES) <= 1)
        ? 1 : $clog2(TOKENS / M_LANES),
    parameter integer UP_K_TILE_WIDTH = ((UP_INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(UP_INPUT_SIZE / 32),
    parameter integer DOWN_K_TILE_WIDTH = ((DOWN_INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(DOWN_INPUT_SIZE / 32)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire activation_load_valid,
    input  wire [GROUP_WIDTH-1:0] activation_load_group,
    input  wire [UP_K_TILE_WIDTH-1:0] activation_load_k_tile,
    input  wire [M_LANES*32*DATA_WIDTH-1:0] activation_load_data,
    input  wire weight_load_valid,
    input  wire weight_load_bank,
    input  wire [UP_K_TILE_WIDTH-1:0] weight_load_k_tile,
    input  wire [N_LANES*32*DATA_WIDTH-1:0] weight_load_data,
    output wire weight_load_ready,
    input  wire metadata_load_valid,
    input  wire metadata_load_bank,
    input  wire [N_LANES*OUTPUT_FACTOR_WIDTH-1:0]
        metadata_load_output_factors,
    input  wire [N_LANES*ACC_WIDTH-1:0] metadata_load_biases,
    input  wire [N_LANES*MULTIPLIER_WIDTH-1:0]
        metadata_load_interstage_multipliers,
    output wire metadata_load_ready,
    input  wire token_factor_load_valid,
    input  wire [GROUP_WIDTH-1:0] token_factor_load_group,
    input  wire [M_LANES*TOKEN_FACTOR_WIDTH-1:0]
        token_factor_load_factors,
    output wire token_factor_load_ready,
    input  wire start,
    input  wire start_bank,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] start_output_tile,
    output wire start_ready,
    output wire busy,
    output wire down_activation_load_valid,
    output wire [GROUP_WIDTH-1:0] down_activation_load_group,
    output wire [DOWN_K_TILE_WIDTH-1:0] down_activation_load_k_tile,
    output wire [M_LANES*32*DATA_WIDTH-1:0] down_activation_load_data,
    output wire array_request_valid,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [OUTPUT_TILE_TAG_WIDTH+GROUP_WIDTH:0] array_request_tag,
    output wire [M_LANES*32*DATA_WIDTH-1:0] array_request_activations,
    output wire [N_LANES*32*DATA_WIDTH-1:0] array_request_weights,
    input  wire array_response_valid,
    input  wire [OUTPUT_TILE_TAG_WIDTH+GROUP_WIDTH:0] array_response_tag,
    input  wire [M_LANES*N_LANES*ACC_WIDTH-1:0]
        array_response_accumulators,
    output wire up_done,
    output wire done
);

    wire up_valid;
    wire up_bank;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] up_output_tile;
    wire [GROUP_WIDTH-1:0] up_group;
    wire [M_LANES*N_LANES*16-1:0] up_gelu;
    wire [N_LANES*MULTIPLIER_WIDTH-1:0] up_interstage_multipliers;
    wire interstage_ready;

    mlp_up_pingpong_pipeline #(
        .TOKENS(TOKENS),
        .INPUT_SIZE(UP_INPUT_SIZE),
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .TOKEN_FACTOR_WIDTH(TOKEN_FACTOR_WIDTH),
        .OUTPUT_FACTOR_WIDTH(OUTPUT_FACTOR_WIDTH),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .INTERNAL_MAC(INTERNAL_MAC),
        .POSTPROCESS_PARALLEL4(UP_POSTPROCESS_PARALLEL4),
        .GROUP_WIDTH(GROUP_WIDTH),
        .K_TILE_WIDTH(UP_K_TILE_WIDTH)
    ) up_pipeline (
        .clk(clk),
        .rst_n(rst_n),
        .activation_load_valid(activation_load_valid),
        .activation_load_group(activation_load_group),
        .activation_load_k_tile(activation_load_k_tile),
        .activation_load_data(activation_load_data),
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
        .token_factor_load_valid(token_factor_load_valid),
        .token_factor_load_group(token_factor_load_group),
        .token_factor_load_factors(token_factor_load_factors),
        .token_factor_load_ready(token_factor_load_ready),
        .start(start),
        .start_bank(start_bank),
        .start_output_tile(start_output_tile),
        .start_ready(start_ready),
        .busy(busy),
        .valid_out(up_valid),
        .bank_out(up_bank),
        .output_tile_out(up_output_tile),
        .group_out(up_group),
        .gelu_packed(up_gelu),
        .interstage_multipliers_out(up_interstage_multipliers),
        .array_request_valid(array_request_valid),
        .array_request_clear(array_request_clear),
        .array_request_last(array_request_last),
        .array_request_tag(array_request_tag),
        .array_request_activations(array_request_activations),
        .array_request_weights(array_request_weights),
        .array_response_valid(array_response_valid),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators),
        .done(up_done)
    );

    mlp_interstage_pipeline #(
        .TOKENS(TOKENS),
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .INPUT_SIZE(DOWN_INPUT_SIZE),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .GROUP_WIDTH(GROUP_WIDTH),
        .K_TILE_WIDTH(DOWN_K_TILE_WIDTH)
    ) interstage (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(up_valid),
        .ready_in(interstage_ready),
        .output_tile_in(up_output_tile),
        .group_in(up_group),
        .gelu_q10_packed(up_gelu),
        .multipliers_packed(up_interstage_multipliers),
        .activation_load_valid(down_activation_load_valid),
        .activation_load_group(down_activation_load_group),
        .activation_load_k_tile(down_activation_load_k_tile),
        .activation_load_data(down_activation_load_data),
        .done(done)
    );

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && up_valid && !interstage_ready)
            $error("up-to-down interstage queue overflow");
`endif
    end

endmodule
