`timescale 1ns/1ps

module mlp_interstage_pipeline #(
    parameter integer TOKENS = 64,
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer INPUT_SIZE = 3072,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer RIGHT_SHIFT = 20,
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10,
    parameter integer GROUP_WIDTH = ((TOKENS / M_LANES) <= 1)
        ? 1 : $clog2(TOKENS / M_LANES),
    parameter integer K_TILE_WIDTH = ((INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(INPUT_SIZE / 32)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    output wire ready_in,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile_in,
    input  wire [GROUP_WIDTH-1:0] group_in,
    input  wire [M_LANES*N_LANES*16-1:0] gelu_q10_packed,
    input  wire [N_LANES*MULTIPLIER_WIDTH-1:0]
        multipliers_packed,
    output wire activation_load_valid,
    output wire [GROUP_WIDTH-1:0] activation_load_group,
    output wire [K_TILE_WIDTH-1:0] activation_load_k_tile,
    output wire [M_LANES*32*8-1:0] activation_load_data,
    output wire done
);

    localparam integer LANES = M_LANES * N_LANES;
    localparam integer TAG_WIDTH = OUTPUT_TILE_TAG_WIDTH + GROUP_WIDTH;
    wire converted_valid;
    wire [TAG_WIDTH-1:0] converted_tag;
    wire [LANES*8-1:0] converted_values;
    wire [LANES*MULTIPLIER_WIDTH-1:0] expanded_multipliers;
    genvar token_index;
    genvar channel_index;

    generate
        for (token_index = 0; token_index < M_LANES;
             token_index = token_index + 1) begin : multiplier_tokens
            for (channel_index = 0; channel_index < N_LANES;
                 channel_index = channel_index + 1) begin : multiplier_channels
                assign expanded_multipliers[
                    (token_index*N_LANES + channel_index)*MULTIPLIER_WIDTH
                    +: MULTIPLIER_WIDTH
                ] = multipliers_packed[
                    channel_index*MULTIPLIER_WIDTH +: MULTIPLIER_WIDTH
                ];
            end
        end
    endgenerate

    smoothquant_int8_vector_serial #(
        .LANES(LANES),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .RIGHT_SHIFT(RIGHT_SHIFT),
        .TAG_WIDTH(TAG_WIDTH)
    ) converter (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .tag_in({output_tile_in, group_in}),
        .inputs_packed(gelu_q10_packed),
        .multipliers_packed(expanded_multipliers),
        .valid_out(converted_valid),
        .tag_out(converted_tag),
        .outputs_packed(converted_values)
    );

    mlp_interstage_tile_bridge_bram #(
        .TOKENS(TOKENS),
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .INPUT_SIZE(INPUT_SIZE),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .GROUP_WIDTH(GROUP_WIDTH),
        .K_TILE_WIDTH(K_TILE_WIDTH)
    ) bridge (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(converted_valid),
        .output_tile_in(converted_tag[
            GROUP_WIDTH +: OUTPUT_TILE_TAG_WIDTH
        ]),
        .group_in(converted_tag[GROUP_WIDTH-1:0]),
        .values_packed(converted_values),
        .activation_load_valid(activation_load_valid),
        .activation_load_group(activation_load_group),
        .activation_load_k_tile(activation_load_k_tile),
        .activation_load_data(activation_load_data),
        .done(done)
    );

endmodule
