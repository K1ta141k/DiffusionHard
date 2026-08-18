`timescale 1ns/1ps

module mixed_precision_mac_tile_pipelined #(
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer STORAGE_WIDTH = 18,
    parameter integer ACC_WIDTH = 48,
    parameter integer TAG_WIDTH = 8
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire narrow_int8_mode,
    input  wire clear_accumulators,
    input  wire last_k_tile,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [M_LANES*32*STORAGE_WIDTH-1:0] activations_packed,
    input  wire [N_LANES*32*STORAGE_WIDTH-1:0] weights_packed,
    output wire valid_out,
    output wire [TAG_WIDTH-1:0] tag_out,
    output wire [M_LANES*N_LANES*ACC_WIDTH-1:0] accumulators_packed
);

    reg [M_LANES*32*STORAGE_WIDTH-1:0] adjusted_activations;
    reg [N_LANES*32*STORAGE_WIDTH-1:0] adjusted_weights;
    integer activation_index;
    integer weight_index;

    always @* begin
        adjusted_activations = activations_packed;
        adjusted_weights = weights_packed;
        if (narrow_int8_mode) begin
            for (activation_index = 0; activation_index < M_LANES*32;
                 activation_index = activation_index + 1)
                adjusted_activations[
                    activation_index*STORAGE_WIDTH +: STORAGE_WIDTH
                ] = {
                    {(STORAGE_WIDTH-8){activations_packed[
                        activation_index*STORAGE_WIDTH + 7
                    ]}},
                    activations_packed[
                        activation_index*STORAGE_WIDTH +: 8
                    ]
                };
            for (weight_index = 0; weight_index < N_LANES*32;
                 weight_index = weight_index + 1)
                adjusted_weights[
                    weight_index*STORAGE_WIDTH +: STORAGE_WIDTH
                ] = {
                    {(STORAGE_WIDTH-8){weights_packed[
                        weight_index*STORAGE_WIDTH + 7
                    ]}},
                    weights_packed[weight_index*STORAGE_WIDTH +: 8]
                };
        end
    end

    int8_mac_tile_pipelined #(
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .DATA_WIDTH(STORAGE_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .TAG_WIDTH(TAG_WIDTH)
    ) shared_mac (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .clear_accumulators(clear_accumulators),
        .last_k_tile(last_k_tile),
        .tag_in(tag_in),
        .activations_packed(adjusted_activations),
        .weights_packed(adjusted_weights),
        .valid_out(valid_out),
        .tag_out(tag_out),
        .accumulators_packed(accumulators_packed)
    );

endmodule
