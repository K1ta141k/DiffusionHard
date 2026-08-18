`timescale 1ns/1ps

module attention_projection_weight_tile_buffer #(
    parameter integer OUTPUT_LANES = 6,
    parameter integer K_LANES = 32,
    parameter integer K_TILES = 24
) (
    input  wire clk,
    input  wire write_valid,
    input  wire [4:0] write_tile,
    input  wire [OUTPUT_LANES*K_LANES*8-1:0] write_weights_packed,
    input  wire [4:0] read_tile,
    output wire [OUTPUT_LANES*K_LANES*8-1:0] read_weights_packed
);

    genvar lane;
    generate
        for (lane = 0; lane < OUTPUT_LANES*K_LANES;
             lane = lane + 1) begin : weight_lanes
            (* ram_style = "distributed" *) reg [7:0] memory [0:K_TILES-1];
            always @(posedge clk)
                if (write_valid)
                    memory[write_tile] <= write_weights_packed[lane*8 +: 8];
            assign read_weights_packed[lane*8 +: 8] = memory[read_tile];
        end
    endgenerate

endmodule
