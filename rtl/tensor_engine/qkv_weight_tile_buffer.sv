`timescale 1ns/1ps

module qkv_weight_tile_buffer (
    input  wire clk,
    input  wire write_valid,
    input  wire [4:0] write_tile,
    input  wire [6*32*16-1:0] write_weights_packed,
    input  wire [4:0] read_tile,
    output wire [6*32*16-1:0] read_weights_packed
);

    genvar lane;
    generate
        for (lane = 0; lane < 6*32; lane = lane + 1) begin : weight_lanes
            (* ram_style = "distributed" *) reg [15:0] memory [0:23];
            always @(posedge clk)
                if (write_valid)
                    memory[write_tile] <= write_weights_packed[lane*16 +: 16];
            assign read_weights_packed[lane*16 +: 16] = memory[read_tile];
        end
    endgenerate

endmodule
