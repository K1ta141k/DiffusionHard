`timescale 1ns/1ps

module attention_residual_canvas_uram #(
    parameter integer DATA_WIDTH = 24,
    parameter integer LANES = 24,
    parameter integer DEPTH = 2048
) (
    input  wire clk,
    input  wire load_valid,
    input  wire [3:0] load_group,
    input  wire [6:0] load_output_tile,
    input  wire [LANES*DATA_WIDTH-1:0] load_data_packed,
    input  wire read_valid,
    input  wire [3:0] read_group,
    input  wire [6:0] read_output_tile,
    output reg  read_data_valid,
    output wire [LANES*DATA_WIDTH-1:0] read_data_packed
);

    wire [10:0] load_address = {load_group, load_output_tile};
    wire [10:0] read_address = {read_group, read_output_tile};
    genvar lane;

    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : residual_lanes
            (* ram_style = "ultra" *) reg [DATA_WIDTH-1:0] memory [0:DEPTH-1];
            reg [DATA_WIDTH-1:0] read_value;
            always @(posedge clk) begin
                if (load_valid)
                    memory[load_address] <= load_data_packed[
                        lane*DATA_WIDTH +: DATA_WIDTH
                    ];
                if (read_valid)
                    read_value <= memory[read_address];
            end
            assign read_data_packed[lane*DATA_WIDTH +: DATA_WIDTH] = read_value;
        end
    endgenerate

    always @(posedge clk)
        read_data_valid <= read_valid;

endmodule
