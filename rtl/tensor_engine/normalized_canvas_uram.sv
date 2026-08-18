`timescale 1ns/1ps

module normalized_canvas_uram (
    input  wire clk,
    input  wire load_valid,
    input  wire [3:0] load_group,
    input  wire [9:0] load_channel,
    input  wire [4*18-1:0] load_q12_packed,
    input  wire read_valid,
    input  wire [3:0] read_group,
    input  wire [4:0] read_input_tile,
    output reg  read_data_valid,
    output wire [4*32*18-1:0] read_q12_packed
);

    wire [4:0] load_bank = load_channel[4:0];
    wire [4:0] load_input_tile = load_channel[9:5];
    wire [8:0] load_address = {load_group, load_input_tile};
    wire [8:0] read_address = {read_group, read_input_tile};
    genvar bank;

    generate
        for (bank = 0; bank < 32; bank = bank + 1) begin : channel_banks
            (* ram_style = "ultra" *) reg [4*18-1:0] memory [0:511];
            reg [4*18-1:0] read_value;
            always @(posedge clk) begin
                if (load_valid && load_bank == bank)
                    memory[load_address] <= load_q12_packed;
                if (read_valid)
                    read_value <= memory[read_address];
            end
            genvar token_lane;
            for (token_lane = 0; token_lane < 4;
                 token_lane = token_lane + 1) begin : pack_tokens
                assign read_q12_packed[
                    (token_lane*32+bank)*18 +: 18
                ] = read_value[token_lane*18 +: 18];
            end
        end
    endgenerate

    always @(posedge clk)
        read_data_valid <= read_valid;

endmodule
