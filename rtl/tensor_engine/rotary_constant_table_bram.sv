`timescale 1ns/1ps

module rotary_constant_table_bram (
    input  wire clk,
    input  wire load_valid,
    input  wire [5:0] load_token,
    input  wire [4:0] load_pair,
    input  wire signed [15:0] load_cosine_q15,
    input  wire signed [15:0] load_sine_q15,
    input  wire read_valid,
    input  wire [5:0] read_token,
    input  wire [4:0] read_pair,
    output reg  read_data_valid,
    output reg  [5:0] read_token_out,
    output reg  [4:0] read_pair_out,
    output reg  signed [15:0] cosine_q15,
    output reg  signed [15:0] sine_q15
);

    wire [10:0] load_address = {load_token, load_pair};
    wire [10:0] read_address = {read_token, read_pair};
    (* ram_style = "block" *) reg [31:0] memory [0:2047];

    always @(posedge clk) begin
        if (load_valid)
            memory[load_address] <= {load_sine_q15, load_cosine_q15};
        read_data_valid <= read_valid;
        if (read_valid) begin
            {sine_q15, cosine_q15} <= memory[read_address];
            read_token_out <= read_token;
            read_pair_out <= read_pair;
        end
    end

endmodule
