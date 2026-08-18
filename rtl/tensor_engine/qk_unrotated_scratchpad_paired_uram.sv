`timescale 1ns/1ps

module qk_unrotated_scratchpad_paired_uram (
    input  wire clk,
    input  wire query_load_valid,
    input  wire key_load_valid,
    input  wire [5:0] load_token,
    input  wire [5:0] load_channel,
    input  wire signed [17:0] load_query_q12,
    input  wire signed [17:0] load_key_q12,
    input  wire read_valid,
    input  wire [5:0] read_token,
    input  wire [4:0] read_pair,
    output reg  read_data_valid,
    output reg  [5:0] read_token_out,
    output reg  [4:0] read_pair_out,
    output wire signed [17:0] query_first_q12,
    output wire signed [17:0] query_second_q12,
    output wire signed [17:0] key_first_q12,
    output wire signed [17:0] key_second_q12
);

    (* ram_style = "ultra" *) reg [17:0] query_memory [0:4095];
    (* ram_style = "ultra" *) reg [17:0] key_memory [0:4095];
    reg signed [17:0] query_first_value;
    reg signed [17:0] query_second_value;
    reg signed [17:0] key_first_value;
    reg signed [17:0] key_second_value;
    wire [11:0] load_address = {load_token, load_channel};
    wire [11:0] first_read_address = {read_token, 1'b0, read_pair};
    wire [11:0] second_read_address = {read_token, 1'b1, read_pair};

    assign query_first_q12 = query_first_value;
    assign query_second_q12 = query_second_value;
    assign key_first_q12 = key_first_value;
    assign key_second_q12 = key_second_value;

    always @(posedge clk) begin
        if (query_load_valid)
            query_memory[load_address] <= load_query_q12;
        else if (read_valid)
            query_first_value <= query_memory[first_read_address];
        if (key_load_valid)
            key_memory[load_address] <= load_key_q12;
        else if (read_valid)
            key_first_value <= key_memory[first_read_address];
        if (read_valid) begin
            query_second_value <= query_memory[second_read_address];
            key_second_value <= key_memory[second_read_address];
        end
    end

    always @(posedge clk) begin
        read_data_valid <= read_valid;
        if (read_valid) begin
            read_token_out <= read_token;
            read_pair_out <= read_pair;
        end
    end

endmodule
