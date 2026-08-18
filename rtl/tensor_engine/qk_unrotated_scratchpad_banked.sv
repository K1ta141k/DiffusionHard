`timescale 1ns/1ps

module qk_unrotated_scratchpad_banked (
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

    wire [7:0] load_address = {load_channel[5:4], load_token};
    wire [7:0] first_read_address = {read_pair[4], read_token};
    wire [7:0] second_read_address = {1'b1, read_pair[4], read_token};
    reg [3:0] selected_bank;
    wire [16*18-1:0] query_first_banks;
    wire [16*18-1:0] query_second_banks;
    wire [16*18-1:0] key_first_banks;
    wire [16*18-1:0] key_second_banks;
    genvar bank;

    assign query_first_q12 = query_first_banks[selected_bank*18 +: 18];
    assign query_second_q12 = query_second_banks[selected_bank*18 +: 18];
    assign key_first_q12 = key_first_banks[selected_bank*18 +: 18];
    assign key_second_q12 = key_second_banks[selected_bank*18 +: 18];

    generate
        for (bank = 0; bank < 16; bank = bank + 1) begin : banks
            (* ram_style = "distributed" *) reg [17:0] query_memory [0:255];
            (* ram_style = "distributed" *) reg [17:0] key_memory [0:255];
            reg [17:0] query_first_value;
            reg [17:0] query_second_value;
            reg [17:0] key_first_value;
            reg [17:0] key_second_value;

            assign query_first_banks[bank*18 +: 18] = query_first_value;
            assign query_second_banks[bank*18 +: 18] = query_second_value;
            assign key_first_banks[bank*18 +: 18] = key_first_value;
            assign key_second_banks[bank*18 +: 18] = key_second_value;

            always @(posedge clk) begin
                if (query_load_valid && load_channel[3:0] == bank)
                    query_memory[load_address] <= load_query_q12;
                if (key_load_valid && load_channel[3:0] == bank)
                    key_memory[load_address] <= load_key_q12;
                if (read_valid) begin
                    query_first_value <= query_memory[first_read_address];
                    query_second_value <= query_memory[second_read_address];
                    key_first_value <= key_memory[first_read_address];
                    key_second_value <= key_memory[second_read_address];
                end
            end
        end
    endgenerate

    always @(posedge clk) begin
        read_data_valid <= read_valid;
        if (read_valid) begin
            selected_bank <= read_pair[3:0];
            read_token_out <= read_token;
            read_pair_out <= read_pair;
        end
    end

endmodule
