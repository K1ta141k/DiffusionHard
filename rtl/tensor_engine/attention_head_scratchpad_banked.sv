`timescale 1ns/1ps

module attention_head_scratchpad_banked #(
    parameter integer DATA_WIDTH = 18,
    parameter integer TOKENS = 64,
    parameter integer HEAD_DIM = 64,
    parameter integer BANKS = 16,
    parameter integer TOKEN_WIDTH = 6,
    parameter integer CHANNEL_WIDTH = 6,
    parameter integer BLOCK_WIDTH = 2,
    parameter integer ADDRESS_WIDTH = 8,
    parameter integer SEPARATE_VALUE_LOAD_ADDRESS = 0
) (
    input  wire clk,
    input  wire load_valid,
    input  wire query_load_valid,
    input  wire key_load_valid,
    input  wire value_load_valid,
    input  wire [TOKEN_WIDTH-1:0] load_token,
    input  wire [CHANNEL_WIDTH-1:0] load_channel,
    input  wire [TOKEN_WIDTH-1:0] value_load_token,
    input  wire [CHANNEL_WIDTH-1:0] value_load_channel,
    input  wire signed [DATA_WIDTH-1:0] load_query_q12,
    input  wire signed [DATA_WIDTH-1:0] load_key_q12,
    input  wire signed [DATA_WIDTH-1:0] load_value_q12,
    input  wire query_read_valid,
    input  wire [TOKEN_WIDTH-1:0] query_read_token,
    input  wire [BLOCK_WIDTH-1:0] query_read_channel_block,
    output reg  query_data_valid,
    output wire [BANKS*DATA_WIDTH-1:0] query_data_packed,
    input  wire key_read_valid,
    input  wire [TOKEN_WIDTH-1:0] key_read_token,
    input  wire [BLOCK_WIDTH-1:0] key_read_channel_block,
    output reg  key_data_valid,
    output wire [BANKS*DATA_WIDTH-1:0] key_data_packed,
    input  wire value_read_valid,
    input  wire [BLOCK_WIDTH-1:0] value_read_key_block,
    input  wire [CHANNEL_WIDTH-1:0] value_read_channel,
    output reg  value_data_valid,
    output wire [BANKS*DATA_WIDTH-1:0] value_data_packed
);

    wire [ADDRESS_WIDTH-1:0] query_key_write_address = {
        load_channel[CHANNEL_WIDTH-1:4], load_token
    };
    wire [ADDRESS_WIDTH-1:0] query_read_address = {
        query_read_channel_block, query_read_token
    };
    wire [ADDRESS_WIDTH-1:0] key_read_address = {
        key_read_channel_block, key_read_token
    };
    wire [TOKEN_WIDTH-1:0] selected_value_load_token =
        SEPARATE_VALUE_LOAD_ADDRESS ? value_load_token : load_token;
    wire [CHANNEL_WIDTH-1:0] selected_value_load_channel =
        SEPARATE_VALUE_LOAD_ADDRESS ? value_load_channel : load_channel;
    wire [ADDRESS_WIDTH-1:0] value_write_address = {
        selected_value_load_token[TOKEN_WIDTH-1:4],
        selected_value_load_channel
    };
    wire [ADDRESS_WIDTH-1:0] value_read_address = {
        value_read_key_block, value_read_channel
    };

    genvar bank_index;
    generate
        for (bank_index = 0; bank_index < BANKS;
             bank_index = bank_index + 1) begin : banks
            (* ram_style = "block" *) reg [DATA_WIDTH-1:0]
                query_memory [0:255];
            (* ram_style = "block" *) reg [DATA_WIDTH-1:0]
                key_memory [0:255];
            (* ram_style = "block" *) reg [DATA_WIDTH-1:0]
                value_memory [0:255];
            reg [DATA_WIDTH-1:0] query_read_value;
            reg [DATA_WIDTH-1:0] key_read_value;
            reg [DATA_WIDTH-1:0] value_read_value;

            assign query_data_packed[
                bank_index*DATA_WIDTH +: DATA_WIDTH
            ] = query_read_value;
            assign key_data_packed[
                bank_index*DATA_WIDTH +: DATA_WIDTH
            ] = key_read_value;
            assign value_data_packed[
                bank_index*DATA_WIDTH +: DATA_WIDTH
            ] = value_read_value;

            always @(posedge clk) begin
                if ((load_valid || query_load_valid)
                    && load_channel[3:0] == bank_index)
                    query_memory[query_key_write_address] <= load_query_q12;
                if ((load_valid || key_load_valid)
                    && load_channel[3:0] == bank_index)
                    key_memory[query_key_write_address] <= load_key_q12;
                if ((load_valid || value_load_valid)
                    && selected_value_load_token[3:0] == bank_index)
                    value_memory[value_write_address] <= load_value_q12;
                if (query_read_valid)
                    query_read_value <= query_memory[query_read_address];
                if (key_read_valid)
                    key_read_value <= key_memory[key_read_address];
                if (value_read_valid)
                    value_read_value <= value_memory[value_read_address];
            end
        end
    endgenerate

    always @(posedge clk) begin
        query_data_valid <= query_read_valid;
        key_data_valid <= key_read_valid;
        value_data_valid <= value_read_valid;
    end

    initial begin
        if (TOKENS != 64 || HEAD_DIM != 64 || BANKS != 16)
            $error("attention scratchpad is specialized for 64 by 64 heads");
    end

endmodule
