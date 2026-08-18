`timescale 1ns/1ps

module attention_head_scratchpad_qk_combined_banked #(
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
            wire query_write_enable = (load_valid || query_load_valid)
                && load_channel[3:0] == bank_index;
            wire key_write_enable = (load_valid || key_load_valid)
                && load_channel[3:0] == bank_index;
            wire [DATA_WIDTH-1:0] query_read_value;
            wire [DATA_WIDTH-1:0] key_read_value;
`ifndef SYNTHESIS
            (* ram_style = "block" *) reg [DATA_WIDTH-1:0]
                query_key_memory [0:511];
            reg [DATA_WIDTH-1:0] behavioral_query_read_value;
            reg [DATA_WIDTH-1:0] behavioral_key_read_value;
            assign query_read_value = behavioral_query_read_value;
            assign key_read_value = behavioral_key_read_value;
`endif
            (* ram_style = "block" *) reg [DATA_WIDTH-1:0]
                value_memory [0:255];
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

`ifdef SYNTHESIS
            wire [15:0] query_data_bits;
            wire [1:0] query_parity_bits;
            wire [15:0] key_data_bits;
            wire [1:0] key_parity_bits;
            wire [8:0] query_memory_address = query_write_enable
                ? {1'b0, query_key_write_address}
                : {1'b0, query_read_address};
            wire [8:0] key_memory_address = key_write_enable
                ? {1'b1, query_key_write_address}
                : {1'b1, key_read_address};

            assign query_read_value = {query_parity_bits, query_data_bits};
            assign key_read_value = {key_parity_bits, key_data_bits};

            RAMB18E2 #(
                .CLOCK_DOMAINS("COMMON"),
                .DOA_REG(0), .DOB_REG(0),
                .READ_WIDTH_A(18), .READ_WIDTH_B(18),
                .WRITE_WIDTH_A(18), .WRITE_WIDTH_B(18),
                .WRITE_MODE_A("READ_FIRST"), .WRITE_MODE_B("READ_FIRST")
            ) query_key_bram (
                .DOUTADOUT(query_data_bits),
                .DOUTPADOUTP(query_parity_bits),
                .DOUTBDOUT(key_data_bits),
                .DOUTPBDOUTP(key_parity_bits),
                .ADDRARDADDR({1'b0, query_memory_address, 4'b0}),
                .ADDRBWRADDR({1'b0, key_memory_address, 4'b0}),
                .CLKARDCLK(clk), .CLKBWRCLK(clk),
                .DINADIN(load_query_q12[15:0]),
                .DINPADINP(load_query_q12[17:16]),
                .DINBDIN(load_key_q12[15:0]),
                .DINPBDINP(load_key_q12[17:16]),
                .ENARDEN(query_write_enable || query_read_valid),
                .ENBWREN(key_write_enable || key_read_valid),
                .REGCEAREGCE(1'b1), .REGCEB(1'b1),
                .RSTRAMARSTRAM(1'b0), .RSTRAMB(1'b0),
                .RSTREGARSTREG(1'b0), .RSTREGB(1'b0),
                .SLEEP(1'b0),
                .WEA({2{query_write_enable}}),
                .WEBWE({4{key_write_enable}}),
                .ADDRENA(1'b0), .ADDRENB(1'b0),
                .CASDIMUXA(1'b0), .CASDIMUXB(1'b0),
                .CASDINA(16'b0), .CASDINB(16'b0),
                .CASDINPA(2'b0), .CASDINPB(2'b0),
                .CASDOMUXA(1'b0), .CASDOMUXB(1'b0),
                .CASDOMUXEN_A(1'b0), .CASDOMUXEN_B(1'b0),
                .CASOREGIMUXA(1'b0), .CASOREGIMUXB(1'b0),
                .CASOREGIMUXEN_A(1'b0), .CASOREGIMUXEN_B(1'b0)
            );
`else
            always @(posedge clk) begin
                if (query_write_enable)
                    query_key_memory[{1'b0, query_key_write_address}] <=
                        load_query_q12;
                else if (query_read_valid)
                    behavioral_query_read_value <= query_key_memory[
                        {1'b0, query_read_address}
                    ];

                if (key_write_enable)
                    query_key_memory[{1'b1, query_key_write_address}] <=
                        load_key_q12;
                else if (key_read_valid)
                    behavioral_key_read_value <= query_key_memory[
                        {1'b1, key_read_address}
                    ];
            end
`endif

            always @(posedge clk) begin
                if ((load_valid || value_load_valid)
                    && selected_value_load_token[3:0] == bank_index)
                    value_memory[value_write_address] <= load_value_q12;
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
