`timescale 1ns/1ps

module attention_canvas_scratchpad_banked #(
    parameter integer DATA_WIDTH = 18,
    parameter integer BANKS = 64,
    parameter integer DEPTH = 768,
    parameter integer ADDRESS_WIDTH = 10
) (
    input  wire clk,
    input  wire rst_n,
    input  wire tile_valid,
    output wire tile_ready,
    input  wire [3:0] tile_head,
    input  wire [3:0] tile_group,
    input  wire [3:0] tile_channel_tile,
    input  wire [2:0] tile_valid_channels,
    input  wire [4*6*DATA_WIDTH-1:0] tile_data_packed,
    output reg  tile_done,
    input  wire read_valid,
    input  wire [3:0] read_head,
    input  wire [5:0] read_token,
    output reg  read_data_valid,
    output wire [BANKS*DATA_WIDTH-1:0] read_data_packed
);

    reg write_active;
    reg [3:0] active_head;
    reg [3:0] active_group;
    reg [3:0] active_channel_tile;
    reg [2:0] active_valid_channels;
    reg [1:0] write_token;
    reg [4*6*DATA_WIDTH-1:0] tile_buffer;
    reg [6*DATA_WIDTH-1:0] active_token_data;
    wire [ADDRESS_WIDTH-1:0] write_address = {
        active_head, active_group, write_token
    };
    wire [ADDRESS_WIDTH-1:0] read_address = {read_head, read_token};

    genvar bank_index;

    assign tile_ready = !write_active;

    // Select the current token lane once, then fan the six channel values out
    // to their fixed banks.  Keeping the selectors outside the bank generate
    // avoids replicating a wide variable part-select in every RAM write port.
    always @* begin
        case (write_token)
            2'd0: active_token_data = tile_buffer[0*6*DATA_WIDTH +: 6*DATA_WIDTH];
            2'd1: active_token_data = tile_buffer[1*6*DATA_WIDTH +: 6*DATA_WIDTH];
            2'd2: active_token_data = tile_buffer[2*6*DATA_WIDTH +: 6*DATA_WIDTH];
            default: active_token_data = tile_buffer[3*6*DATA_WIDTH +: 6*DATA_WIDTH];
        endcase
    end

    generate
        for (bank_index = 0; bank_index < BANKS;
             bank_index = bank_index + 1) begin : canvas_banks
            localparam integer BANK_TILE = bank_index / 6;
            localparam integer BANK_LANE = bank_index % 6;
            (* ram_style = "block" *) reg [DATA_WIDTH-1:0] memory [0:DEPTH-1];
            reg [DATA_WIDTH-1:0] read_value;
            assign read_data_packed[
                bank_index*DATA_WIDTH +: DATA_WIDTH
            ] = read_value;

            always @(posedge clk) begin
                if (write_active
                    && active_channel_tile == BANK_TILE
                    && BANK_LANE < active_valid_channels)
                    memory[write_address] <= active_token_data[
                        BANK_LANE*DATA_WIDTH +: DATA_WIDTH
                    ];
                if (read_valid)
                    read_value <= memory[read_address];
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            write_active <= 1'b0;
            active_head <= 0;
            active_group <= 0;
            active_channel_tile <= 0;
            active_valid_channels <= 0;
            write_token <= 0;
            tile_buffer <= 0;
            tile_done <= 1'b0;
            read_data_valid <= 1'b0;
        end else begin
            tile_done <= 1'b0;
            read_data_valid <= read_valid;
            if (tile_valid && tile_ready) begin
                write_active <= 1'b1;
                active_head <= tile_head;
                active_group <= tile_group;
                active_channel_tile <= tile_channel_tile;
                active_valid_channels <= tile_valid_channels;
                write_token <= 0;
                tile_buffer <= tile_data_packed;
            end else if (write_active) begin
                if (write_token == 3) begin
                    write_active <= 1'b0;
                    tile_done <= 1'b1;
                end else begin
                    write_token <= write_token + 1'b1;
                end
            end
        end
    end

    initial begin
        if (BANKS != 64 || DEPTH != 768)
            $error("attention canvas is specialized for 64 tokens by 12 heads");
    end

endmodule
