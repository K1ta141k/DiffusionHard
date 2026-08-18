`timescale 1ns/1ps

module wide_synchronous_uram #(
    parameter integer WIDTH = 2048,
    parameter integer DEPTH = 768,
    parameter integer ADDR_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire clk,
    input  wire write_valid,
    input  wire [ADDR_WIDTH-1:0] write_address,
    input  wire [WIDTH-1:0] write_data,
    input  wire read_valid,
    input  wire [ADDR_WIDTH-1:0] read_address,
    output reg  read_data_valid,
    output wire [WIDTH-1:0] read_data
);

    localparam integer BANK_WIDTH = 72;
    localparam integer BANKS = (WIDTH + BANK_WIDTH - 1) / BANK_WIDTH;
    localparam integer PADDED_WIDTH = BANKS * BANK_WIDTH;
    localparam integer PAD_BITS = PADDED_WIDTH - WIDTH;
    wire [PADDED_WIDTH-1:0] padded_write_data = {
        {PAD_BITS{1'b0}}, write_data
    };
    wire [PADDED_WIDTH-1:0] padded_read_data;

    genvar bank;
    generate
        for (bank = 0; bank < BANKS; bank = bank + 1) begin : banks
            (* ram_style = "ultra" *)
            reg [BANK_WIDTH-1:0] memory [0:DEPTH-1];
            reg [BANK_WIDTH-1:0] read_value;

            always @(posedge clk) begin
                if (write_valid)
                    memory[write_address] <= padded_write_data[
                        bank*BANK_WIDTH +: BANK_WIDTH
                    ];
                if (read_valid)
                    read_value <= memory[read_address];
            end

            assign padded_read_data[bank*BANK_WIDTH +: BANK_WIDTH] =
                read_value;
        end
    endgenerate

    assign read_data = padded_read_data[WIDTH-1:0];

    always @(posedge clk)
        read_data_valid <= read_valid;

    initial begin
        if (WIDTH <= 0 || DEPTH <= 0)
            $error("wide_synchronous_uram requires positive dimensions");
    end

endmodule
