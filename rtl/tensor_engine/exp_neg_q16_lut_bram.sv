`timescale 1ns/1ps

module exp_neg_q16_lut_bram #(
    parameter LUT_FILE = "rtl/tensor_engine/exp_neg_q16_lut.hex"
) (
    input  wire clk,
    input  wire [10:0] address,
    output reg  [15:0] value_q16
);

    (* rom_style = "block" *) reg [15:0] memory [0:1024];

    initial begin
        $readmemh(LUT_FILE, memory);
    end

    always @(posedge clk) begin
        value_q16 <= memory[address];
    end

endmodule
