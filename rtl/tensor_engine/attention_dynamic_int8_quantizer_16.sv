`timescale 1ns/1ps

module attention_dynamic_int8_quantizer_16 #(
    parameter integer LANES = 16,
    parameter integer INPUT_WIDTH = 18,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer RIGHT_SHIFT = 17,
    parameter integer TAG_WIDTH = 8,
    parameter integer PRODUCT_WIDTH = INPUT_WIDTH + MULTIPLIER_WIDTH + 1
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [LANES*INPUT_WIDTH-1:0] values_q12_packed,
    input  wire [MULTIPLIER_WIDTH-1:0] multiplier_q17,
    output reg  valid_out,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output reg  [LANES*8-1:0] values_int8_packed
);

    localparam signed [PRODUCT_WIDTH-1:0] ROUNDING_OFFSET =
        {{(PRODUCT_WIDTH-RIGHT_SHIFT){1'b0}}, 1'b1,
         {(RIGHT_SHIFT-1){1'b0}}};
    localparam signed [PRODUCT_WIDTH-1:0] INT8_MAX = 127;
    localparam signed [PRODUCT_WIDTH-1:0] INT8_MIN = -127;

    reg signed [PRODUCT_WIDTH-1:0] product [0:LANES-1];
    reg signed [PRODUCT_WIDTH-1:0] rounded [0:LANES-1];
    reg signed [7:0] quantized [0:LANES-1];
    reg [LANES*8-1:0] quantized_packed;
    integer lane;

    always @* begin
        quantized_packed = 0;
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            product[lane] = $signed(
                values_q12_packed[lane*INPUT_WIDTH +: INPUT_WIDTH]
            ) * $signed({1'b0, multiplier_q17});
            if (product[lane] >= 0)
                rounded[lane] =
                    (product[lane] + ROUNDING_OFFSET) >>> RIGHT_SHIFT;
            else
                rounded[lane] = -(
                    ((-product[lane]) + ROUNDING_OFFSET) >>> RIGHT_SHIFT
                );
            if (rounded[lane] > INT8_MAX)
                quantized[lane] = 8'sd127;
            else if (rounded[lane] < INT8_MIN)
                quantized[lane] = -8'sd127;
            else
                quantized[lane] = rounded[lane][7:0];
            quantized_packed[lane*8 +: 8] = quantized[lane];
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            tag_out <= 0;
            values_int8_packed <= 0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                tag_out <= tag_in;
                values_int8_packed <= quantized_packed;
            end
        end
    end

endmodule
