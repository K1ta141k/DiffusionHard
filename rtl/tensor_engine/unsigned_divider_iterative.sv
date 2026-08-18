`timescale 1ns/1ps

module unsigned_divider_iterative #(
    parameter integer WIDTH = 26,
    parameter integer COUNT_WIDTH = (WIDTH <= 1) ? 1 : $clog2(WIDTH + 1)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output wire ready,
    input  wire [WIDTH-1:0] dividend,
    input  wire [WIDTH-1:0] divisor,
    output reg  busy,
    output reg  valid_out,
    output reg  [WIDTH-1:0] quotient,
    output reg  [WIDTH-1:0] remainder
);

    reg [WIDTH-1:0] dividend_shift;
    reg [WIDTH-1:0] divisor_reg;
    reg [WIDTH-1:0] quotient_work;
    reg [WIDTH-1:0] remainder_work;
    reg [COUNT_WIDTH-1:0] iterations_left;
    reg [WIDTH-1:0] shifted_remainder;
    reg [WIDTH-1:0] next_remainder;
    reg [WIDTH-1:0] next_quotient;

    assign ready = !busy;

    always @* begin
        shifted_remainder = {
            remainder_work[WIDTH-2:0], dividend_shift[WIDTH-1]
        };
        if (shifted_remainder >= divisor_reg) begin
            next_remainder = shifted_remainder - divisor_reg;
            next_quotient = {quotient_work[WIDTH-2:0], 1'b1};
        end else begin
            next_remainder = shifted_remainder;
            next_quotient = {quotient_work[WIDTH-2:0], 1'b0};
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0;
            valid_out <= 1'b0;
            quotient <= {WIDTH{1'b0}};
            remainder <= {WIDTH{1'b0}};
            dividend_shift <= {WIDTH{1'b0}};
            divisor_reg <= {WIDTH{1'b0}};
            quotient_work <= {WIDTH{1'b0}};
            remainder_work <= {WIDTH{1'b0}};
            iterations_left <= {COUNT_WIDTH{1'b0}};
        end else begin
            valid_out <= 1'b0;
            if (start && ready) begin
                if (divisor == 0) begin
                    quotient <= {WIDTH{1'b1}};
                    remainder <= dividend;
                    valid_out <= 1'b1;
                end else begin
                    busy <= 1'b1;
                    dividend_shift <= dividend;
                    divisor_reg <= divisor;
                    quotient_work <= {WIDTH{1'b0}};
                    remainder_work <= {WIDTH{1'b0}};
                    iterations_left <= WIDTH;
                end
            end else if (busy) begin
                dividend_shift <= {dividend_shift[WIDTH-2:0], 1'b0};
                quotient_work <= next_quotient;
                remainder_work <= next_remainder;
                iterations_left <= iterations_left - 1'b1;
                if (iterations_left == 1) begin
                    busy <= 1'b0;
                    quotient <= next_quotient;
                    remainder <= next_remainder;
                    valid_out <= 1'b1;
                end
            end
        end
    end

endmodule
