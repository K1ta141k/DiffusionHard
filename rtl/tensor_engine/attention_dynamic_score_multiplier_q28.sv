`timescale 1ns/1ps

module attention_dynamic_score_multiplier_q28 #(
    parameter integer TAG_WIDTH = 8
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [17:0] query_maximum,
    input  wire [17:0] key_maximum,
    output reg  valid_out,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output reg  [31:0] multiplier_q28
);

    // D = 127^2 * 2^17.  The reciprocal constant is floor(2^60 / D).
    // For every 18-bit magnitude product, the provisional quotient is less
    // than one below the exact result.  One remainder comparison therefore
    // recovers exact round-to-nearest, ties-to-even behavior.
    localparam [30:0] DENOMINATOR = 31'd2114060288;
    localparam [13:0] DENOMINATOR_BASE = 14'd16129;
    localparam [29:0] RECIPROCAL_Q60_MINUS_Q28 = 30'd545358858;

    reg [35:0] maxima_product;
    reg [35:0] maxima_product_delay_1;
    reg [35:0] maxima_product_delay_2;
    reg [35:0] maxima_product_delay_3;
    reg [65:0] reciprocal_product;
    reg [31:0] provisional_quotient;
    reg [31:0] provisional_quotient_delay;
    reg [45:0] quotient_times_denominator_base;
    reg [3:0] valid_pipeline;
    reg [TAG_WIDTH-1:0] tag_pipeline [0:3];

    wire [63:0] scaled_numerator = {maxima_product_delay_3, 28'b0};
    wire [62:0] provisional_denominator_product =
        {quotient_times_denominator_base, 17'b0};
    wire [63:0] residual = scaled_numerator
        - {1'b0, provisional_denominator_product};
    wire [64:0] doubled_residual = {residual, 1'b0};
    wire [64:0] extended_denominator = {{34{1'b0}}, DENOMINATOR};
    wire round_up = (doubled_residual > extended_denominator)
        || ((doubled_residual == extended_denominator)
            && provisional_quotient_delay[0]);

    integer stage;
    always @(posedge clk) begin
        if (!rst_n) begin
            maxima_product <= 0;
            maxima_product_delay_1 <= 0;
            maxima_product_delay_2 <= 0;
            maxima_product_delay_3 <= 0;
            reciprocal_product <= 0;
            provisional_quotient <= 0;
            provisional_quotient_delay <= 0;
            quotient_times_denominator_base <= 0;
            valid_pipeline <= 0;
            valid_out <= 1'b0;
            tag_out <= 0;
            multiplier_q28 <= 0;
            for (stage = 0; stage < 4; stage = stage + 1)
                tag_pipeline[stage] <= 0;
        end else begin
            maxima_product <= query_maximum * key_maximum;
            reciprocal_product <=
                maxima_product * RECIPROCAL_Q60_MINUS_Q28;
            maxima_product_delay_1 <= maxima_product;
            provisional_quotient <= reciprocal_product[63:32];
            maxima_product_delay_2 <= maxima_product_delay_1;
            maxima_product_delay_3 <= maxima_product_delay_2;
            quotient_times_denominator_base <=
                provisional_quotient * DENOMINATOR_BASE;
            provisional_quotient_delay <= provisional_quotient;

            valid_pipeline[0] <= valid_in;
            tag_pipeline[0] <= tag_in;
            for (stage = 1; stage < 4; stage = stage + 1) begin
                valid_pipeline[stage] <= valid_pipeline[stage-1];
                tag_pipeline[stage] <= tag_pipeline[stage-1];
            end
            valid_out <= valid_pipeline[3];
            if (valid_pipeline[3]) begin
                tag_out <= tag_pipeline[3];
                multiplier_q28 <= provisional_quotient_delay + round_up;
            end
        end
    end

endmodule
