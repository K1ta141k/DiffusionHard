`timescale 1ns/1ps

module attention_dynamic_score_requantizer_q10 #(
    parameter integer DOT_WIDTH = 21,
    parameter integer TAG_WIDTH = 8,
    parameter integer PRODUCT_WIDTH = DOT_WIDTH + 33
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire signed [DOT_WIDTH-1:0] dot_product_int8,
    input  wire [17:0] query_maximum,
    input  wire [17:0] key_maximum,
    output reg  valid_out,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output reg  signed [17:0] score_q10
);

    localparam signed [PRODUCT_WIDTH-1:0] ROUNDING_OFFSET =
        {{(PRODUCT_WIDTH-28){1'b0}}, 1'b1, 27'b0};
    localparam signed [PRODUCT_WIDTH-1:0] SCORE_MAX = 131071;
    localparam signed [PRODUCT_WIDTH-1:0] SCORE_MIN = -131072;

    wire multiplier_valid;
    wire [TAG_WIDTH-1:0] multiplier_tag;
    wire [31:0] multiplier_q28;
    reg signed [DOT_WIDTH-1:0] dot_delay [0:4];
    reg signed [PRODUCT_WIDTH-1:0] scaled_product;
    reg signed [PRODUCT_WIDTH-1:0] rounded_score;
    reg signed [17:0] saturated_score;
    integer stage;

    attention_dynamic_score_multiplier_q28 #(
        .TAG_WIDTH(TAG_WIDTH)
    ) multiplier (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .tag_in(tag_in),
        .query_maximum(query_maximum), .key_maximum(key_maximum),
        .valid_out(multiplier_valid), .tag_out(multiplier_tag),
        .multiplier_q28(multiplier_q28)
    );

    always @* begin
        scaled_product = $signed(dot_delay[4])
            * $signed({1'b0, multiplier_q28});
        if (scaled_product >= 0)
            rounded_score =
                (scaled_product + ROUNDING_OFFSET) >>> 28;
        else
            rounded_score = -(
                ((-scaled_product) + ROUNDING_OFFSET) >>> 28
            );
        if (rounded_score > SCORE_MAX)
            saturated_score = 18'sd131071;
        else if (rounded_score < SCORE_MIN)
            saturated_score = -18'sd131072;
        else
            saturated_score = rounded_score[17:0];
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            for (stage = 0; stage < 5; stage = stage + 1)
                dot_delay[stage] <= 0;
            valid_out <= 1'b0;
            tag_out <= 0;
            score_q10 <= 0;
        end else begin
            dot_delay[0] <= dot_product_int8;
            for (stage = 1; stage < 5; stage = stage + 1)
                dot_delay[stage] <= dot_delay[stage-1];
            valid_out <= multiplier_valid;
            if (multiplier_valid) begin
                tag_out <= multiplier_tag;
                score_q10 <= saturated_score;
            end
        end
    end

endmodule
