`timescale 1ns/1ps

module fixed_requantize #(
    parameter integer LANES = 24,
    parameter integer ACC_WIDTH = 32,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer OUTPUT_WIDTH = 16,
    parameter integer RIGHT_SHIFT = 20
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire [LANES*ACC_WIDTH-1:0] accumulators_packed,
    input  wire [LANES*MULTIPLIER_WIDTH-1:0] multipliers_packed,
    input  wire [LANES*ACC_WIDTH-1:0] biases_packed,
    output reg  valid_out,
    output reg  [LANES*OUTPUT_WIDTH-1:0] outputs_packed
);

    localparam integer PRODUCT_WIDTH = ACC_WIDTH + MULTIPLIER_WIDTH + 1;
    localparam signed [PRODUCT_WIDTH-1:0] ROUNDING_OFFSET =
        {{(PRODUCT_WIDTH-RIGHT_SHIFT){1'b0}}, 1'b1, {(RIGHT_SHIFT-1){1'b0}}};
    localparam signed [PRODUCT_WIDTH-1:0] OUTPUT_MAX =
        (1 <<< (OUTPUT_WIDTH-1)) - 1;
    localparam signed [PRODUCT_WIDTH-1:0] OUTPUT_MIN =
        -(1 <<< (OUTPUT_WIDTH-1));

    reg signed [PRODUCT_WIDTH-1:0] products [0:LANES-1];
    reg signed [PRODUCT_WIDTH-1:0] rounded [0:LANES-1];
    reg signed [PRODUCT_WIDTH-1:0] biased [0:LANES-1];
    reg signed [OUTPUT_WIDTH-1:0] output_next [0:LANES-1];
    integer lane_index;

    initial begin
        if (RIGHT_SHIFT <= 0 || RIGHT_SHIFT >= PRODUCT_WIDTH) begin
            $error("RIGHT_SHIFT must be between zero and PRODUCT_WIDTH");
        end
    end

    always @* begin
        for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin
            products[lane_index] =
                $signed(accumulators_packed[lane_index*ACC_WIDTH +: ACC_WIDTH])
                * $signed({
                    1'b0,
                    multipliers_packed[
                        lane_index*MULTIPLIER_WIDTH +: MULTIPLIER_WIDTH
                    ]
                });
            if (products[lane_index] >= 0) begin
                rounded[lane_index] =
                    (products[lane_index] + ROUNDING_OFFSET) >>> RIGHT_SHIFT;
            end else begin
                rounded[lane_index] = -(
                    ((-products[lane_index]) + ROUNDING_OFFSET) >>> RIGHT_SHIFT
                );
            end
            biased[lane_index] = rounded[lane_index]
                + $signed(biases_packed[lane_index*ACC_WIDTH +: ACC_WIDTH]);
            if (biased[lane_index] > OUTPUT_MAX) begin
                output_next[lane_index] = OUTPUT_MAX[OUTPUT_WIDTH-1:0];
            end else if (biased[lane_index] < OUTPUT_MIN) begin
                output_next[lane_index] = OUTPUT_MIN[OUTPUT_WIDTH-1:0];
            end else begin
                output_next[lane_index] = biased[lane_index][OUTPUT_WIDTH-1:0];
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            outputs_packed <= {LANES*OUTPUT_WIDTH{1'b0}};
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin
                    outputs_packed[
                        lane_index*OUTPUT_WIDTH +: OUTPUT_WIDTH
                    ] <= output_next[lane_index];
                end
            end
        end
    end

endmodule
