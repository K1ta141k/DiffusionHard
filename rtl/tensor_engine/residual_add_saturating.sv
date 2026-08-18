`timescale 1ns/1ps

module residual_add_saturating #(
    parameter integer LANES = 24,
    parameter integer DATA_WIDTH = 24,
    parameter integer TAG_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [LANES*DATA_WIDTH-1:0] values_packed,
    input  wire [LANES*DATA_WIDTH-1:0] residuals_packed,
    output reg  valid_out,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output reg  [LANES*DATA_WIDTH-1:0] outputs_packed
);

    localparam signed [DATA_WIDTH:0] OUTPUT_MAX =
        (1 <<< (DATA_WIDTH-1)) - 1;
    localparam signed [DATA_WIDTH:0] OUTPUT_MIN =
        -(1 <<< (DATA_WIDTH-1));

    reg signed [DATA_WIDTH:0] sums [0:LANES-1];
    reg signed [DATA_WIDTH-1:0] output_next [0:LANES-1];
    integer lane_index;

    always @* begin
        for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin
            sums[lane_index] =
                $signed(values_packed[lane_index*DATA_WIDTH +: DATA_WIDTH])
                + $signed(residuals_packed[lane_index*DATA_WIDTH +: DATA_WIDTH]);
            if (sums[lane_index] > OUTPUT_MAX) begin
                output_next[lane_index] = OUTPUT_MAX[DATA_WIDTH-1:0];
            end else if (sums[lane_index] < OUTPUT_MIN) begin
                output_next[lane_index] = OUTPUT_MIN[DATA_WIDTH-1:0];
            end else begin
                output_next[lane_index] = sums[lane_index][DATA_WIDTH-1:0];
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            tag_out <= {TAG_WIDTH{1'b0}};
            outputs_packed <= {LANES*DATA_WIDTH{1'b0}};
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                tag_out <= tag_in;
                for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin
                    outputs_packed[
                        lane_index*DATA_WIDTH +: DATA_WIDTH
                    ] <= output_next[lane_index];
                end
            end
        end
    end

endmodule
