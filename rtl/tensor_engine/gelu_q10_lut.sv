`timescale 1ns/1ps

module gelu_q10_lut #(
    parameter integer LANES = 24,
    parameter integer DATA_WIDTH = 16,
    parameter LUT_FILE = "rtl/tensor_engine/gelu_q10_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire [LANES*DATA_WIDTH-1:0] inputs_packed,
    output reg  valid_out,
    output reg  [LANES*DATA_WIDTH-1:0] outputs_packed
);

    localparam integer FRACTION_BITS = 10;
    localparam integer STEP_BITS = 4;
    localparam signed [DATA_WIDTH-1:0] LOWER_BOUND = -(8 <<< FRACTION_BITS);
    localparam signed [DATA_WIDTH-1:0] UPPER_BOUND =  (8 <<< FRACTION_BITS);

    reg signed [DATA_WIDTH-1:0] lookup [0:1023];
    reg signed [DATA_WIDTH-1:0] input_value [0:LANES-1];
    reg signed [DATA_WIDTH-1:0] output_next [0:LANES-1];
    reg [9:0] lookup_address [0:LANES-1];
    integer lane_index;

    initial begin
        $readmemh(LUT_FILE, lookup);
    end

    always @* begin
        for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin
            input_value[lane_index] = $signed(
                inputs_packed[lane_index*DATA_WIDTH +: DATA_WIDTH]
            );
            lookup_address[lane_index] = 10'b0;
            if (input_value[lane_index] <= LOWER_BOUND) begin
                output_next[lane_index] = {DATA_WIDTH{1'b0}};
            end else if (input_value[lane_index] >= UPPER_BOUND) begin
                output_next[lane_index] = input_value[lane_index];
            end else begin
                lookup_address[lane_index] =
                    (input_value[lane_index] - LOWER_BOUND) >>> STEP_BITS;
                output_next[lane_index] = lookup[lookup_address[lane_index]];
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            outputs_packed <= {LANES*DATA_WIDTH{1'b0}};
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin
                    outputs_packed[
                        lane_index*DATA_WIDTH +: DATA_WIDTH
                    ] <= output_next[lane_index];
                end
            end
        end
    end

endmodule
