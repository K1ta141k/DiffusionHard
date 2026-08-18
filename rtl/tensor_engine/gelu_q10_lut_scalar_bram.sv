`timescale 1ns/1ps

module gelu_q10_lut_scalar_bram #(
    parameter integer DATA_WIDTH = 16,
    parameter LUT_FILE = "rtl/tensor_engine/gelu_q10_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire [DATA_WIDTH-1:0] input_value,
    output reg  valid_out,
    output reg  [DATA_WIDTH-1:0] output_value
);

    localparam integer FRACTION_BITS = 10;
    localparam integer STEP_BITS = 4;
    localparam signed [DATA_WIDTH-1:0] LOWER_BOUND = -(8 <<< FRACTION_BITS);
    localparam signed [DATA_WIDTH-1:0] UPPER_BOUND =  (8 <<< FRACTION_BITS);

    (* rom_style = "block" *) reg signed [DATA_WIDTH-1:0] lookup [0:1023];
    reg [9:0] address_stage;
    reg signed [DATA_WIDTH-1:0] lookup_value;
    reg lower_stage_0;
    reg upper_stage_0;
    reg lower_stage_1;
    reg upper_stage_1;
    reg signed [DATA_WIDTH-1:0] input_stage_0;
    reg signed [DATA_WIDTH-1:0] input_stage_1;
    reg valid_stage_0;
    reg valid_stage_1;
    wire signed [DATA_WIDTH-1:0] signed_input;

    assign signed_input = $signed(input_value);

    initial begin
        $readmemh(LUT_FILE, lookup);
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            address_stage <= 10'b0;
            lookup_value <= {DATA_WIDTH{1'b0}};
            lower_stage_0 <= 1'b0;
            upper_stage_0 <= 1'b0;
            lower_stage_1 <= 1'b0;
            upper_stage_1 <= 1'b0;
            input_stage_0 <= {DATA_WIDTH{1'b0}};
            input_stage_1 <= {DATA_WIDTH{1'b0}};
            valid_stage_0 <= 1'b0;
            valid_stage_1 <= 1'b0;
            valid_out <= 1'b0;
            output_value <= {DATA_WIDTH{1'b0}};
        end else begin
            valid_stage_0 <= valid_in;
            valid_stage_1 <= valid_stage_0;
            valid_out <= valid_stage_1;

            if (valid_in) begin
                lower_stage_0 <= (signed_input <= LOWER_BOUND);
                upper_stage_0 <= (signed_input >= UPPER_BOUND);
                input_stage_0 <= signed_input;
                if (signed_input <= LOWER_BOUND || signed_input >= UPPER_BOUND) begin
                    address_stage <= 10'b0;
                end else begin
                    address_stage <=
                        (signed_input - LOWER_BOUND) >>> STEP_BITS;
                end
            end

            if (valid_stage_0) begin
                lookup_value <= lookup[address_stage];
                lower_stage_1 <= lower_stage_0;
                upper_stage_1 <= upper_stage_0;
                input_stage_1 <= input_stage_0;
            end

            if (valid_stage_1) begin
                if (lower_stage_1) begin
                    output_value <= {DATA_WIDTH{1'b0}};
                end else if (upper_stage_1) begin
                    output_value <= input_stage_1;
                end else begin
                    output_value <= lookup_value;
                end
            end
        end
    end

endmodule
