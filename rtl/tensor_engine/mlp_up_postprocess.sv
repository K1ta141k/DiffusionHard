`timescale 1ns/1ps

module mlp_up_postprocess #(
    parameter integer LANES = 24,
    parameter integer ACC_WIDTH = 32,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer DATA_WIDTH = 16,
    parameter integer RIGHT_SHIFT = 20,
    parameter integer TAG_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [LANES*ACC_WIDTH-1:0] accumulators_packed,
    input  wire [LANES*MULTIPLIER_WIDTH-1:0] multipliers_packed,
    input  wire [LANES*ACC_WIDTH-1:0] biases_packed,
    output wire valid_out,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output wire [LANES*DATA_WIDTH-1:0] gelu_packed
);

    wire requant_valid;
    wire [LANES*DATA_WIDTH-1:0] requantized_packed;
    reg [TAG_WIDTH-1:0] requant_tag;

    fixed_requantize #(
        .LANES(LANES),
        .ACC_WIDTH(ACC_WIDTH),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .OUTPUT_WIDTH(DATA_WIDTH),
        .RIGHT_SHIFT(RIGHT_SHIFT)
    ) requantizer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .accumulators_packed(accumulators_packed),
        .multipliers_packed(multipliers_packed),
        .biases_packed(biases_packed),
        .valid_out(requant_valid),
        .outputs_packed(requantized_packed)
    );

    gelu_q10_lut #(
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH)
    ) gelu (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(requant_valid),
        .inputs_packed(requantized_packed),
        .valid_out(valid_out),
        .outputs_packed(gelu_packed)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            requant_tag <= {TAG_WIDTH{1'b0}};
            tag_out <= {TAG_WIDTH{1'b0}};
        end else begin
            if (valid_in) begin
                requant_tag <= tag_in;
            end
            if (requant_valid) begin
                tag_out <= requant_tag;
            end
        end
    end

endmodule
