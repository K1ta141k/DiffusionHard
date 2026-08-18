`timescale 1ns/1ps

// Computes two signed INT8 products that share one signed INT8 weight with one
// 27x18 multiply. The activation offsets are separated by eighteen guard bits.
module int8_shared_weight_pair_multiplier (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire signed [7:0] activation_0,
    input  wire signed [7:0] activation_1,
    input  wire signed [7:0] weight,
    output reg  valid_out,
    output reg  signed [15:0] product_0,
    output reg  signed [15:0] product_1
);

    wire signed [8:0] activation_0_extended = {activation_0[7], activation_0};
    wire signed [8:0] activation_1_extended = {activation_1[7], activation_1};
    wire [8:0] activation_0_offset = activation_0_extended + 9'sd128;
    wire [8:0] activation_1_offset = activation_1_extended + 9'sd128;
    wire signed [26:0] packed_activations = $signed({
        1'b0,
        activation_1_offset[7:0],
        10'b0,
        activation_0_offset[7:0]
    });
    wire signed [17:0] extended_weight = {{10{weight[7]}}, weight};
    wire signed [44:0] packed_product = packed_activations * extended_weight;

    reg signed [44:0] packed_product_pipeline;
    reg signed [7:0] weight_pipeline;
    reg valid_pipeline;
    wire signed [17:0] low_offset_product =
        $signed(packed_product_pipeline[17:0]);
    wire signed [26:0] high_floor = $signed(packed_product_pipeline) >>> 18;
    wire signed [26:0] high_offset_product = high_floor
        + (low_offset_product < 0 ? 27'sd1 : 27'sd0);
    wire signed [15:0] offset_bias = $signed(weight_pipeline) <<< 7;
    wire signed [18:0] corrected_product_0 = low_offset_product - offset_bias;
    wire signed [27:0] corrected_product_1 = high_offset_product - offset_bias;

    always @(posedge clk) begin
        if (!rst_n) begin
            packed_product_pipeline <= 45'sd0;
            weight_pipeline <= 8'sd0;
            valid_pipeline <= 1'b0;
            valid_out <= 1'b0;
            product_0 <= 16'sd0;
            product_1 <= 16'sd0;
        end else begin
            packed_product_pipeline <= packed_product;
            weight_pipeline <= weight;
            valid_pipeline <= valid_in;
            valid_out <= valid_pipeline;
            if (valid_pipeline) begin
                product_0 <= corrected_product_0[15:0];
                product_1 <= corrected_product_1[15:0];
            end
        end
    end

endmodule

// Offset-product form for a dot-product tree. Bias correction is deliberately
// deferred until after the 32-way sum, which shares it across all token lanes.
module int8_shared_weight_pair_offset_multiplier (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire signed [7:0] activation_0,
    input  wire signed [7:0] activation_1,
    input  wire signed [7:0] weight,
    output reg  valid_out,
    output wire signed [17:0] offset_product_0,
    output wire signed [17:0] offset_product_1
);

    wire signed [8:0] activation_0_extended = {activation_0[7], activation_0};
    wire signed [8:0] activation_1_extended = {activation_1[7], activation_1};
    wire [8:0] activation_0_offset = activation_0_extended + 9'sd128;
    wire [8:0] activation_1_offset = activation_1_extended + 9'sd128;
    wire signed [26:0] packed_activations = $signed({
        1'b0,
        activation_1_offset[7:0],
        10'b0,
        activation_0_offset[7:0]
    });
    wire signed [17:0] extended_weight = {{10{weight[7]}}, weight};
    wire signed [44:0] packed_product = packed_activations * extended_weight;
    reg signed [44:0] packed_product_pipeline;
    wire signed [17:0] low_product =
        $signed(packed_product_pipeline[17:0]);
    wire signed [26:0] high_floor = $signed(packed_product_pipeline) >>> 18;
    wire signed [26:0] high_product = high_floor
        + (low_product < 0 ? 27'sd1 : 27'sd0);

    assign offset_product_0 = low_product;
    assign offset_product_1 = high_product[17:0];

    always @(posedge clk) begin
        if (!rst_n) begin
            packed_product_pipeline <= 45'sd0;
            valid_out <= 1'b0;
        end else begin
            packed_product_pipeline <= packed_product;
            valid_out <= valid_in;
        end
    end

endmodule
