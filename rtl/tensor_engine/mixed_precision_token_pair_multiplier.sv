`timescale 1ns/1ps

// One DSP serves either one signed 18x18 attention product or two signed INT8
// MLP token products that share a weight.
module mixed_precision_selected_pair_multiplier (
    input  wire clk,
    input  wire rst_n,
    input  wire signed [26:0] selected_activation,
    input  wire signed [17:0] selected_weight,
    output wire signed [35:0] attention_product,
    output wire signed [17:0] mlp_offset_product_0,
    output wire signed [17:0] mlp_offset_product_1
);

    wire signed [44:0] selected_product =
        selected_activation * selected_weight;
    reg signed [44:0] product_pipeline;
    wire signed [17:0] low_mlp_product = $signed(product_pipeline[17:0]);
    wire signed [26:0] high_mlp_floor = $signed(product_pipeline) >>> 18;
    wire signed [26:0] high_mlp_product = high_mlp_floor
        + (low_mlp_product < 0 ? 27'sd1 : 27'sd0);

    assign attention_product = product_pipeline[35:0];
    assign mlp_offset_product_0 = low_mlp_product;
    assign mlp_offset_product_1 = high_mlp_product[17:0];

    always @(posedge clk) begin
        if (!rst_n)
            product_pipeline <= 45'sd0;
        else
            product_pipeline <= selected_product;
    end

endmodule

module mixed_precision_token_pair_multiplier (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire narrow_int8_mode,
    input  wire signed [17:0] attention_activation,
    input  wire signed [17:0] attention_weight,
    input  wire signed [7:0] mlp_activation_0,
    input  wire signed [7:0] mlp_activation_1,
    input  wire signed [7:0] mlp_weight,
    output reg  valid_out,
    output reg  narrow_int8_mode_out,
    output wire signed [35:0] attention_product,
    output wire signed [17:0] mlp_offset_product_0,
    output wire signed [17:0] mlp_offset_product_1
);

    wire signed [8:0] activation_0_extended = {
        mlp_activation_0[7], mlp_activation_0
    };
    wire signed [8:0] activation_1_extended = {
        mlp_activation_1[7], mlp_activation_1
    };
    wire [8:0] activation_0_offset = activation_0_extended + 9'sd128;
    wire [8:0] activation_1_offset = activation_1_extended + 9'sd128;
    wire signed [26:0] packed_mlp_activations = $signed({
        1'b0,
        activation_1_offset[7:0],
        10'b0,
        activation_0_offset[7:0]
    });
    wire signed [26:0] selected_activation = narrow_int8_mode
        ? packed_mlp_activations
        : {{9{attention_activation[17]}}, attention_activation};
    wire signed [17:0] selected_weight = narrow_int8_mode
        ? {{10{mlp_weight[7]}}, mlp_weight}
        : attention_weight;
    mixed_precision_selected_pair_multiplier selected_multiplier (
        .clk(clk),
        .rst_n(rst_n),
        .selected_activation(selected_activation),
        .selected_weight(selected_weight),
        .attention_product(attention_product),
        .mlp_offset_product_0(mlp_offset_product_0),
        .mlp_offset_product_1(mlp_offset_product_1)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            narrow_int8_mode_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            narrow_int8_mode_out <= narrow_int8_mode;
        end
    end

endmodule
