`timescale 1ns/1ps

// One UltraScale+ DSP cell. In attention mode, a cell may add PCIN to its
// signed 27x18 product. In packed MLP mode, every cell exposes its raw packed
// product so the two INT8 token products can be decoded independently.
module mixed_precision_dsp48e2_cascade_cell (
    input  wire clk,
    input  wire rst_n,
    input  wire cascade_enable,
    input  wire signed [26:0] selected_activation,
    input  wire signed [17:0] selected_weight,
    input  wire signed [47:0] cascade_in,
    output wire signed [47:0] result,
    output wire signed [47:0] cascade_out
);

`ifdef SYNTHESIS
    wire [29:0] acout_unused;
    wire [17:0] bcout_unused;
    wire carrycascout_unused;
    wire [3:0] carryout_unused;
    wire multsignout_unused;
    wire overflow_unused;
    wire patternbdetect_unused;
    wire patterndetect_unused;
    wire underflow_unused;
    wire [7:0] xorout_unused;
    wire [47:0] primitive_p;
    wire [47:0] primitive_pcout;

    DSP48E2 #(
        .ACASCREG(0),
        .ADREG(0),
        .A_INPUT("DIRECT"),
        .ALUMODEREG(0),
        .AREG(0),
        .BCASCREG(0),
        .B_INPUT("DIRECT"),
        .BREG(0),
        .CARRYINREG(0),
        .CARRYINSELREG(0),
        .CREG(0),
        .DREG(0),
        .INMODEREG(0),
        .MREG(1),
        .OPMODEREG(0),
        .PREG(0),
        .USE_MULT("MULTIPLY"),
        .USE_SIMD("ONE48"),
        .AMULTSEL("A"),
        .BMULTSEL("B")
    ) primitive (
        .A({{3{selected_activation[26]}}, selected_activation}),
        .B(selected_weight),
        .C(48'b0),
        .D(27'b0),
        .P(primitive_p),
        .PCOUT(primitive_pcout),
        .PCIN(cascade_in),
        .INMODE(5'b00000),
        .ALUMODE(4'b0000),
        .OPMODE(cascade_enable ? 9'b000010101 : 9'b000000101),
        .CARRYINSEL(3'b000),
        .ACIN(30'b0),
        .BCIN(18'b0),
        .CARRYCASCIN(1'b0),
        .CARRYIN(1'b0),
        .MULTSIGNIN(1'b0),
        .CEA1(1'b1),
        .CEA2(1'b1),
        .CEAD(1'b1),
        .CEALUMODE(1'b1),
        .CEB1(1'b1),
        .CEB2(1'b1),
        .CEC(1'b1),
        .CECARRYIN(1'b1),
        .CECTRL(1'b1),
        .CED(1'b1),
        .CEINMODE(1'b1),
        .CEM(1'b1),
        .CEP(1'b1),
        .CLK(clk),
        .RSTA(!rst_n),
        .RSTALLCARRYIN(!rst_n),
        .RSTALUMODE(!rst_n),
        .RSTB(!rst_n),
        .RSTC(!rst_n),
        .RSTCTRL(!rst_n),
        .RSTD(!rst_n),
        .RSTINMODE(!rst_n),
        .RSTM(!rst_n),
        .RSTP(!rst_n),
        .ACOUT(acout_unused),
        .BCOUT(bcout_unused),
        .CARRYCASCOUT(carrycascout_unused),
        .CARRYOUT(carryout_unused),
        .MULTSIGNOUT(multsignout_unused),
        .OVERFLOW(overflow_unused),
        .PATTERNBDETECT(patternbdetect_unused),
        .PATTERNDETECT(patterndetect_unused),
        .UNDERFLOW(underflow_unused),
        .XOROUT(xorout_unused)
    );

    assign result = primitive_p;
    assign cascade_out = primitive_pcout;
`else
    reg signed [44:0] product_pipeline;
    wire signed [47:0] extended_product = {
        {3{product_pipeline[44]}}, product_pipeline
    };

    assign result = cascade_enable
        ? extended_product + cascade_in : extended_product;
    assign cascade_out = result;

    always @(posedge clk) begin
        if (!rst_n)
            product_pipeline <= 45'sd0;
        else
            product_pipeline <= selected_activation * selected_weight;
    end
`endif

endmodule
