`timescale 1ns/1ps

module dual_requantizer_q20 (
    input  logic               clk,
    input  logic               rst_n,

    input  logic               input_valid,
    output logic               input_ready,
    input  logic signed [31:0] accumulator0,
    input  logic signed [31:0] accumulator1,
    input  logic signed [31:0] multiplier0_q20,
    input  logic signed [31:0] multiplier1_q20,
    input  logic signed [31:0] bias0_q10,
    input  logic signed [31:0] bias1_q10,

    output logic               output_valid,
    input  logic               output_ready,
    output logic signed [32:0] model_score0_q10,
    output logic signed [32:0] model_score1_q10
);

    localparam logic signed [63:0] ROUND_HALF_Q20 = 64'sd524288;

    logic signed [63:0] product0;
    logic signed [63:0] product1;
    logic signed [63:0] rounded0;
    logic signed [63:0] rounded1;
    logic signed [31:0] requantized0;
    logic signed [31:0] requantized1;
    logic signed [32:0] next_model_score0;
    logic signed [32:0] next_model_score1;

    assign input_ready = !output_valid || output_ready;
    assign product0 = accumulator0 * multiplier0_q20;
    assign product1 = accumulator1 * multiplier1_q20;
    assign rounded0 = product0 >= 0
        ? (product0 + ROUND_HALF_Q20) >>> 20
        : -(((-product0) + ROUND_HALF_Q20) >>> 20);
    assign rounded1 = product1 >= 0
        ? (product1 + ROUND_HALF_Q20) >>> 20
        : -(((-product1) + ROUND_HALF_Q20) >>> 20);
    assign requantized0 = rounded0[31:0];
    assign requantized1 = rounded1[31:0];
    assign next_model_score0 =
        {requantized0[31], requantized0} + {bias0_q10[31], bias0_q10};
    assign next_model_score1 =
        {requantized1[31], requantized1} + {bias1_q10[31], bias1_q10};

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            output_valid <= 1'b0;
            model_score0_q10 <= 0;
            model_score1_q10 <= 0;
        end else if (input_ready) begin
            output_valid <= input_valid;
            if (input_valid) begin
                model_score0_q10 <= next_model_score0;
                model_score1_q10 <= next_model_score1;
            end
        end
    end

endmodule
