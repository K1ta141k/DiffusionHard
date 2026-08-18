`timescale 1ns/1ps

module philox_accumulator_argmax_stream (
    input  logic               clk,
    input  logic               rst_n,

    input  logic               start_valid,
    output logic               start_ready,
    input  logic [6:0]         position_count,
    input  logic [15:0]        vocabulary_size,
    input  logic [15:0]        mask_token_id,
    input  logic [31:0]        evaluation_id,
    input  logic [31:0]        stream_id,
    input  logic [31:0]        seed_low,
    input  logic [31:0]        seed_high,

    input  logic               accumulator_valid,
    output logic               accumulator_ready,
    input  logic signed [31:0] accumulator0,
    input  logic signed [31:0] accumulator1,
    input  logic signed [31:0] multiplier0_q20,
    input  logic signed [31:0] multiplier1_q20,
    input  logic signed [31:0] bias0_q10,
    input  logic signed [31:0] bias1_q10,

    output logic               candidate_valid,
    input  logic               candidate_ready,
    output logic [5:0]         candidate_position,
    output logic               candidate_id_valid,
    output logic [15:0]        candidate_id,
    output logic signed [33:0] candidate_score_q10,

    output logic               busy,
    output logic               done,
    output logic [2:0]         status
);

    logic requantized_valid;
    logic requantized_ready;
    logic signed [32:0] model_score0_q10;
    logic signed [32:0] model_score1_q10;

    dual_requantizer_q20 requantizer (
        .clk,
        .rst_n,
        .input_valid(accumulator_valid),
        .input_ready(accumulator_ready),
        .accumulator0,
        .accumulator1,
        .multiplier0_q20,
        .multiplier1_q20,
        .bias0_q10,
        .bias1_q10,
        .output_valid(requantized_valid),
        .output_ready(requantized_ready),
        .model_score0_q10,
        .model_score1_q10
    );

    philox_noisy_argmax_stream engine (
        .clk,
        .rst_n,
        .start_valid,
        .start_ready,
        .position_count,
        .vocabulary_size,
        .mask_token_id,
        .evaluation_id,
        .stream_id,
        .seed_low,
        .seed_high,
        .model_valid(requantized_valid),
        .model_ready(requantized_ready),
        .model_score0_q10,
        .model_score1_q10,
        .candidate_valid,
        .candidate_ready,
        .candidate_position,
        .candidate_id_valid,
        .candidate_id,
        .candidate_score_q10,
        .busy,
        .done,
        .status
    );

endmodule
