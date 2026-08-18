`timescale 1ns/1ps

module philox_noisy_argmax_stream (
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

    input  logic               model_valid,
    output logic               model_ready,
    input  logic signed [32:0] model_score0_q10,
    input  logic signed [32:0] model_score1_q10,

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

    localparam logic [2:0] STATUS_OK = 3'd0;
    localparam logic [2:0] STATUS_INVALID_POSITION_COUNT = 3'd1;
    localparam logic [2:0] STATUS_INVALID_VOCABULARY_SIZE = 3'd2;
    localparam logic [2:0] STATUS_INVALID_MASK_TOKEN = 3'd3;
    localparam logic [2:0] STATUS_RNG_ERROR = 3'd4;
    localparam logic [2:0] STATUS_REDUCER_ERROR = 3'd5;

    logic rng_start_ready;
    logic rng_score_valid;
    logic rng_score_ready;
    logic [1:0] rng_score_valid_mask;
    logic [5:0] rng_score_position;
    logic [15:0] rng_score_token_base;
    logic signed [15:0] rng_score0_q10;
    logic signed [15:0] rng_score1_q10;
    logic rng_busy;
    logic rng_done;
    logic [1:0] rng_status;

    logic reducer_start_ready;
    logic reducer_in_valid;
    logic reducer_in_ready;
    logic reducer_in_last;
    logic reducer_busy;
    logic reducer_done;
    logic [1:0] reducer_status;

    logic active;
    logic rng_complete;
    logic reducer_complete;
    logic [6:0] position_count_reg;
    logic [15:0] vocabulary_size_reg;
    logic command_valid;
    logic start_fire;
    logic [16:0] score_pair_end;

    assign command_valid = position_count != 0 && position_count <= 64 &&
        vocabulary_size != 0 && mask_token_id < vocabulary_size;
    assign start_ready = !active && rng_start_ready && reducer_start_ready;
    assign start_fire = start_valid && start_ready;
    assign busy = active;

    assign reducer_in_valid = rng_score_valid && model_valid;
    assign rng_score_ready = reducer_in_ready && model_valid;
    assign model_ready = reducer_in_ready && rng_score_valid;
    assign score_pair_end = {1'b0, rng_score_token_base} +
        rng_score_valid_mask[0] + rng_score_valid_mask[1];
    assign reducer_in_last =
        rng_score_position + 1'b1 == position_count_reg &&
        score_pair_end == {1'b0, vocabulary_size_reg};

    philox_gumbel_farm_stream rng (
        .clk,
        .rst_n,
        .start_valid(start_fire && command_valid),
        .start_ready(rng_start_ready),
        .position_count,
        .vocabulary_size,
        .evaluation_id,
        .stream_id,
        .seed_low,
        .seed_high,
        .score_valid(rng_score_valid),
        .score_ready(rng_score_ready),
        .score_valid_mask(rng_score_valid_mask),
        .score_position(rng_score_position),
        .score_token_base(rng_score_token_base),
        .score0_q10(rng_score0_q10),
        .score1_q10(rng_score1_q10),
        .busy(rng_busy),
        .done(rng_done),
        .status(rng_status)
    );

    ordered_noisy_argmax_reducer reducer (
        .clk,
        .rst_n,
        .start_valid(start_fire && command_valid),
        .start_ready(reducer_start_ready),
        .position_count,
        .mask_token_id,
        .in_valid(reducer_in_valid),
        .in_ready(reducer_in_ready),
        .in_last(reducer_in_last),
        .in_valid_mask(rng_score_valid_mask),
        .in_position(rng_score_position),
        .in_token_base(rng_score_token_base),
        .in_model_score0_q10(model_score0_q10),
        .in_model_score1_q10(model_score1_q10),
        .in_noise0_q10(rng_score0_q10),
        .in_noise1_q10(rng_score1_q10),
        .out_valid(candidate_valid),
        .out_ready(candidate_ready),
        .out_position(candidate_position),
        .out_candidate_valid(candidate_id_valid),
        .out_candidate_id(candidate_id),
        .out_candidate_score_q10(candidate_score_q10),
        .busy(reducer_busy),
        .done(reducer_done),
        .status(reducer_status)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            active <= 1'b0;
            rng_complete <= 1'b0;
            reducer_complete <= 1'b0;
            position_count_reg <= 0;
            vocabulary_size_reg <= 0;
            done <= 1'b0;
            status <= STATUS_OK;
        end else begin
            done <= 1'b0;
            if (start_fire) begin
                status <= STATUS_OK;
                rng_complete <= 1'b0;
                reducer_complete <= 1'b0;
                if (position_count == 0 || position_count > 64) begin
                    done <= 1'b1;
                    status <= STATUS_INVALID_POSITION_COUNT;
                end else if (vocabulary_size == 0) begin
                    done <= 1'b1;
                    status <= STATUS_INVALID_VOCABULARY_SIZE;
                end else if (mask_token_id >= vocabulary_size) begin
                    done <= 1'b1;
                    status <= STATUS_INVALID_MASK_TOKEN;
                end else begin
                    position_count_reg <= position_count;
                    vocabulary_size_reg <= vocabulary_size;
                    active <= 1'b1;
                end
            end else if (active) begin
                if (rng_done) begin
                    rng_complete <= 1'b1;
                end
                if (reducer_done) begin
                    reducer_complete <= 1'b1;
                end
                if (
                    (rng_complete || rng_done) &&
                    (reducer_complete || reducer_done)
                ) begin
                    active <= 1'b0;
                    rng_complete <= 1'b0;
                    reducer_complete <= 1'b0;
                    done <= 1'b1;
                    if (rng_status != 0) begin
                        status <= STATUS_RNG_ERROR;
                    end else if (reducer_status != 0) begin
                        status <= STATUS_REDUCER_ERROR;
                    end
                end
            end
        end
    end

endmodule
