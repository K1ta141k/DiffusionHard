`timescale 1ns/1ps

module noisy_argmax_reducer #(
    parameter integer MAX_POSITIONS = 64
) (
    input  logic               clk,
    input  logic               rst_n,

    input  logic               start_valid,
    output logic               start_ready,
    input  logic [6:0]         position_count,
    input  logic [15:0]        mask_token_id,

    input  logic               in_valid,
    output logic               in_ready,
    input  logic               in_last,
    input  logic [1:0]         in_valid_mask,
    input  logic [5:0]         in_position,
    input  logic [15:0]        in_token_base,
    input  logic signed [32:0] in_model_score0_q10,
    input  logic signed [32:0] in_model_score1_q10,
    input  logic signed [15:0] in_noise0_q10,
    input  logic signed [15:0] in_noise1_q10,

    output logic               out_valid,
    input  logic               out_ready,
    output logic [5:0]         out_position,
    output logic               out_candidate_valid,
    output logic [15:0]        out_candidate_id,
    output logic signed [33:0] out_candidate_score_q10,

    output logic               busy,
    output logic               done,
    output logic [1:0]         status
);

    localparam logic [1:0] STATUS_OK = 2'd0;
    localparam logic [1:0] STATUS_INVALID_POSITION_COUNT = 2'd1;
    localparam logic [1:0] STATUS_INVALID_STREAM_POSITION = 2'd2;

    logic [6:0] position_count_reg;
    logic [15:0] mask_token_id_reg;
    logic best_valid [0:MAX_POSITIONS-1];
    logic signed [33:0] best_score [0:MAX_POSITIONS-1];
    logic [15:0] best_token [0:MAX_POSITIONS-1];
    logic emitting;
    logic [5:0] emit_position;
    integer reset_index;

    logic signed [33:0] lane0_score;
    logic signed [33:0] lane1_score;
    logic lane0_eligible;
    logic lane1_eligible;
    logic next_best_valid;
    logic signed [33:0] next_best_score;
    logic [15:0] next_best_token;

    assign start_ready = !busy;
    assign in_ready = busy && !emitting;
    assign out_valid = emitting;
    assign out_position = emit_position;
    assign out_candidate_valid = best_valid[emit_position];
    assign out_candidate_id = best_token[emit_position];
    assign out_candidate_score_q10 = best_score[emit_position];

    assign lane0_score =
        {in_model_score0_q10[32], in_model_score0_q10} +
        {{18{in_noise0_q10[15]}}, in_noise0_q10};
    assign lane1_score =
        {in_model_score1_q10[32], in_model_score1_q10} +
        {{18{in_noise1_q10[15]}}, in_noise1_q10};
    assign lane0_eligible = in_valid_mask[0] &&
        in_token_base != mask_token_id_reg;
    assign lane1_eligible = in_valid_mask[1] &&
        in_token_base + 1'b1 != mask_token_id_reg;

    always_comb begin
        next_best_valid = best_valid[in_position];
        next_best_score = best_score[in_position];
        next_best_token = best_token[in_position];
        if (
            lane0_eligible &&
            (
                !next_best_valid || lane0_score > next_best_score ||
                (
                    lane0_score == next_best_score &&
                    in_token_base < next_best_token
                )
            )
        ) begin
            next_best_valid = 1'b1;
            next_best_score = lane0_score;
            next_best_token = in_token_base;
        end
        if (
            lane1_eligible &&
            (
                !next_best_valid || lane1_score > next_best_score ||
                (
                    lane1_score == next_best_score &&
                    in_token_base + 1'b1 < next_best_token
                )
            )
        ) begin
            next_best_valid = 1'b1;
            next_best_score = lane1_score;
            next_best_token = in_token_base + 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            position_count_reg <= 0;
            mask_token_id_reg <= 0;
            emitting <= 1'b0;
            emit_position <= 0;
            busy <= 1'b0;
            done <= 1'b0;
            status <= STATUS_OK;
            for (reset_index = 0; reset_index < MAX_POSITIONS;
                 reset_index = reset_index + 1) begin
                best_valid[reset_index] <= 1'b0;
                best_score[reset_index] <= 0;
                best_token[reset_index] <= 0;
            end
        end else begin
            done <= 1'b0;
            if (!busy) begin
                if (start_valid && start_ready) begin
                    status <= STATUS_OK;
                    if (
                        position_count == 0 ||
                        position_count > MAX_POSITIONS
                    ) begin
                        done <= 1'b1;
                        status <= STATUS_INVALID_POSITION_COUNT;
                    end else begin
                        position_count_reg <= position_count;
                        mask_token_id_reg <= mask_token_id;
                        emitting <= 1'b0;
                        emit_position <= 0;
                        busy <= 1'b1;
                        for (reset_index = 0; reset_index < MAX_POSITIONS;
                             reset_index = reset_index + 1) begin
                            best_valid[reset_index] <= 1'b0;
                        end
                    end
                end
            end else if (emitting) begin
                if (out_valid && out_ready) begin
                    if (emit_position + 1'b1 == position_count_reg) begin
                        emitting <= 1'b0;
                        busy <= 1'b0;
                        done <= 1'b1;
                    end else begin
                        emit_position <= emit_position + 1'b1;
                    end
                end
            end else if (in_valid && in_ready) begin
                if (in_position >= position_count_reg) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    status <= STATUS_INVALID_STREAM_POSITION;
                end else begin
                    best_valid[in_position] <= next_best_valid;
                    best_score[in_position] <= next_best_score;
                    best_token[in_position] <= next_best_token;
                    if (in_last) begin
                        emitting <= 1'b1;
                        emit_position <= 0;
                    end
                end
            end
        end
    end

endmodule
