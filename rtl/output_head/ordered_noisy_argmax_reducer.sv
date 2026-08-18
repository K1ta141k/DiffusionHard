`timescale 1ns/1ps

module ordered_noisy_argmax_reducer #(
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
    localparam logic [1:0] STATUS_INVALID_STREAM_ORDER = 2'd2;

    logic [6:0] position_count_reg;
    logic [15:0] mask_token_id_reg;
    logic accepting;
    logic current_valid;
    logic [5:0] current_position;
    logic current_best_valid;
    logic signed [33:0] current_best_score;
    logic [15:0] current_best_token;
    logic final_pending;
    logic finish_after_output;

    logic output_slot_ready;
    logic stream_order_valid;
    logic same_position;
    logic next_position;
    logic pair_base_valid;
    logic signed [33:0] pair_base_score;
    logic [15:0] pair_base_token;
    logic pair_best_valid;
    logic signed [33:0] pair_best_score;
    logic [15:0] pair_best_token;
    logic signed [33:0] lane0_score;
    logic signed [33:0] lane1_score;
    logic lane0_eligible;
    logic lane1_eligible;

    assign start_ready = !busy;
    assign output_slot_ready = !out_valid || out_ready;
    assign same_position = current_valid && in_position == current_position;
    assign next_position = current_valid &&
        in_position == current_position + 1'b1;
    assign stream_order_valid =
        in_position < position_count_reg &&
        (
            (!current_valid && in_position == 0) ||
            same_position || next_position
        ) &&
        (!in_last || in_position + 1'b1 == position_count_reg);
    assign in_ready = busy && accepting && !final_pending &&
        (
            !stream_order_valid ||
            (
                (!next_position || output_slot_ready) &&
                (!in_last || next_position || output_slot_ready)
            )
        );

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
        pair_base_valid = same_position && current_best_valid;
        pair_base_score = current_best_score;
        pair_base_token = current_best_token;
        pair_best_valid = pair_base_valid;
        pair_best_score = pair_base_score;
        pair_best_token = pair_base_token;
        if (
            lane0_eligible &&
            (
                !pair_best_valid || lane0_score > pair_best_score ||
                (
                    lane0_score == pair_best_score &&
                    in_token_base < pair_best_token
                )
            )
        ) begin
            pair_best_valid = 1'b1;
            pair_best_score = lane0_score;
            pair_best_token = in_token_base;
        end
        if (
            lane1_eligible &&
            (
                !pair_best_valid || lane1_score > pair_best_score ||
                (
                    lane1_score == pair_best_score &&
                    in_token_base + 1'b1 < pair_best_token
                )
            )
        ) begin
            pair_best_valid = 1'b1;
            pair_best_score = lane1_score;
            pair_best_token = in_token_base + 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            position_count_reg <= 0;
            mask_token_id_reg <= 0;
            accepting <= 1'b0;
            current_valid <= 1'b0;
            current_position <= 0;
            current_best_valid <= 1'b0;
            current_best_score <= 0;
            current_best_token <= 0;
            final_pending <= 1'b0;
            finish_after_output <= 1'b0;
            out_valid <= 1'b0;
            out_position <= 0;
            out_candidate_valid <= 1'b0;
            out_candidate_id <= 0;
            out_candidate_score_q10 <= 0;
            busy <= 1'b0;
            done <= 1'b0;
            status <= STATUS_OK;
        end else begin
            done <= 1'b0;

            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
                if (finish_after_output) begin
                    finish_after_output <= 1'b0;
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end

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
                        accepting <= 1'b1;
                        current_valid <= 1'b0;
                        current_position <= 0;
                        current_best_valid <= 1'b0;
                        final_pending <= 1'b0;
                        finish_after_output <= 1'b0;
                        out_valid <= 1'b0;
                        busy <= 1'b1;
                    end
                end
            end else if (final_pending && output_slot_ready) begin
                out_valid <= 1'b1;
                out_position <= current_position;
                out_candidate_valid <= current_best_valid;
                out_candidate_id <= current_best_token;
                out_candidate_score_q10 <= current_best_score;
                current_valid <= 1'b0;
                final_pending <= 1'b0;
                finish_after_output <= 1'b1;
            end else if (in_valid && in_ready) begin
                if (!stream_order_valid) begin
                    accepting <= 1'b0;
                    current_valid <= 1'b0;
                    busy <= 1'b0;
                    done <= 1'b1;
                    status <= STATUS_INVALID_STREAM_ORDER;
                end else if (!current_valid) begin
                    current_valid <= 1'b1;
                    current_position <= in_position;
                    current_best_valid <= pair_best_valid;
                    current_best_score <= pair_best_score;
                    current_best_token <= pair_best_token;
                    if (in_last) begin
                        out_valid <= 1'b1;
                        out_position <= in_position;
                        out_candidate_valid <= pair_best_valid;
                        out_candidate_id <= pair_best_token;
                        out_candidate_score_q10 <= pair_best_score;
                        current_valid <= 1'b0;
                        accepting <= 1'b0;
                        finish_after_output <= 1'b1;
                    end
                end else if (same_position) begin
                    current_best_valid <= pair_best_valid;
                    current_best_score <= pair_best_score;
                    current_best_token <= pair_best_token;
                    if (in_last) begin
                        out_valid <= 1'b1;
                        out_position <= current_position;
                        out_candidate_valid <= pair_best_valid;
                        out_candidate_id <= pair_best_token;
                        out_candidate_score_q10 <= pair_best_score;
                        current_valid <= 1'b0;
                        accepting <= 1'b0;
                        finish_after_output <= 1'b1;
                    end
                end else begin
                    out_valid <= 1'b1;
                    out_position <= current_position;
                    out_candidate_valid <= current_best_valid;
                    out_candidate_id <= current_best_token;
                    out_candidate_score_q10 <= current_best_score;
                    current_valid <= 1'b1;
                    current_position <= in_position;
                    current_best_valid <= pair_best_valid;
                    current_best_score <= pair_best_score;
                    current_best_token <= pair_best_token;
                    if (in_last) begin
                        accepting <= 1'b0;
                        final_pending <= 1'b1;
                    end
                end
            end
        end
    end

endmodule
