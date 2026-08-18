`timescale 1ns/1ps

module candidate_reveal_stream #(
    parameter integer MAX_POSITIONS = 256,
    parameter integer TOKEN_WIDTH = 16,
    parameter integer COUNT_WIDTH = $clog2(MAX_POSITIONS + 1),
    parameter integer INDEX_WIDTH = $clog2(MAX_POSITIONS)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [COUNT_WIDTH-1:0]       position_count,
    input  logic [32:0]                  reveal_threshold_q32,

    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic                         in_active,
    input  logic                         in_candidate_valid,
    input  logic [TOKEN_WIDTH-1:0]       in_token_id,
    input  logic [TOKEN_WIDTH-1:0]       in_candidate_id,
    input  logic [31:0]                  in_random_word,

    output logic                         out_valid,
    input  logic                         out_ready,
    output logic [INDEX_WIDTH-1:0]       out_position,
    output logic                         out_active,
    output logic                         out_candidate_valid,
    output logic [TOKEN_WIDTH-1:0]       out_token_id,
    output logic                         out_changed,

    output logic                         busy,
    output logic                         done,
    output logic                         invalidate_all,
    output logic [COUNT_WIDTH-1:0]       changed_count,
    output logic [1:0]                   status
);

    localparam logic [1:0] STATUS_OK = 2'd0;
    localparam logic [1:0] STATUS_INVALID_COUNT = 2'd1;
    localparam logic [1:0] STATUS_INVALID_THRESHOLD = 2'd2;
    localparam logic [32:0] PROBABILITY_ONE_Q32 = 33'h1_0000_0000;

    logic [COUNT_WIDTH-1:0] accepted_count;
    logic [COUNT_WIDTH-1:0] changed_accumulator;
    logic any_changed;
    logic reveal;
    logic transfer_changed;

    assign start_ready = !busy;
    assign in_ready = busy && out_ready;
    assign out_valid = busy && in_valid;
    assign out_position = accepted_count[INDEX_WIDTH-1:0];
    assign reveal = ({1'b0, in_random_word} < reveal_threshold_q32);
    assign transfer_changed = in_active && in_candidate_valid && reveal;
    assign out_changed = transfer_changed;
    assign out_active = in_active && !transfer_changed;
    assign out_candidate_valid = in_candidate_valid && !transfer_changed;
    assign out_token_id = transfer_changed ? in_candidate_id : in_token_id;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            invalidate_all <= 1'b0;
            changed_count <= '0;
            status <= STATUS_OK;
            accepted_count <= '0;
            changed_accumulator <= '0;
            any_changed <= 1'b0;
        end else begin
            done <= 1'b0;
            if (!busy) begin
                if (start_valid && start_ready) begin
                    accepted_count <= '0;
                    changed_accumulator <= '0;
                    changed_count <= '0;
                    any_changed <= 1'b0;
                    invalidate_all <= 1'b0;
                    status <= STATUS_OK;
                    if (position_count == 0 || position_count > MAX_POSITIONS) begin
                        done <= 1'b1;
                        status <= STATUS_INVALID_COUNT;
                    end else if (reveal_threshold_q32 > PROBABILITY_ONE_Q32) begin
                        done <= 1'b1;
                        status <= STATUS_INVALID_THRESHOLD;
                    end else begin
                        busy <= 1'b1;
                    end
                end
            end else if (in_valid && in_ready) begin
                if (transfer_changed) begin
                    changed_accumulator <= changed_accumulator + 1'b1;
                    any_changed <= 1'b1;
                end
                if (accepted_count + 1'b1 == position_count) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    invalidate_all <= any_changed || transfer_changed;
                    changed_count <= changed_accumulator + transfer_changed;
                    accepted_count <= '0;
                end else begin
                    accepted_count <= accepted_count + 1'b1;
                end
            end
        end
    end

endmodule
