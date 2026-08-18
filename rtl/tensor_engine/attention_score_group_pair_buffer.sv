`timescale 1ns/1ps

module attention_score_group_pair_buffer #(
    parameter integer N_LANES = 6,
    parameter integer SCORE_WIDTH = 18,
    parameter integer KEY_TILES = 11,
    parameter integer HALF_SCORES = 4 * N_LANES,
    parameter integer HALF_WIDTH = HALF_SCORES * SCORE_WIDTH
) (
    input  wire clk,
    input  wire rst_n,
    input  wire pair_tile_valid,
    output wire pair_tile_ready,
    input  wire [2:0] pair_group,
    input  wire [5:0] pair_key_tile,
    input  wire [2:0] pair_valid_keys,
    input  wire [2*HALF_WIDTH-1:0] pair_scores_q10_packed,
    output wire score_tile_valid,
    input  wire score_tile_ready,
    output wire [3:0] score_group,
    output wire [5:0] score_key_tile,
    output wire [2:0] score_valid_keys,
    output wire [HALF_WIDTH-1:0] scores_q10_packed,
    output wire busy,
    output reg  done
);

    localparam STATE_LOWER_STREAM = 1'b0;
    localparam STATE_UPPER_REPLAY = 1'b1;

    reg state;
    reg [2:0] active_pair_group;
    reg [5:0] upper_replay_tile;
    reg [HALF_WIDTH-1:0] upper_score_memory [0:KEY_TILES-1];

    assign pair_tile_ready = state == STATE_LOWER_STREAM
        && score_tile_ready;
    assign score_tile_valid = state == STATE_LOWER_STREAM
        ? pair_tile_valid : 1'b1;
    assign score_group = state == STATE_LOWER_STREAM
        ? {pair_group, 1'b0} : {active_pair_group, 1'b1};
    assign score_key_tile = state == STATE_LOWER_STREAM
        ? pair_key_tile : upper_replay_tile;
    assign score_valid_keys = state == STATE_LOWER_STREAM
        ? pair_valid_keys : ((upper_replay_tile == KEY_TILES-1) ? 3'd4 : 3'd6);
    assign scores_q10_packed = state == STATE_LOWER_STREAM
        ? pair_scores_q10_packed[0 +: HALF_WIDTH]
        : upper_score_memory[upper_replay_tile];
    assign busy = state != STATE_LOWER_STREAM || pair_tile_valid;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_LOWER_STREAM;
            active_pair_group <= 0;
            upper_replay_tile <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_LOWER_STREAM
                && pair_tile_valid && pair_tile_ready) begin
                if (pair_key_tile == 0)
                    active_pair_group <= pair_group;
                upper_score_memory[pair_key_tile] <=
                    pair_scores_q10_packed[HALF_WIDTH +: HALF_WIDTH];
                if (pair_key_tile == KEY_TILES-1) begin
                    upper_replay_tile <= 0;
                    state <= STATE_UPPER_REPLAY;
                end
            end else if (state == STATE_UPPER_REPLAY
                         && score_tile_ready) begin
                if (upper_replay_tile == KEY_TILES-1) begin
                    state <= STATE_LOWER_STREAM;
                    done <= 1'b1;
                end else begin
                    upper_replay_tile <= upper_replay_tile + 1'b1;
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && state == STATE_LOWER_STREAM && pair_tile_valid
            && pair_key_tile != 0 && pair_group != active_pair_group)
            $error("score group-pair changed before the prior pair completed");
`endif
    end

endmodule
