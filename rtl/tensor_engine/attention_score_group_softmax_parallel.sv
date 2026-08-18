`timescale 1ns/1ps

module attention_score_group_softmax_parallel #(
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer SCORE_WIDTH = 18,
    parameter integer GROUP_WIDTH = 4,
    parameter integer KEY_TILE_WIDTH = 4,
    parameter LUT_FILE = "rtl/tensor_engine/exp_neg_q16_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire score_tile_valid,
    output wire score_tile_ready,
    input  wire [GROUP_WIDTH-1:0] score_tile_group,
    input  wire [KEY_TILE_WIDTH-1:0] score_key_tile,
    input  wire [2:0] score_valid_keys,
    input  wire [M_LANES*N_LANES*SCORE_WIDTH-1:0]
        scores_q10_packed,
    output wire probability_group_valid,
    input  wire probability_group_ready,
    output wire [GROUP_WIDTH-1:0] probability_group,
    output wire [M_LANES*64*16-1:0] probabilities_q16_packed,
    output wire busy,
    output reg  done
);

    localparam [2:0] STATE_LOAD = 3'd0;
    localparam [2:0] STATE_START = 3'd1;
    localparam [2:0] STATE_FEED = 3'd2;
    localparam [2:0] STATE_WAIT = 3'd3;
    localparam [2:0] STATE_OUTPUT = 3'd4;

    reg [2:0] state;
    reg [GROUP_WIDTH-1:0] active_group;
    reg [KEY_TILE_WIDTH-1:0] expected_key_tile;
    reg [6:0] feed_key;
    reg signed [SCORE_WIDTH-1:0] score_memory [0:M_LANES-1][0:63];
    reg [15:0] probability_memory [0:M_LANES-1][0:63];

    wire [M_LANES-1:0] row_start_ready;
    wire [M_LANES-1:0] row_score_ready;
    wire [M_LANES-1:0] row_probability_valid;
    wire [M_LANES-1:0] row_done;
    wire [5:0] row_probability_key [0:M_LANES-1];
    wire [15:0] row_probability [0:M_LANES-1];
    wire launch_rows = (state == STATE_START) && &row_start_ready;
    wire feed_rows = (state == STATE_FEED) && (feed_key < 64);

    integer query_index;
    integer key_index;
    genvar row_index;
    genvar output_query;
    genvar output_key;

    assign score_tile_ready = (state == STATE_LOAD);
    assign probability_group_valid = (state == STATE_OUTPUT);
    assign probability_group = active_group;
    assign busy = (state != STATE_LOAD) || (expected_key_tile != 0);

    generate
        for (output_query = 0; output_query < M_LANES;
             output_query = output_query + 1) begin : pack_query
            for (output_key = 0; output_key < 64;
                 output_key = output_key + 1) begin : pack_key
                assign probabilities_q16_packed[
                    (output_query*64+output_key)*16 +: 16
                ] = probability_memory[output_query][output_key];
            end
        end
        for (row_index = 0; row_index < M_LANES;
             row_index = row_index + 1) begin : row_engines
            attention_softmax_row_q16 #(
                .LUT_FILE(LUT_FILE)
            ) row_softmax (
                .clk(clk), .rst_n(rst_n), .start(launch_rows),
                .head_in(4'd0), .query_in({active_group, row_index[1:0]}),
                .start_ready(row_start_ready[row_index]),
                .score_valid(feed_rows),
                .score_ready(row_score_ready[row_index]),
                .score_q10(score_memory[row_index][feed_key[5:0]]),
                .probability_valid(row_probability_valid[row_index]),
                .probability_ready(1'b1), .head_out(), .query_out(),
                .key_out(row_probability_key[row_index]),
                .probability_q16(row_probability[row_index]),
                .busy(), .done(row_done[row_index])
            );
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_LOAD;
            active_group <= 0;
            expected_key_tile <= 0;
            feed_key <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_LOAD && score_tile_valid) begin
                if (expected_key_tile == 0)
                    active_group <= score_tile_group;
                for (query_index = 0; query_index < M_LANES;
                     query_index = query_index + 1)
                    for (key_index = 0; key_index < N_LANES;
                         key_index = key_index + 1)
                        if (key_index < score_valid_keys)
                            score_memory[query_index][
                                score_key_tile*N_LANES + key_index
                            ] <= scores_q10_packed[
                                (query_index*N_LANES+key_index)*SCORE_WIDTH
                                +: SCORE_WIDTH
                            ];
                if (score_key_tile == 10) begin
                    expected_key_tile <= 0;
                    state <= STATE_START;
                end else begin
                    expected_key_tile <= expected_key_tile + 1'b1;
                end
            end else if (state == STATE_START && &row_start_ready) begin
                feed_key <= 0;
                state <= STATE_FEED;
            end else if (state == STATE_FEED && feed_rows
                         && &row_score_ready) begin
                if (feed_key == 63)
                    state <= STATE_WAIT;
                else
                    feed_key <= feed_key + 1'b1;
            end else if (state == STATE_WAIT && &row_done) begin
                state <= STATE_OUTPUT;
            end else if (state == STATE_OUTPUT && probability_group_ready) begin
                state <= STATE_LOAD;
                done <= 1'b1;
            end

            for (query_index = 0; query_index < M_LANES;
                 query_index = query_index + 1)
                if (row_probability_valid[query_index])
                    probability_memory[query_index][
                        row_probability_key[query_index]
                    ] <= row_probability[query_index];
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && state == STATE_LOAD && score_tile_valid
            && score_key_tile != expected_key_tile)
            $error("parallel softmax score tile arrived out of order");
        if (rst_n && state == STATE_FEED && feed_rows
            && row_score_ready != {M_LANES{1'b1}})
            $error("parallel softmax rows lost lockstep");
`endif
    end

endmodule
