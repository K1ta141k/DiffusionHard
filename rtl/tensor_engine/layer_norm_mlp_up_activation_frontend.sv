`timescale 1ns/1ps

module layer_norm_mlp_up_activation_frontend #(
    parameter integer INPUT_SIZE = 768,
    parameter integer M_LANES = 4,
    parameter integer GROUP_WIDTH = 4,
    parameter integer K_TILE_WIDTH = ((INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(INPUT_SIZE / 32),
    parameter integer TOKEN_FACTOR_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [GROUP_WIDTH-1:0] group_in,
    output wire start_ready,
    input  wire residual_input_valid,
    output wire residual_input_ready,
    input  wire [M_LANES*24-1:0] residual_q10_packed,
    input  wire [17:0] smoothing_reciprocal_q15,
    output wire token_factor_valid,
    output wire [GROUP_WIDTH-1:0] token_factor_group,
    output wire [M_LANES*TOKEN_FACTOR_WIDTH-1:0]
        token_factors_packed,
    output wire activation_load_valid,
    output wire [GROUP_WIDTH-1:0] activation_load_group,
    output wire [K_TILE_WIDTH-1:0] activation_load_k_tile,
    output wire [M_LANES*32*8-1:0] activation_load_data,
    output wire busy,
    output reg  done
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_WAIT_FIRST_REPLAY = 2'd1;
    localparam [1:0] STATE_WAIT_SECOND_REPLAY = 2'd2;
    localparam [1:0] STATE_SECOND_REPLAY = 2'd3;

    reg [1:0] state;
    reg [GROUP_WIDTH-1:0] active_group;
    reg [17:0] smoothing_reciprocal_delay;

    wire norm_start_ready;
    wire norm_replay_ready;
    wire norm_input_ready;
    wire norm_output_valid;
    wire [GROUP_WIDTH-1:0] norm_output_group;
    wire [M_LANES*18-1:0] norm_output_q12_packed;
    wire norm_busy;
    wire norm_done;
    wire quantizer_start_ready;
    wire quantizer_pass2_ready;
    wire quantizer_input_ready;
    wire quantizer_busy;
    wire quantizer_done;

    wire accept_start = start && start_ready;
    wire launch_first_replay = (state == STATE_WAIT_FIRST_REPLAY)
        && norm_replay_ready && quantizer_start_ready;
    wire launch_second_replay = (state == STATE_WAIT_SECOND_REPLAY)
        && norm_replay_ready && quantizer_pass2_ready;

    assign start_ready = (state == STATE_IDLE) && norm_start_ready
        && quantizer_start_ready;
    assign residual_input_ready = norm_input_ready;
    assign busy = (state != STATE_IDLE) || norm_busy || quantizer_busy;

    layer_norm_q12_group #(
        .INPUT_SIZE(INPUT_SIZE),
        .M_LANES(M_LANES),
        .GROUP_WIDTH(GROUP_WIDTH)
    ) layer_norm (
        .clk(clk),
        .rst_n(rst_n),
        .start(accept_start),
        .group_in(group_in),
        .start_ready(norm_start_ready),
        .start_replay(launch_first_replay || launch_second_replay),
        .final_replay(launch_second_replay),
        .replay_ready(norm_replay_ready),
        .input_valid(residual_input_valid),
        .input_ready(norm_input_ready),
        .input_q10_packed(residual_q10_packed),
        .output_valid(norm_output_valid),
        .output_group(norm_output_group),
        .output_channel(),
        .output_q12_packed(norm_output_q12_packed),
        .busy(norm_busy),
        .done(norm_done)
    );

    mlp_up_activation_quantizer #(
        .INPUT_SIZE(INPUT_SIZE),
        .M_LANES(M_LANES),
        .TOKEN_FACTOR_WIDTH(TOKEN_FACTOR_WIDTH),
        .GROUP_WIDTH(GROUP_WIDTH),
        .K_TILE_WIDTH(K_TILE_WIDTH)
    ) activation_quantizer (
        .clk(clk),
        .rst_n(rst_n),
        .start(launch_first_replay),
        .group_in(active_group),
        .start_ready(quantizer_start_ready),
        .start_pass2(launch_second_replay),
        .pass2_ready(quantizer_pass2_ready),
        .input_valid(norm_output_valid),
        .input_ready(quantizer_input_ready),
        .normalized_q12_packed(norm_output_q12_packed),
        .smoothing_reciprocal_q15(smoothing_reciprocal_delay),
        .token_factor_valid(token_factor_valid),
        .token_factor_group(token_factor_group),
        .token_factors_packed(token_factors_packed),
        .activation_load_valid(activation_load_valid),
        .activation_load_group(activation_load_group),
        .activation_load_k_tile(activation_load_k_tile),
        .activation_load_data(activation_load_data),
        .busy(quantizer_busy),
        .done(quantizer_done)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_group <= {GROUP_WIDTH{1'b0}};
            smoothing_reciprocal_delay <= 18'd0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (residual_input_valid && residual_input_ready)
                smoothing_reciprocal_delay <= smoothing_reciprocal_q15;
            if (accept_start) begin
                active_group <= group_in;
                state <= STATE_WAIT_FIRST_REPLAY;
            end else if (launch_first_replay)
                state <= STATE_WAIT_SECOND_REPLAY;
            else if (launch_second_replay)
                state <= STATE_SECOND_REPLAY;
            else if (state == STATE_SECOND_REPLAY && quantizer_done) begin
                state <= STATE_IDLE;
                done <= 1'b1;
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && norm_output_valid && !quantizer_input_ready)
            $error("LayerNorm replay outran the activation quantizer");
        if (rst_n && norm_done && state != STATE_SECOND_REPLAY)
            $error("LayerNorm ended outside the final replay");
`endif
    end

endmodule
