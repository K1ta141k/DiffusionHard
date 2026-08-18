`timescale 1ns/1ps

module hidden_canvas_mlp_frontend (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [3:0] group_in,
    input  wire [17:0] smoothing_reciprocal_q15,
    output wire [9:0] smoothing_reciprocal_channel,
    output wire start_ready,
    output wire canvas_read_valid,
    output wire [3:0] canvas_read_group,
    output wire [6:0] canvas_read_output_tile,
    input  wire canvas_read_data_valid,
    input  wire [4*6*24-1:0] canvas_read_q10_packed,
    output wire token_factor_valid,
    output wire [3:0] token_factor_group,
    output wire [4*16-1:0] token_factors_packed,
    output wire activation_load_valid,
    output wire [3:0] activation_load_group,
    output wire [4:0] activation_load_k_tile,
    output wire [4*32*8-1:0] activation_load_data,
    output wire busy,
    output reg  done
);

    localparam [2:0] STATE_IDLE = 3'd0;
    localparam [2:0] STATE_PASS0 = 3'd1;
    localparam [2:0] STATE_WAIT1 = 3'd2;
    localparam [2:0] STATE_PASS1 = 3'd3;
    localparam [2:0] STATE_WAIT2 = 3'd4;
    localparam [2:0] STATE_PASS2 = 3'd5;
    localparam [2:0] STATE_WAIT_DONE = 3'd6;

    reg [2:0] state;
    reg [3:0] active_group;
    wire replay_start_ready;
    wire replay_output_valid;
    wire replay_output_ready;
    wire [3:0] replay_output_group;
    wire [9:0] replay_output_channel;
    wire [4*24-1:0] replay_output_q10;
    wire replay_busy;
    wire replay_done;
    wire frontend_start_ready;
    wire frontend_busy;
    wire frontend_done;
    wire launch_initial = state == STATE_IDLE && start && start_ready;
    wire launch_replay1 = state == STATE_WAIT1 && replay_start_ready
        && replay_output_ready;
    wire launch_replay2 = state == STATE_WAIT2 && replay_start_ready
        && replay_output_ready;
    wire replay_start = launch_initial || launch_replay1 || launch_replay2;

    assign start_ready = state == STATE_IDLE && replay_start_ready
        && frontend_start_ready;
    assign busy = state != STATE_IDLE || replay_busy || frontend_busy;
    assign smoothing_reciprocal_channel = replay_output_channel;

    hidden_canvas_group_replay replay (
        .clk(clk), .rst_n(rst_n), .start(replay_start),
        .group_in(launch_initial ? group_in : active_group),
        .start_ready(replay_start_ready), .canvas_read_valid(canvas_read_valid),
        .canvas_read_group(canvas_read_group),
        .canvas_read_output_tile(canvas_read_output_tile),
        .canvas_read_data_valid(canvas_read_data_valid),
        .canvas_read_q10_packed(canvas_read_q10_packed),
        .output_valid(replay_output_valid),
        .output_ready(replay_output_ready),
        .output_group(replay_output_group),
        .output_channel(replay_output_channel),
        .output_q10_packed(replay_output_q10), .busy(replay_busy),
        .done(replay_done)
    );

    layer_norm_mlp_up_activation_frontend frontend (
        .clk(clk), .rst_n(rst_n), .start(launch_initial),
        .group_in(group_in), .start_ready(frontend_start_ready),
        .residual_input_valid(replay_output_valid),
        .residual_input_ready(replay_output_ready),
        .residual_q10_packed(replay_output_q10),
        .smoothing_reciprocal_q15(smoothing_reciprocal_q15),
        .token_factor_valid(token_factor_valid),
        .token_factor_group(token_factor_group),
        .token_factors_packed(token_factors_packed),
        .activation_load_valid(activation_load_valid),
        .activation_load_group(activation_load_group),
        .activation_load_k_tile(activation_load_k_tile),
        .activation_load_data(activation_load_data),
        .busy(frontend_busy), .done(frontend_done)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_group <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (launch_initial) begin
                active_group <= group_in;
                state <= STATE_PASS0;
            end else if (state == STATE_PASS0 && replay_done) begin
                state <= STATE_WAIT1;
            end else if (launch_replay1) begin
                state <= STATE_PASS1;
            end else if (state == STATE_PASS1 && replay_done) begin
                state <= STATE_WAIT2;
            end else if (launch_replay2) begin
                state <= STATE_PASS2;
            end else if (state == STATE_PASS2 && replay_done) begin
                state <= STATE_WAIT_DONE;
            end else if (state == STATE_WAIT_DONE && frontend_done) begin
                state <= STATE_IDLE;
                done <= 1'b1;
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && replay_output_valid
            && replay_output_group != active_group)
            $error("hidden canvas replay changed groups during norm2");
`endif
    end

endmodule
