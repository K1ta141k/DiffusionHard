`timescale 1ns/1ps

module attention_softmax_row_q16 #(
    parameter integer ROW_SIZE = 64,
    parameter integer SCORE_WIDTH = 18,
    parameter integer HEAD_WIDTH = 4,
    parameter integer QUERY_WIDTH = 6,
    parameter integer KEY_WIDTH = 6,
    parameter integer DIVIDER_WIDTH = 31,
    parameter LUT_FILE = "rtl/tensor_engine/exp_neg_q16_lut.hex"
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [HEAD_WIDTH-1:0] head_in,
    input  wire [QUERY_WIDTH-1:0] query_in,
    output wire start_ready,
    input  wire score_valid,
    output wire score_ready,
    input  wire signed [SCORE_WIDTH-1:0] score_q10,
    output wire probability_valid,
    input  wire probability_ready,
    output wire [HEAD_WIDTH-1:0] head_out,
    output wire [QUERY_WIDTH-1:0] query_out,
    output wire [KEY_WIDTH-1:0] key_out,
    output wire [15:0] probability_q16,
    output wire busy,
    output reg  done
);

    localparam [2:0] STATE_IDLE = 3'd0;
    localparam [2:0] STATE_LOAD = 3'd1;
    localparam [2:0] STATE_EXP = 3'd2;
    localparam [2:0] STATE_DIV_START = 3'd3;
    localparam [2:0] STATE_DIV_WAIT = 3'd4;
    localparam [2:0] STATE_OUTPUT = 3'd5;

    reg [2:0] state;
    reg [HEAD_WIDTH-1:0] active_head;
    reg [QUERY_WIDTH-1:0] active_query;
    reg [KEY_WIDTH:0] load_index;
    reg [KEY_WIDTH:0] exp_issue_index;
    reg exp_capture_valid;
    reg [KEY_WIDTH-1:0] exp_capture_index;
    reg [KEY_WIDTH-1:0] output_index;
    reg signed [SCORE_WIDTH-1:0] row_maximum;
    reg [21:0] exponential_sum;
    reg [15:0] reciprocal_q14;
    reg signed [SCORE_WIDTH-1:0] score_memory [0:ROW_SIZE-1];
    reg [15:0] exponential_memory [0:ROW_SIZE-1];

    reg signed [SCORE_WIDTH:0] score_delta;
    reg [SCORE_WIDTH:0] delta_magnitude;
    reg [10:0] exponential_address;
    wire [15:0] exponential_lut_value;
    wire divider_start = (state == STATE_DIV_START);
    wire divider_valid;
    wire [DIVIDER_WIDTH-1:0] divider_quotient;
    wire [DIVIDER_WIDTH-1:0] divider_dividend =
        31'd1073741824 + (exponential_sum >> 1);
    wire [DIVIDER_WIDTH-1:0] divider_divisor = exponential_sum;
    wire [31:0] probability_product =
        exponential_memory[output_index] * reciprocal_q14;
    wire [17:0] probability_rounded =
        (probability_product + 32'd8192) >> 14;

    assign start_ready = (state == STATE_IDLE);
    assign score_ready = (state == STATE_LOAD);
    assign probability_valid = (state == STATE_OUTPUT);
    assign head_out = active_head;
    assign query_out = active_query;
    assign key_out = output_index;
    assign probability_q16 = (probability_rounded > 18'd65535)
        ? 16'hffff : probability_rounded[15:0];
    assign busy = (state != STATE_IDLE);

    always @* begin
        score_delta = 0;
        delta_magnitude = 0;
        exponential_address = 11'd0;
        if (exp_issue_index < ROW_SIZE) begin
            score_delta = score_memory[exp_issue_index] - row_maximum;
            if (score_delta < -19'sd16384)
                exponential_address = 11'd1024;
            else if (score_delta < 0) begin
                delta_magnitude = -score_delta;
                exponential_address = (delta_magnitude + 19'd8) >> 4;
            end
        end
    end

    exp_neg_q16_lut_bram #(
        .LUT_FILE(LUT_FILE)
    ) exponential_lut (
        .clk(clk),
        .address(exponential_address),
        .value_q16(exponential_lut_value)
    );

    unsigned_divider_iterative #(
        .WIDTH(DIVIDER_WIDTH)
    ) reciprocal_divider (
        .clk(clk),
        .rst_n(rst_n),
        .start(divider_start),
        .ready(),
        .dividend(divider_dividend),
        .divisor(divider_divisor),
        .busy(),
        .valid_out(divider_valid),
        .quotient(divider_quotient),
        .remainder()
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_head <= {HEAD_WIDTH{1'b0}};
            active_query <= {QUERY_WIDTH{1'b0}};
            load_index <= {(KEY_WIDTH+1){1'b0}};
            exp_issue_index <= {(KEY_WIDTH+1){1'b0}};
            exp_capture_valid <= 1'b0;
            exp_capture_index <= {KEY_WIDTH{1'b0}};
            output_index <= {KEY_WIDTH{1'b0}};
            row_maximum <= {1'b1, {(SCORE_WIDTH-1){1'b0}}};
            exponential_sum <= 22'd0;
            reciprocal_q14 <= 16'd0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start && start_ready) begin
                state <= STATE_LOAD;
                active_head <= head_in;
                active_query <= query_in;
                load_index <= 0;
                row_maximum <= {1'b1, {(SCORE_WIDTH-1){1'b0}}};
            end else if (state == STATE_LOAD && score_valid) begin
                score_memory[load_index] <= score_q10;
                if (score_q10 > row_maximum)
                    row_maximum <= score_q10;
                if (load_index == ROW_SIZE-1) begin
                    state <= STATE_EXP;
                    exp_issue_index <= 0;
                    exp_capture_valid <= 1'b0;
                    exponential_sum <= 22'd0;
                end else begin
                    load_index <= load_index + 1'b1;
                end
            end else if (state == STATE_EXP) begin
                if (exp_capture_valid) begin
                    exponential_memory[exp_capture_index] <=
                        exponential_lut_value;
                    exponential_sum <= exponential_sum
                        + exponential_lut_value;
                    if (exp_capture_index == ROW_SIZE-1) begin
                        state <= STATE_DIV_START;
                        exp_capture_valid <= 1'b0;
                    end
                end
                if (exp_issue_index < ROW_SIZE) begin
                    exp_capture_valid <= 1'b1;
                    exp_capture_index <= exp_issue_index[KEY_WIDTH-1:0];
                    exp_issue_index <= exp_issue_index + 1'b1;
                end else if (!(exp_capture_valid
                               && exp_capture_index == ROW_SIZE-1)) begin
                    exp_capture_valid <= 1'b0;
                end
            end else if (state == STATE_DIV_START) begin
                state <= STATE_DIV_WAIT;
            end else if (state == STATE_DIV_WAIT && divider_valid) begin
                reciprocal_q14 <= divider_quotient[15:0];
                output_index <= 0;
                state <= STATE_OUTPUT;
            end else if (state == STATE_OUTPUT && probability_ready) begin
                if (output_index == ROW_SIZE-1) begin
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else begin
                    output_index <= output_index + 1'b1;
                end
            end
        end
    end

endmodule
