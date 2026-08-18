`timescale 1ns/1ps

module layer_norm_q12_group #(
    parameter integer INPUT_SIZE = 768,
    parameter integer M_LANES = 4,
    parameter integer INPUT_WIDTH = 24,
    parameter integer OUTPUT_WIDTH = 18,
    parameter integer GROUP_WIDTH = 4,
    parameter integer CHANNEL_WIDTH = (INPUT_SIZE <= 1)
        ? 1 : $clog2(INPUT_SIZE),
    parameter integer SUM_WIDTH = 35,
    parameter integer SUM_SQUARE_WIDTH = 58,
    parameter integer INVERSE_WIDTH = 20,
    parameter integer DIVIDER_WIDTH = 58
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [GROUP_WIDTH-1:0] group_in,
    output wire start_ready,
    input  wire start_replay,
    input  wire final_replay,
    output wire replay_ready,
    input  wire input_valid,
    output wire input_ready,
    input  wire [M_LANES*INPUT_WIDTH-1:0] input_q10_packed,
    output reg  output_valid,
    output reg  [GROUP_WIDTH-1:0] output_group,
    output reg  [CHANNEL_WIDTH-1:0] output_channel,
    output reg  [M_LANES*OUTPUT_WIDTH-1:0] output_q12_packed,
    output reg  busy,
    output reg  done
);

    localparam [3:0] STATE_IDLE = 4'd0;
    localparam [3:0] STATE_STATS = 4'd1;
    localparam [3:0] STATE_DIV_MEAN_START = 4'd2;
    localparam [3:0] STATE_DIV_MEAN_WAIT = 4'd3;
    localparam [3:0] STATE_DIV_M2_START = 4'd4;
    localparam [3:0] STATE_DIV_M2_WAIT = 4'd5;
    localparam [3:0] STATE_SQRT_START = 4'd6;
    localparam [3:0] STATE_SQRT_WAIT = 4'd7;
    localparam [3:0] STATE_DIV_INV_START = 4'd8;
    localparam [3:0] STATE_DIV_INV_WAIT = 4'd9;
    localparam [3:0] STATE_WAIT_REPLAY = 4'd10;
    localparam [3:0] STATE_REPLAY = 4'd11;
    localparam integer NORMALIZE_PRODUCT_WIDTH = INPUT_WIDTH + INVERSE_WIDTH + 2;

    reg [3:0] state;
    reg [GROUP_WIDTH-1:0] active_group;
    reg [CHANNEL_WIDTH-1:0] channel_counter;
    reg active_final_replay;
    reg signed [SUM_WIDTH-1:0] sums [0:M_LANES-1];
    reg [SUM_SQUARE_WIDTH-1:0] sum_squares [0:M_LANES-1];
    reg signed [INPUT_WIDTH-1:0] means_q10 [0:M_LANES-1];
    reg [31:0] mean_squares_q20 [0:M_LANES-1];
    reg [31:0] variances_q20 [0:M_LANES-1];
    reg [21:0] sqrt_roots_q16 [0:M_LANES-1];
    reg [INVERSE_WIDTH-1:0] inverse_q18 [0:M_LANES-1];

    wire divider_start = (state == STATE_DIV_MEAN_START)
        || (state == STATE_DIV_M2_START)
        || (state == STATE_DIV_INV_START);
    wire [DIVIDER_WIDTH-1:0] divider_dividend [0:M_LANES-1];
    wire [DIVIDER_WIDTH-1:0] divider_divisor [0:M_LANES-1];
    wire [SUM_WIDTH-1:0] rounded_absolute_sum [0:M_LANES-1];
    wire divider_valid [0:M_LANES-1];
    wire [DIVIDER_WIDTH-1:0] divider_quotient [0:M_LANES-1];
    wire [M_LANES-1:0] divider_valid_vector;
    wire sqrt_start = (state == STATE_SQRT_START);
    wire sqrt_valid [0:M_LANES-1];
    wire [21:0] sqrt_root [0:M_LANES-1];
    wire [M_LANES-1:0] sqrt_valid_vector;
    wire [43:0] sqrt_radicand [0:M_LANES-1];
    wire [2*INPUT_WIDTH-1:0] mean_squared_q20 [0:M_LANES-1];

    reg signed [INPUT_WIDTH-1:0] input_value [0:M_LANES-1];
    reg [2*INPUT_WIDTH-1:0] input_square [0:M_LANES-1];
    reg signed [INPUT_WIDTH:0] deviation [0:M_LANES-1];
    reg signed [NORMALIZE_PRODUCT_WIDTH-1:0] normalize_product
        [0:M_LANES-1];
    reg signed [NORMALIZE_PRODUCT_WIDTH-1:0] normalize_rounded
        [0:M_LANES-1];
    reg signed [OUTPUT_WIDTH-1:0] normalized_value [0:M_LANES-1];
    genvar arithmetic_index;
    integer token_index;

    assign start_ready = (state == STATE_IDLE);
    assign replay_ready = (state == STATE_WAIT_REPLAY);
    assign input_ready = (state == STATE_STATS) || (state == STATE_REPLAY);

    generate
        for (arithmetic_index = 0; arithmetic_index < M_LANES;
             arithmetic_index = arithmetic_index + 1) begin : setup_arithmetic
            assign rounded_absolute_sum[arithmetic_index] =
                (sums[arithmetic_index] < 0)
                ? $unsigned(-sums[arithmetic_index]) + 35'd384
                : $unsigned(sums[arithmetic_index]) + 35'd384;
            assign divider_dividend[arithmetic_index] =
                (state == STATE_DIV_MEAN_START)
                ? {{(DIVIDER_WIDTH-SUM_WIDTH){1'b0}},
                   rounded_absolute_sum[arithmetic_index]}
                : (state == STATE_DIV_M2_START)
                    ? sum_squares[arithmetic_index] + 384
                    : (58'd17179869184
                       + (sqrt_roots_q16[arithmetic_index] >> 1));
            assign divider_divisor[arithmetic_index] =
                (state == STATE_DIV_INV_START)
                ? sqrt_roots_q16[arithmetic_index] : 768;
            assign divider_valid_vector[arithmetic_index] =
                divider_valid[arithmetic_index];
            assign sqrt_valid_vector[arithmetic_index] =
                sqrt_valid[arithmetic_index];
            assign sqrt_radicand[arithmetic_index] = {
                variances_q20[arithmetic_index] + 32'd10,
                12'b0
            };
            assign mean_squared_q20[arithmetic_index] =
                $signed(means_q10[arithmetic_index])
                * $signed(means_q10[arithmetic_index]);

            unsigned_divider_iterative #(
                .WIDTH(DIVIDER_WIDTH)
            ) divider (
                .clk(clk), .rst_n(rst_n), .start(divider_start), .ready(),
                .dividend(divider_dividend[arithmetic_index]),
                .divisor(divider_divisor[arithmetic_index]), .busy(),
                .valid_out(divider_valid[arithmetic_index]),
                .quotient(divider_quotient[arithmetic_index]), .remainder()
            );

            unsigned_sqrt_iterative #(
                .RADICAND_WIDTH(44)
            ) square_root (
                .clk(clk), .rst_n(rst_n), .start(sqrt_start), .ready(),
                .radicand(sqrt_radicand[arithmetic_index]),
                .busy(), .valid_out(sqrt_valid[arithmetic_index]),
                .root(sqrt_root[arithmetic_index]), .remainder()
            );
        end
    endgenerate

    always @* begin
        for (token_index = 0; token_index < M_LANES;
             token_index = token_index + 1) begin
            input_value[token_index] = $signed(input_q10_packed[
                token_index*INPUT_WIDTH +: INPUT_WIDTH
            ]);
            input_square[token_index] = input_value[token_index]
                * input_value[token_index];
            deviation[token_index] = input_value[token_index]
                - means_q10[token_index];
            normalize_product[token_index] = deviation[token_index]
                * $signed({1'b0, inverse_q18[token_index]});
            if (normalize_product[token_index] >= 0)
                normalize_rounded[token_index] =
                    (normalize_product[token_index] + 32768) >>> 16;
            else
                normalize_rounded[token_index] = -(
                    ((-normalize_product[token_index]) + 32768) >>> 16
                );
            if (normalize_rounded[token_index] > ((1 << 17)-1))
                normalized_value[token_index] = (1 << 17)-1;
            else if (normalize_rounded[token_index] < -(1 << 17))
                normalized_value[token_index] = -(1 << 17);
            else
                normalized_value[token_index] =
                    normalize_rounded[token_index][OUTPUT_WIDTH-1:0];
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_group <= {GROUP_WIDTH{1'b0}};
            channel_counter <= {CHANNEL_WIDTH{1'b0}};
            active_final_replay <= 1'b0;
            output_valid <= 1'b0;
            output_group <= {GROUP_WIDTH{1'b0}};
            output_channel <= {CHANNEL_WIDTH{1'b0}};
            output_q12_packed <= {M_LANES*OUTPUT_WIDTH{1'b0}};
            busy <= 1'b0;
            done <= 1'b0;
            for (token_index = 0; token_index < M_LANES;
                 token_index = token_index + 1) begin
                sums[token_index] <= {SUM_WIDTH{1'b0}};
                sum_squares[token_index] <= {SUM_SQUARE_WIDTH{1'b0}};
                means_q10[token_index] <= {INPUT_WIDTH{1'b0}};
                mean_squares_q20[token_index] <= 32'd0;
                variances_q20[token_index] <= 32'd0;
                sqrt_roots_q16[token_index] <= 22'd0;
                inverse_q18[token_index] <= {INVERSE_WIDTH{1'b0}};
            end
        end else begin
            output_valid <= 1'b0;
            done <= 1'b0;
            if (start && start_ready) begin
                state <= STATE_STATS;
                active_group <= group_in;
                channel_counter <= {CHANNEL_WIDTH{1'b0}};
                busy <= 1'b1;
                for (token_index = 0; token_index < M_LANES;
                     token_index = token_index + 1) begin
                    sums[token_index] <= {SUM_WIDTH{1'b0}};
                    sum_squares[token_index] <= {SUM_SQUARE_WIDTH{1'b0}};
                end
            end else if (state == STATE_STATS && input_valid) begin
                for (token_index = 0; token_index < M_LANES;
                     token_index = token_index + 1) begin
                    sums[token_index] <= sums[token_index] + input_value[token_index];
                    sum_squares[token_index] <= sum_squares[token_index]
                        + input_square[token_index];
                end
                if (channel_counter == INPUT_SIZE-1) begin
                    channel_counter <= {CHANNEL_WIDTH{1'b0}};
                    state <= STATE_DIV_MEAN_START;
                end else begin
                    channel_counter <= channel_counter + 1'b1;
                end
            end else if (state == STATE_DIV_MEAN_START) begin
                state <= STATE_DIV_MEAN_WAIT;
            end else if (state == STATE_DIV_MEAN_WAIT
                         && &divider_valid_vector) begin
                for (token_index = 0; token_index < M_LANES;
                     token_index = token_index + 1)
                    means_q10[token_index] <= sums[token_index] < 0
                        ? -divider_quotient[token_index][INPUT_WIDTH-1:0]
                        : divider_quotient[token_index][INPUT_WIDTH-1:0];
                state <= STATE_DIV_M2_START;
            end else if (state == STATE_DIV_M2_START) begin
                state <= STATE_DIV_M2_WAIT;
            end else if (state == STATE_DIV_M2_WAIT
                         && &divider_valid_vector) begin
                for (token_index = 0; token_index < M_LANES;
                     token_index = token_index + 1)
                    mean_squares_q20[token_index] <=
                        divider_quotient[token_index][31:0];
                state <= STATE_SQRT_START;
                for (token_index = 0; token_index < M_LANES;
                     token_index = token_index + 1) begin
                    if (|divider_quotient[token_index][DIVIDER_WIDTH-1:32])
                        variances_q20[token_index] <= 32'hffffffff;
                    else if (divider_quotient[token_index][31:0]
                             <= mean_squared_q20[token_index])
                        variances_q20[token_index] <= 32'd0;
                    else
                        variances_q20[token_index] <=
                            divider_quotient[token_index][31:0]
                            - mean_squared_q20[token_index][31:0];
                end
            end else if (state == STATE_SQRT_START) begin
                state <= STATE_SQRT_WAIT;
            end else if (state == STATE_SQRT_WAIT && &sqrt_valid_vector) begin
                for (token_index = 0; token_index < M_LANES;
                     token_index = token_index + 1)
                    sqrt_roots_q16[token_index] <= sqrt_root[token_index];
                state <= STATE_DIV_INV_START;
            end else if (state == STATE_DIV_INV_START) begin
                state <= STATE_DIV_INV_WAIT;
            end else if (state == STATE_DIV_INV_WAIT
                         && &divider_valid_vector) begin
                for (token_index = 0; token_index < M_LANES;
                     token_index = token_index + 1)
                    inverse_q18[token_index] <=
                        divider_quotient[token_index][INVERSE_WIDTH-1:0];
                state <= STATE_WAIT_REPLAY;
            end else if (state == STATE_WAIT_REPLAY && start_replay) begin
                active_final_replay <= final_replay;
                channel_counter <= {CHANNEL_WIDTH{1'b0}};
                state <= STATE_REPLAY;
            end else if (state == STATE_REPLAY && input_valid) begin
                output_valid <= 1'b1;
                output_group <= active_group;
                output_channel <= channel_counter;
                for (token_index = 0; token_index < M_LANES;
                     token_index = token_index + 1)
                    output_q12_packed[
                        token_index*OUTPUT_WIDTH +: OUTPUT_WIDTH
                    ] <= normalized_value[token_index];
                if (channel_counter == INPUT_SIZE-1) begin
                    channel_counter <= {CHANNEL_WIDTH{1'b0}};
                    if (active_final_replay) begin
                        state <= STATE_IDLE;
                        busy <= 1'b0;
                        done <= 1'b1;
                    end else begin
                        state <= STATE_WAIT_REPLAY;
                    end
                end else begin
                    channel_counter <= channel_counter + 1'b1;
                end
            end
        end
    end

endmodule
