`timescale 1ns/1ps

module rotary_qk_pair_serial #(
    parameter integer DATA_WIDTH = 18,
    parameter integer CONSTANT_WIDTH = 16,
    parameter integer GROUP_WIDTH = 4,
    parameter integer TOKEN_WIDTH = 2,
    parameter integer HEAD_WIDTH = 4,
    parameter integer PAIR_WIDTH = 5
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    output wire ready_in,
    input  wire [GROUP_WIDTH-1:0] group_in,
    input  wire [TOKEN_WIDTH-1:0] token_in,
    input  wire [HEAD_WIDTH-1:0] head_in,
    input  wire [PAIR_WIDTH-1:0] pair_in,
    input  wire signed [DATA_WIDTH-1:0] query_first_q12,
    input  wire signed [DATA_WIDTH-1:0] query_second_q12,
    input  wire signed [DATA_WIDTH-1:0] key_first_q12,
    input  wire signed [DATA_WIDTH-1:0] key_second_q12,
    input  wire signed [CONSTANT_WIDTH-1:0] cosine_q15,
    input  wire signed [CONSTANT_WIDTH-1:0] sine_q15,
    output reg  valid_out,
    output reg  [GROUP_WIDTH-1:0] group_out,
    output reg  [TOKEN_WIDTH-1:0] token_out,
    output reg  [HEAD_WIDTH-1:0] head_out,
    output reg  [PAIR_WIDTH-1:0] pair_out,
    output reg  signed [DATA_WIDTH-1:0] query_first_rotated_q12,
    output reg  signed [DATA_WIDTH-1:0] query_second_rotated_q12,
    output reg  signed [DATA_WIDTH-1:0] key_first_rotated_q12,
    output reg  signed [DATA_WIDTH-1:0] key_second_rotated_q12
);

    localparam integer PRODUCT_WIDTH = DATA_WIDTH + CONSTANT_WIDTH;
    localparam integer NUMERATOR_WIDTH = PRODUCT_WIDTH + 1;
    localparam integer ROTARY_SHIFT = 15;

    reg stage_valid;
    reg [GROUP_WIDTH-1:0] stage_group;
    reg [TOKEN_WIDTH-1:0] stage_token;
    reg [HEAD_WIDTH-1:0] stage_head;
    reg [PAIR_WIDTH-1:0] stage_pair;
    reg signed [PRODUCT_WIDTH-1:0] q_first_cos;
    reg signed [PRODUCT_WIDTH-1:0] q_second_sin;
    reg signed [PRODUCT_WIDTH-1:0] q_second_cos;
    reg signed [PRODUCT_WIDTH-1:0] q_first_sin;
    reg signed [PRODUCT_WIDTH-1:0] k_first_cos;
    reg signed [PRODUCT_WIDTH-1:0] k_second_sin;
    reg signed [PRODUCT_WIDTH-1:0] k_second_cos;
    reg signed [PRODUCT_WIDTH-1:0] k_first_sin;

    reg signed [NUMERATOR_WIDTH-1:0] query_first_numerator;
    reg signed [NUMERATOR_WIDTH-1:0] query_second_numerator;
    reg signed [NUMERATOR_WIDTH-1:0] key_first_numerator;
    reg signed [NUMERATOR_WIDTH-1:0] key_second_numerator;
    reg signed [NUMERATOR_WIDTH-1:0] query_first_rounded;
    reg signed [NUMERATOR_WIDTH-1:0] query_second_rounded;
    reg signed [NUMERATOR_WIDTH-1:0] key_first_rounded;
    reg signed [NUMERATOR_WIDTH-1:0] key_second_rounded;
    reg signed [DATA_WIDTH-1:0] query_first_saturated;
    reg signed [DATA_WIDTH-1:0] query_second_saturated;
    reg signed [DATA_WIDTH-1:0] key_first_saturated;
    reg signed [DATA_WIDTH-1:0] key_second_saturated;

    assign ready_in = 1'b1;

    function automatic signed [NUMERATOR_WIDTH-1:0] rounded_q12;
        input signed [NUMERATOR_WIDTH-1:0] numerator;
        begin
            if (numerator >= 0)
                rounded_q12 = (numerator + 35'sd16384) >>> ROTARY_SHIFT;
            else
                rounded_q12 = -(((-numerator) + 35'sd16384)
                                >>> ROTARY_SHIFT);
        end
    endfunction

    function automatic signed [DATA_WIDTH-1:0] saturated_q12;
        input signed [NUMERATOR_WIDTH-1:0] value;
        begin
            if (value > 35'sd131071)
                saturated_q12 = 18'sd131071;
            else if (value < -35'sd131072)
                saturated_q12 = -18'sd131072;
            else
                saturated_q12 = value[DATA_WIDTH-1:0];
        end
    endfunction

    always @* begin
        query_first_numerator =
            $signed({q_first_cos[PRODUCT_WIDTH-1], q_first_cos})
            - $signed({q_second_sin[PRODUCT_WIDTH-1], q_second_sin});
        query_second_numerator =
            $signed({q_second_cos[PRODUCT_WIDTH-1], q_second_cos})
            + $signed({q_first_sin[PRODUCT_WIDTH-1], q_first_sin});
        key_first_numerator =
            $signed({k_first_cos[PRODUCT_WIDTH-1], k_first_cos})
            - $signed({k_second_sin[PRODUCT_WIDTH-1], k_second_sin});
        key_second_numerator =
            $signed({k_second_cos[PRODUCT_WIDTH-1], k_second_cos})
            + $signed({k_first_sin[PRODUCT_WIDTH-1], k_first_sin});
        query_first_rounded = rounded_q12(query_first_numerator);
        query_second_rounded = rounded_q12(query_second_numerator);
        key_first_rounded = rounded_q12(key_first_numerator);
        key_second_rounded = rounded_q12(key_second_numerator);
        query_first_saturated = saturated_q12(query_first_rounded);
        query_second_saturated = saturated_q12(query_second_rounded);
        key_first_saturated = saturated_q12(key_first_rounded);
        key_second_saturated = saturated_q12(key_second_rounded);
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            stage_valid <= 1'b0;
            valid_out <= 1'b0;
            stage_group <= {GROUP_WIDTH{1'b0}};
            stage_token <= {TOKEN_WIDTH{1'b0}};
            stage_head <= {HEAD_WIDTH{1'b0}};
            stage_pair <= {PAIR_WIDTH{1'b0}};
            group_out <= {GROUP_WIDTH{1'b0}};
            token_out <= {TOKEN_WIDTH{1'b0}};
            head_out <= {HEAD_WIDTH{1'b0}};
            pair_out <= {PAIR_WIDTH{1'b0}};
            query_first_rotated_q12 <= {DATA_WIDTH{1'b0}};
            query_second_rotated_q12 <= {DATA_WIDTH{1'b0}};
            key_first_rotated_q12 <= {DATA_WIDTH{1'b0}};
            key_second_rotated_q12 <= {DATA_WIDTH{1'b0}};
        end else begin
            stage_valid <= valid_in;
            valid_out <= stage_valid;
            if (valid_in) begin
                stage_group <= group_in;
                stage_token <= token_in;
                stage_head <= head_in;
                stage_pair <= pair_in;
                q_first_cos <= query_first_q12 * cosine_q15;
                q_second_sin <= query_second_q12 * sine_q15;
                q_second_cos <= query_second_q12 * cosine_q15;
                q_first_sin <= query_first_q12 * sine_q15;
                k_first_cos <= key_first_q12 * cosine_q15;
                k_second_sin <= key_second_q12 * sine_q15;
                k_second_cos <= key_second_q12 * cosine_q15;
                k_first_sin <= key_first_q12 * sine_q15;
            end
            if (stage_valid) begin
                group_out <= stage_group;
                token_out <= stage_token;
                head_out <= stage_head;
                pair_out <= stage_pair;
                query_first_rotated_q12 <= query_first_saturated;
                query_second_rotated_q12 <= query_second_saturated;
                key_first_rotated_q12 <= key_first_saturated;
                key_second_rotated_q12 <= key_second_saturated;
            end
        end
    end

endmodule
