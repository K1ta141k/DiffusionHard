`timescale 1ns/1ps

module mlp_up_activation_quantizer #(
    parameter integer INPUT_SIZE = 768,
    parameter integer M_LANES = 4,
    parameter integer INPUT_WIDTH = 18,
    parameter integer RECIPROCAL_WIDTH = 18,
    parameter integer RECIPROCAL_SHIFT = 15,
    parameter integer QUANT_MULTIPLIER_WIDTH = 16,
    parameter integer QUANT_SHIFT = 18,
    parameter integer TOKEN_FACTOR_WIDTH = 16,
    parameter integer GROUP_WIDTH = 4,
    parameter integer CHANNEL_WIDTH = (INPUT_SIZE <= 1)
        ? 1 : $clog2(INPUT_SIZE),
    parameter integer K_TILE_WIDTH = ((INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(INPUT_SIZE / 32),
    parameter integer DIVIDER_WIDTH = 26
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [GROUP_WIDTH-1:0] group_in,
    output wire start_ready,
    input  wire start_pass2,
    output wire pass2_ready,
    input  wire input_valid,
    output wire input_ready,
    input  wire [M_LANES*INPUT_WIDTH-1:0] normalized_q12_packed,
    input  wire [RECIPROCAL_WIDTH-1:0] smoothing_reciprocal_q15,
    output reg  token_factor_valid,
    output reg  [GROUP_WIDTH-1:0] token_factor_group,
    output reg  [M_LANES*TOKEN_FACTOR_WIDTH-1:0]
        token_factors_packed,
    output reg  activation_load_valid,
    output reg  [GROUP_WIDTH-1:0] activation_load_group,
    output reg  [K_TILE_WIDTH-1:0] activation_load_k_tile,
    output reg  [M_LANES*32*8-1:0] activation_load_data,
    output reg  busy,
    output reg  done
);

    localparam [2:0] STATE_IDLE = 3'd0;
    localparam [2:0] STATE_PASS1 = 3'd1;
    localparam [2:0] STATE_DIV_QUANT_START = 3'd2;
    localparam [2:0] STATE_DIV_QUANT_WAIT = 3'd3;
    localparam [2:0] STATE_DIV_FACTOR_START = 3'd4;
    localparam [2:0] STATE_DIV_FACTOR_WAIT = 3'd5;
    localparam [2:0] STATE_WAIT_PASS2 = 3'd6;
    localparam [2:0] STATE_PASS2 = 3'd7;
    localparam integer TRANSFORM_PRODUCT_WIDTH =
        INPUT_WIDTH + RECIPROCAL_WIDTH + 1;
    localparam integer QUANT_PRODUCT_WIDTH =
        INPUT_WIDTH + QUANT_MULTIPLIER_WIDTH + 1;

    reg [2:0] state;
    reg [GROUP_WIDTH-1:0] active_group;
    reg [CHANNEL_WIDTH-1:0] channel_counter;
    reg [INPUT_WIDTH-1:0] maxima [0:M_LANES-1];
    reg [QUANT_MULTIPLIER_WIDTH-1:0] quant_multipliers
        [0:M_LANES-1];
    reg [M_LANES*32*8-1:0] tile_buffer;

    reg signed [TRANSFORM_PRODUCT_WIDTH-1:0] transform_product
        [0:M_LANES-1];
    reg signed [TRANSFORM_PRODUCT_WIDTH-1:0] transform_rounded
        [0:M_LANES-1];
    reg signed [INPUT_WIDTH-1:0] transformed_value [0:M_LANES-1];
    reg [INPUT_WIDTH-1:0] transformed_abs [0:M_LANES-1];
    reg signed [QUANT_PRODUCT_WIDTH-1:0] quant_product
        [0:M_LANES-1];
    reg signed [QUANT_PRODUCT_WIDTH-1:0] quant_rounded
        [0:M_LANES-1];
    reg signed [7:0] quantized_value [0:M_LANES-1];

    wire divider_start = (state == STATE_DIV_QUANT_START)
        || (state == STATE_DIV_FACTOR_START);
    wire [DIVIDER_WIDTH-1:0] divider_dividend [0:M_LANES-1];
    wire [DIVIDER_WIDTH-1:0] divider_divisor [0:M_LANES-1];
    wire divider_valid [0:M_LANES-1];
    wire [DIVIDER_WIDTH-1:0] divider_quotient [0:M_LANES-1];
    wire [M_LANES-1:0] divider_valid_vector;
    genvar divider_index;
    integer token_index;
    integer slot_index;

    assign start_ready = (state == STATE_IDLE);
    assign pass2_ready = (state == STATE_WAIT_PASS2);
    assign input_ready = (state == STATE_PASS1) || (state == STATE_PASS2);

    generate
        for (divider_index = 0; divider_index < M_LANES;
             divider_index = divider_index + 1) begin : token_dividers
            assign divider_dividend[divider_index] =
                (state == STATE_DIV_QUANT_START)
                ? (maxima[divider_index] == 0 ? 0
                    : 26'd33292288 + (maxima[divider_index] >> 1))
                : (({{(DIVIDER_WIDTH-INPUT_WIDTH){1'b0}},
                      maxima[divider_index]} << 6) + 63);
            assign divider_divisor[divider_index] =
                (state == STATE_DIV_QUANT_START)
                ? (maxima[divider_index] == 0 ? 1 : maxima[divider_index])
                : 127;
            assign divider_valid_vector[divider_index] =
                divider_valid[divider_index];

            unsigned_divider_iterative #(
                .WIDTH(DIVIDER_WIDTH)
            ) divider (
                .clk(clk),
                .rst_n(rst_n),
                .start(divider_start),
                .ready(),
                .dividend(divider_dividend[divider_index]),
                .divisor(divider_divisor[divider_index]),
                .busy(),
                .valid_out(divider_valid[divider_index]),
                .quotient(divider_quotient[divider_index]),
                .remainder()
            );
        end
    endgenerate

    always @* begin
        for (token_index = 0; token_index < M_LANES;
             token_index = token_index + 1) begin
            transform_product[token_index] = $signed(
                normalized_q12_packed[
                    token_index*INPUT_WIDTH +: INPUT_WIDTH
                ]
            ) * $signed({1'b0, smoothing_reciprocal_q15});
            if (transform_product[token_index] >= 0)
                transform_rounded[token_index] =
                    (transform_product[token_index]
                     + (1 << (RECIPROCAL_SHIFT-1))) >>> RECIPROCAL_SHIFT;
            else
                transform_rounded[token_index] = -(
                    ((-transform_product[token_index])
                     + (1 << (RECIPROCAL_SHIFT-1))) >>> RECIPROCAL_SHIFT
                );
            if (transform_rounded[token_index] > ((1 << (INPUT_WIDTH-1))-1))
                transformed_value[token_index] = (1 << (INPUT_WIDTH-1))-1;
            else if (transform_rounded[token_index] < -(1 << (INPUT_WIDTH-1)))
                transformed_value[token_index] = -(1 << (INPUT_WIDTH-1));
            else
                transformed_value[token_index] =
                    transform_rounded[token_index][INPUT_WIDTH-1:0];
            transformed_abs[token_index] = transformed_value[token_index] < 0
                ? -transformed_value[token_index] : transformed_value[token_index];

            quant_product[token_index] = transformed_value[token_index]
                * $signed({1'b0, quant_multipliers[token_index]});
            if (quant_product[token_index] >= 0)
                quant_rounded[token_index] =
                    (quant_product[token_index] + (1 << (QUANT_SHIFT-1)))
                    >>> QUANT_SHIFT;
            else
                quant_rounded[token_index] = -(
                    ((-quant_product[token_index]) + (1 << (QUANT_SHIFT-1)))
                    >>> QUANT_SHIFT
                );
            if (quant_rounded[token_index] > 127)
                quantized_value[token_index] = 8'sd127;
            else if (quant_rounded[token_index] < -127)
                quantized_value[token_index] = -8'sd127;
            else
                quantized_value[token_index] =
                    quant_rounded[token_index][7:0];
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_group <= {GROUP_WIDTH{1'b0}};
            channel_counter <= {CHANNEL_WIDTH{1'b0}};
            tile_buffer <= {M_LANES*32*8{1'b0}};
            token_factor_valid <= 1'b0;
            token_factor_group <= {GROUP_WIDTH{1'b0}};
            token_factors_packed <=
                {M_LANES*TOKEN_FACTOR_WIDTH{1'b0}};
            activation_load_valid <= 1'b0;
            activation_load_group <= {GROUP_WIDTH{1'b0}};
            activation_load_k_tile <= {K_TILE_WIDTH{1'b0}};
            activation_load_data <= {M_LANES*32*8{1'b0}};
            busy <= 1'b0;
            done <= 1'b0;
            for (token_index = 0; token_index < M_LANES;
                 token_index = token_index + 1) begin
                maxima[token_index] <= {INPUT_WIDTH{1'b0}};
                quant_multipliers[token_index] <=
                    {QUANT_MULTIPLIER_WIDTH{1'b0}};
            end
        end else begin
            token_factor_valid <= 1'b0;
            activation_load_valid <= 1'b0;
            done <= 1'b0;

            if (start && start_ready) begin
                state <= STATE_PASS1;
                busy <= 1'b1;
                active_group <= group_in;
                channel_counter <= {CHANNEL_WIDTH{1'b0}};
                for (token_index = 0; token_index < M_LANES;
                     token_index = token_index + 1)
                    maxima[token_index] <= {INPUT_WIDTH{1'b0}};
            end else if (state == STATE_PASS1 && input_valid) begin
                for (token_index = 0; token_index < M_LANES;
                     token_index = token_index + 1)
                    if (transformed_abs[token_index] > maxima[token_index])
                        maxima[token_index] <= transformed_abs[token_index];
                if (channel_counter == INPUT_SIZE-1) begin
                    channel_counter <= {CHANNEL_WIDTH{1'b0}};
                    state <= STATE_DIV_QUANT_START;
                end else begin
                    channel_counter <= channel_counter + 1'b1;
                end
            end else if (state == STATE_DIV_QUANT_START) begin
                state <= STATE_DIV_QUANT_WAIT;
            end else if (state == STATE_DIV_QUANT_WAIT
                         && &divider_valid_vector) begin
                for (token_index = 0; token_index < M_LANES;
                     token_index = token_index + 1)
                    quant_multipliers[token_index] <=
                        divider_quotient[token_index][
                            QUANT_MULTIPLIER_WIDTH-1:0
                        ];
                state <= STATE_DIV_FACTOR_START;
            end else if (state == STATE_DIV_FACTOR_START) begin
                state <= STATE_DIV_FACTOR_WAIT;
            end else if (state == STATE_DIV_FACTOR_WAIT
                         && &divider_valid_vector) begin
                for (token_index = 0; token_index < M_LANES;
                     token_index = token_index + 1)
                    token_factors_packed[
                        token_index*TOKEN_FACTOR_WIDTH +: TOKEN_FACTOR_WIDTH
                    ] <= divider_quotient[token_index][
                        TOKEN_FACTOR_WIDTH-1:0
                    ];
                token_factor_valid <= 1'b1;
                token_factor_group <= active_group;
                state <= STATE_WAIT_PASS2;
            end else if (state == STATE_WAIT_PASS2 && start_pass2) begin
                channel_counter <= {CHANNEL_WIDTH{1'b0}};
                state <= STATE_PASS2;
            end else if (state == STATE_PASS2 && input_valid) begin
                for (slot_index = 0; slot_index < 32;
                     slot_index = slot_index + 1)
                    if (channel_counter[4:0] == slot_index)
                        for (token_index = 0; token_index < M_LANES;
                             token_index = token_index + 1)
                            tile_buffer[
                                (token_index*32 + slot_index)*8 +: 8
                            ] <= quantized_value[token_index];
                if (channel_counter[4:0] == 31) begin
                    activation_load_valid <= 1'b1;
                    activation_load_group <= active_group;
                    activation_load_k_tile <= channel_counter >> 5;
                    activation_load_data <= tile_buffer;
                    for (token_index = 0; token_index < M_LANES;
                         token_index = token_index + 1)
                        activation_load_data[
                            (token_index*32 + 31)*8 +: 8
                        ] <= quantized_value[token_index];
                end
                if (channel_counter == INPUT_SIZE-1) begin
                    state <= STATE_IDLE;
                    busy <= 1'b0;
                    done <= 1'b1;
                    channel_counter <= {CHANNEL_WIDTH{1'b0}};
                end else begin
                    channel_counter <= channel_counter + 1'b1;
                end
            end
        end
    end

endmodule
