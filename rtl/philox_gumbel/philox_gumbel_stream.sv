`timescale 1ns/1ps

module philox_gumbel_stream #(
    parameter integer MAX_POSITIONS = 64,
    parameter integer MAX_VOCABULARY = 65535
) (
    input  logic               clk,
    input  logic               rst_n,

    input  logic               start_valid,
    output logic               start_ready,
    input  logic [6:0]         position_count,
    input  logic [15:0]        vocabulary_size,
    input  logic [31:0]        evaluation_id,
    input  logic [31:0]        stream_id,
    input  logic [31:0]        seed_low,
    input  logic [31:0]        seed_high,

    output logic               out_valid,
    input  logic               out_ready,
    output logic [5:0]         out_position,
    output logic [15:0]        out_token,
    output logic [31:0]        out_random_word,
    output logic signed [15:0] out_gumbel_q10,

    output logic               busy,
    output logic               done,
    output logic [1:0]         status
);

    localparam logic [1:0] STATUS_OK = 2'd0;
    localparam logic [1:0] STATUS_INVALID_POSITION_COUNT = 2'd1;
    localparam logic [1:0] STATUS_INVALID_VOCABULARY_SIZE = 2'd2;
    localparam logic [31:0] PHILOX_M0 = 32'hD2511F53;
    localparam logic [31:0] PHILOX_M1 = 32'hCD9E8D57;
    localparam logic [31:0] PHILOX_W0 = 32'h9E3779B9;
    localparam logic [31:0] PHILOX_W1 = 32'hBB67AE85;
    localparam integer GUMBEL_LOG_TWO_Q10 = 710;
    localparam logic signed [15:0] GUMBEL_TOP_WORD_Q10 = 16'sd23423;

    logic signed [15:0] gumbel_lut [0:1023];
    initial begin
        $readmemh("rtl/philox_gumbel/gumbel_lut_q10.mem", gumbel_lut);
    end

    function automatic [127:0] philox4x32_10;
        input logic [31:0] input_c0;
        input logic [31:0] input_c1;
        input logic [31:0] input_c2;
        input logic [31:0] input_c3;
        input logic [31:0] input_k0;
        input logic [31:0] input_k1;
        logic [31:0] c0;
        logic [31:0] c1;
        logic [31:0] c2;
        logic [31:0] c3;
        logic [31:0] k0;
        logic [31:0] k1;
        logic [63:0] product0;
        logic [63:0] product1;
        integer round_index;
        begin
            c0 = input_c0;
            c1 = input_c1;
            c2 = input_c2;
            c3 = input_c3;
            k0 = input_k0;
            k1 = input_k1;
            for (round_index = 0; round_index < 10; round_index = round_index + 1) begin
                product0 = PHILOX_M0 * c0;
                product1 = PHILOX_M1 * c2;
                c0 = product1[63:32] ^ c1 ^ k0;
                c1 = product1[31:0];
                c2 = product0[63:32] ^ c3 ^ k1;
                c3 = product0[31:0];
                if (round_index != 9) begin
                    k0 = k0 + PHILOX_W0;
                    k1 = k1 + PHILOX_W1;
                end
            end
            philox4x32_10 = {c3, c2, c1, c0};
        end
    endfunction

    function automatic [5:0] leading_zeros32;
        input logic [31:0] value;
        integer bit_index;
        logic [5:0] count;
        begin
            count = 6'd32;
            for (bit_index = 0; bit_index < 32; bit_index = bit_index + 1) begin
                if (value[bit_index]) begin
                    count = 31 - bit_index;
                end
            end
            leading_zeros32 = count;
        end
    endfunction

    logic [6:0] position_count_reg;
    logic [15:0] vocabulary_size_reg;
    logic [31:0] evaluation_id_reg;
    logic [31:0] stream_id_reg;
    logic [31:0] seed_low_reg;
    logic [31:0] seed_high_reg;
    logic [15:0] vocabulary_base_reg;
    logic [5:0] position_reg;
    logic [1:0] lane_block_reg;
    logic [1:0] word_index_reg;
    logic [31:0] random_words [0:3];

    logic [15:0] token_base;
    logic [2:0] word_count;
    logic has_next_block;
    logic [15:0] next_vocabulary_base;
    logic [5:0] next_position;
    logic [1:0] next_lane_block;
    logic [15:0] next_token_base;
    logic [127:0] next_random_block;
    logic [16:0] next_lane_token_base_wide;
    logic [16:0] next_vocabulary_base_wide;

    logic [31:0] distance_from_one;
    logic [5:0] gumbel_exponent;
    logic [5:0] floor_log2;
    logic [31:0] normalized_distance;
    logic [7:0] gumbel_mantissa;
    integer gumbel_lut_address_value;
    logic signed [31:0] gumbel_score_wide;

    assign start_ready = !busy;
    assign out_valid = busy;
    assign out_position = position_reg;
    assign token_base = vocabulary_base_reg + {lane_block_reg, 2'b00};
    assign out_token = token_base + word_index_reg;
    assign word_count =
        (vocabulary_size_reg - token_base >= 4)
            ? 3'd4
            : vocabulary_size_reg - token_base;

    always_comb begin
        case (word_index_reg)
            2'd0: out_random_word = random_words[0];
            2'd1: out_random_word = random_words[1];
            2'd2: out_random_word = random_words[2];
            default: out_random_word = random_words[3];
        endcase
    end

    always_comb begin
        distance_from_one = 32'hffff_ffff - out_random_word;
        gumbel_exponent = leading_zeros32(distance_from_one);
        floor_log2 = 6'd31 - gumbel_exponent;
        if (distance_from_one == 0) begin
            normalized_distance = 0;
            gumbel_mantissa = 0;
            gumbel_lut_address_value = 0;
            gumbel_score_wide = GUMBEL_TOP_WORD_Q10;
        end else begin
            if (floor_log2 >= 8) begin
                normalized_distance = distance_from_one >> (floor_log2 - 8);
            end else begin
                normalized_distance = distance_from_one << (8 - floor_log2);
            end
            gumbel_mantissa = normalized_distance[7:0];
            if (gumbel_exponent < 3) begin
                gumbel_lut_address_value =
                    256 + gumbel_exponent * 256 + gumbel_mantissa;
                gumbel_score_wide = gumbel_lut[gumbel_lut_address_value];
            end else begin
                gumbel_lut_address_value = gumbel_mantissa;
                gumbel_score_wide =
                    $signed(gumbel_lut[gumbel_lut_address_value]) +
                    (gumbel_exponent + 1) * GUMBEL_LOG_TWO_Q10;
            end
        end
    end
    assign out_gumbel_q10 = gumbel_score_wide[15:0];

    always_comb begin
        has_next_block = 1'b1;
        next_vocabulary_base = vocabulary_base_reg;
        next_position = position_reg;
        next_lane_block = lane_block_reg;
        next_lane_token_base_wide = {1'b0, vocabulary_base_reg} +
            ({15'd0, lane_block_reg} + 1'b1) * 17'd4;
        next_vocabulary_base_wide = {1'b0, vocabulary_base_reg} + 17'd16;
        if (
            lane_block_reg < 3 &&
            next_lane_token_base_wide < {1'b0, vocabulary_size_reg}
        ) begin
            next_lane_block = lane_block_reg + 1'b1;
        end else if (position_reg + 1'b1 < position_count_reg) begin
            next_position = position_reg + 1'b1;
            next_lane_block = 0;
        end else if (
            next_vocabulary_base_wide < {1'b0, vocabulary_size_reg}
        ) begin
            next_vocabulary_base = vocabulary_base_reg + 16;
            next_position = 0;
            next_lane_block = 0;
        end else begin
            has_next_block = 1'b0;
        end
        next_token_base =
            next_vocabulary_base + {next_lane_block, 2'b00};
        next_random_block = philox4x32_10(
            {18'd0, next_token_base[15:2]},
            {26'd0, next_position},
            evaluation_id_reg,
            stream_id_reg,
            seed_low_reg,
            seed_high_reg
        );
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            status <= STATUS_OK;
            position_count_reg <= 0;
            vocabulary_size_reg <= 0;
            evaluation_id_reg <= 0;
            stream_id_reg <= 0;
            seed_low_reg <= 0;
            seed_high_reg <= 0;
            vocabulary_base_reg <= 0;
            position_reg <= 0;
            lane_block_reg <= 0;
            word_index_reg <= 0;
            random_words[0] <= 0;
            random_words[1] <= 0;
            random_words[2] <= 0;
            random_words[3] <= 0;
        end else begin
            done <= 1'b0;
            if (!busy) begin
                if (start_valid && start_ready) begin
                    status <= STATUS_OK;
                    if (
                        position_count == 0 ||
                        position_count > MAX_POSITIONS
                    ) begin
                        done <= 1'b1;
                        status <= STATUS_INVALID_POSITION_COUNT;
                    end else if (
                        vocabulary_size == 0 ||
                        vocabulary_size > MAX_VOCABULARY
                    ) begin
                        done <= 1'b1;
                        status <= STATUS_INVALID_VOCABULARY_SIZE;
                    end else begin
                        busy <= 1'b1;
                        position_count_reg <= position_count;
                        vocabulary_size_reg <= vocabulary_size;
                        evaluation_id_reg <= evaluation_id;
                        stream_id_reg <= stream_id;
                        seed_low_reg <= seed_low;
                        seed_high_reg <= seed_high;
                        vocabulary_base_reg <= 0;
                        position_reg <= 0;
                        lane_block_reg <= 0;
                        word_index_reg <= 0;
                        {random_words[3], random_words[2],
                         random_words[1], random_words[0]} <= philox4x32_10(
                            0,
                            0,
                            evaluation_id,
                            stream_id,
                            seed_low,
                            seed_high
                        );
                    end
                end
            end else if (out_valid && out_ready) begin
                if (word_index_reg + 1'b1 < word_count) begin
                    word_index_reg <= word_index_reg + 1'b1;
                end else if (has_next_block) begin
                    vocabulary_base_reg <= next_vocabulary_base;
                    position_reg <= next_position;
                    lane_block_reg <= next_lane_block;
                    word_index_reg <= 0;
                    {random_words[3], random_words[2],
                     random_words[1], random_words[0]} <= next_random_block;
                end else begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end

endmodule
