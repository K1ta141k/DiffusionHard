`timescale 1ns/1ps

module gumbel_q10_lane (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               input_valid,
    output logic               input_ready,
    input  logic [31:0]        input_word,
    output logic               output_valid,
    input  logic               output_ready,
    output logic signed [15:0] output_score_q10
);

    localparam integer LOG_TWO_Q10 = 710;
    localparam logic signed [15:0] TOP_WORD_Q10 = 16'sd23423;

    logic signed [15:0] gumbel_lut [0:1023];
    initial begin
        $readmemh("rtl/philox_gumbel/gumbel_lut_q10.mem", gumbel_lut);
    end

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

    logic [31:0] distance_from_one;
    logic [5:0] exponent;
    logic [5:0] floor_log2;
    logic [31:0] normalized_distance;
    logic [7:0] mantissa;
    logic [9:0] lut_address;
    logic special_top_word;
    logic stage1_valid;
    logic [5:0] stage1_exponent;
    logic [9:0] stage1_lut_address;
    logic stage1_special_top_word;
    logic signed [31:0] score_wide;
    logic output_stage_ready;

    assign output_stage_ready = !output_valid || output_ready;
    assign input_ready = !stage1_valid || output_stage_ready;

    always_comb begin
        distance_from_one = 32'hffff_ffff - input_word;
        exponent = leading_zeros32(distance_from_one);
        floor_log2 = 6'd31 - exponent;
        special_top_word = distance_from_one == 0;
        if (special_top_word) begin
            normalized_distance = 0;
            mantissa = 0;
            lut_address = 0;
        end else begin
            if (floor_log2 >= 8) begin
                normalized_distance = distance_from_one >> (floor_log2 - 8);
            end else begin
                normalized_distance = distance_from_one << (8 - floor_log2);
            end
            mantissa = normalized_distance[7:0];
            if (exponent < 3) begin
                lut_address = 256 + exponent * 256 + mantissa;
            end else begin
                lut_address = mantissa;
            end
        end
    end

    always_comb begin
        if (stage1_special_top_word) begin
            score_wide = TOP_WORD_Q10;
        end else if (stage1_exponent < 3) begin
            score_wide = $signed(gumbel_lut[stage1_lut_address]);
        end else begin
            score_wide = $signed(gumbel_lut[stage1_lut_address]) +
                (stage1_exponent + 1) * LOG_TWO_Q10;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            stage1_valid <= 1'b0;
            stage1_exponent <= 0;
            stage1_lut_address <= 0;
            stage1_special_top_word <= 1'b0;
            output_valid <= 1'b0;
            output_score_q10 <= 0;
        end else begin
            if (output_stage_ready) begin
                output_valid <= stage1_valid;
                if (stage1_valid) begin
                    output_score_q10 <= score_wide[15:0];
                end
            end
            if (input_ready) begin
                stage1_valid <= input_valid;
                if (input_valid) begin
                    stage1_exponent <= exponent;
                    stage1_lut_address <= lut_address;
                    stage1_special_top_word <= special_top_word;
                end
            end
        end
    end

endmodule


module gumbel_q10_dual (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               input_valid,
    output logic               input_ready,
    input  logic [31:0]        input_word0,
    input  logic [31:0]        input_word1,
    output logic               output_valid,
    input  logic               output_ready,
    output logic signed [15:0] output_score0_q10,
    output logic signed [15:0] output_score1_q10
);

    logic output_valid0;
    logic output_valid1;
    logic input_ready0;
    logic input_ready1;

    gumbel_q10_lane lane0 (
        .clk,
        .rst_n,
        .input_valid,
        .input_ready(input_ready0),
        .input_word(input_word0),
        .output_valid(output_valid0),
        .output_ready,
        .output_score_q10(output_score0_q10)
    );
    gumbel_q10_lane lane1 (
        .clk,
        .rst_n,
        .input_valid,
        .input_ready(input_ready1),
        .input_word(input_word1),
        .output_valid(output_valid1),
        .output_ready,
        .output_score_q10(output_score1_q10)
    );

    assign output_valid = output_valid0 && output_valid1;
    assign input_ready = input_ready0 && input_ready1;

endmodule
