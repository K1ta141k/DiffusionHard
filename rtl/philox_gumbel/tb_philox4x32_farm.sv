`timescale 1ns/1ps

module tb_philox4x32_farm;
    logic clk = 0;
    logic rst_n = 0;
    logic start_valid = 0;
    logic start_ready;
    logic [6:0] position_count = 0;
    logic [15:0] vocabulary_size = 0;
    logic [31:0] evaluation_id = 0;
    logic [31:0] stream_id = 0;
    logic [31:0] seed_low = 0;
    logic [31:0] seed_high = 0;
    logic block_valid;
    logic block_ready = 0;
    logic [5:0] block_position;
    logic [15:0] block_token_base;
    logic [2:0] block_valid_words;
    logic [127:0] block_random_words;
    logic busy;
    logic done;
    logic [1:0] status;
    logic [127:0] expected [0:3];
    integer accepted;

    philox4x32_farm dut (.*);
    always #5 clk = ~clk;

    initial begin
        expected[0] = 128'hcc33d2d6_e2c538d3_6fce74b6_807fb1d7;
        expected[1] = 128'h41f9154d_2d97550d_66187336_3787943f;
        expected[2] = 128'h90a09e45_c8f4fa00_c4b910c8_5483f8aa;
        expected[3] = 128'h3baa105b_60bd4c52_ee980d56_e1dc45d1;

        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        position_count = 2;
        vocabulary_size = 5;
        evaluation_id = 3;
        stream_id = 5;
        seed_low = 7;
        seed_high = 11;
        start_valid = 1;
        @(negedge clk);
        start_valid = 0;

        block_ready = 1;
        accepted = 0;
        while (accepted < 4) begin
            if (block_valid) begin
                if (block_position !== accepted[1]) begin
                    $fatal(1, "farm position mismatch at %0d", accepted);
                end
                if (block_token_base !== ((accepted & 1) * 4)) begin
                    $fatal(1, "farm token base mismatch at %0d", accepted);
                end
                if (block_valid_words !== ((accepted & 1) ? 1 : 4)) begin
                    $fatal(1, "farm valid-word count mismatch at %0d", accepted);
                end
                if (block_random_words !== expected[accepted]) begin
                    $fatal(1, "farm random block mismatch at %0d", accepted);
                end
                accepted = accepted + 1;
            end
            @(posedge clk);
            @(negedge clk);
        end
        if (!done || status != 0) begin
            $fatal(1, "farm did not finish cleanly");
        end

        $display("tb_philox4x32_farm: all checks passed");
        $finish;
    end

endmodule
