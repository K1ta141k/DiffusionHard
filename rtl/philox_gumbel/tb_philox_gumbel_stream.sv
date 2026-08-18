`timescale 1ns/1ps

module tb_philox_gumbel_stream;
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
    logic out_valid;
    logic out_ready = 0;
    logic [5:0] out_position;
    logic [15:0] out_token;
    logic [31:0] out_random_word;
    logic signed [15:0] out_gumbel_q10;
    logic busy;
    logic done;
    logic [1:0] status;

    logic [31:0] expected_words [0:3];
    logic signed [15:0] expected_scores [0:3];
    logic [31:0] replay_words [0:37];
    logic signed [15:0] replay_scores [0:37];
    integer index;
    integer accepted;
    integer cycle_count;
    logic was_stalled;
    logic [31:0] stalled_word;
    logic signed [15:0] stalled_score;

    philox_gumbel_stream dut (
        .clk,
        .rst_n,
        .start_valid,
        .start_ready,
        .position_count,
        .vocabulary_size,
        .evaluation_id,
        .stream_id,
        .seed_low,
        .seed_high,
        .out_valid,
        .out_ready,
        .out_position,
        .out_token,
        .out_random_word,
        .out_gumbel_q10,
        .busy,
        .done,
        .status
    );

    always #5 clk = ~clk;

    task automatic start_command(
        input [6:0] positions,
        input [15:0] vocabulary,
        input [31:0] evaluation,
        input [31:0] stream,
        input [31:0] low,
        input [31:0] high
    );
        begin
            @(negedge clk);
            position_count = positions;
            vocabulary_size = vocabulary;
            evaluation_id = evaluation;
            stream_id = stream;
            seed_low = low;
            seed_high = high;
            start_valid = 1;
            @(negedge clk);
            start_valid = 0;
        end
    endtask

    initial begin
        expected_words[0] = 32'h6627e8d5;
        expected_words[1] = 32'he169c58d;
        expected_words[2] = 32'hbc57ac4c;
        expected_words[3] = 32'h9b00dbd8;
        expected_scores[0] = 16'sd88;
        expected_scores[1] = 16'sd2176;
        expected_scores[2] = 16'sd1210;
        expected_scores[3] = 16'sd708;

        repeat (3) @(negedge clk);
        rst_n = 1;

        start_command(1, 4, 0, 0, 0, 0);
        out_ready = 1;
        accepted = 0;
        while (accepted < 4) begin
            if (out_valid && out_ready) begin
                if (out_position !== 0 || out_token !== accepted) begin
                    $fatal(
                        1,
                        "official vector position/token mismatch: position=%0d token=%0d expected=%0d",
                        out_position,
                        out_token,
                        accepted
                    );
                end
                if (out_random_word !== expected_words[accepted]) begin
                    $fatal(1, "official Philox word mismatch at %0d", accepted);
                end
                if (out_gumbel_q10 !== expected_scores[accepted]) begin
                    $fatal(1, "Gumbel score mismatch at %0d", accepted);
                end
                accepted = accepted + 1;
            end
            @(posedge clk);
            @(negedge clk);
        end
        if (!done || status != 0) begin
            $fatal(1, "official vector command did not finish cleanly");
        end

        out_ready = 0;
        start_command(2, 19, 3, 5, 7, 11);
        accepted = 0;
        cycle_count = 0;
        was_stalled = 0;
        while (accepted < 38) begin
            if (was_stalled) begin
                if (
                    out_random_word !== stalled_word ||
                    out_gumbel_q10 !== stalled_score
                ) begin
                    $fatal(1, "output changed under backpressure");
                end
            end
            out_ready = (cycle_count % 5 != 2);
            was_stalled = out_valid && !out_ready;
            if (was_stalled) begin
                stalled_word = out_random_word;
                stalled_score = out_gumbel_q10;
            end
            if (out_valid && out_ready) begin
                replay_words[accepted] = out_random_word;
                replay_scores[accepted] = out_gumbel_q10;
                accepted = accepted + 1;
            end
            @(posedge clk);
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end
        out_ready = 1;
        if (!done) begin
            $fatal(1, "backpressured command did not finish");
        end

        out_ready = 0;
        start_command(2, 19, 3, 5, 7, 11);
        accepted = 0;
        out_ready = 1;
        while (accepted < 38) begin
            if (out_valid && out_ready) begin
                if (
                    out_random_word !== replay_words[accepted] ||
                    out_gumbel_q10 !== replay_scores[accepted]
                ) begin
                    $fatal(1, "deterministic replay mismatch at %0d", accepted);
                end
                accepted = accepted + 1;
            end
            @(posedge clk);
            @(negedge clk);
        end
        if (!done) begin
            $fatal(1, "replay command did not finish");
        end

        out_ready = 0;
        start_command(0, 4, 0, 0, 0, 0);
        if (!done || status != 1) begin
            $fatal(1, "invalid position count was not rejected");
        end
        start_command(1, 0, 0, 0, 0, 0);
        if (!done || status != 2) begin
            $fatal(1, "invalid vocabulary size was not rejected");
        end

        $display("tb_philox_gumbel_stream: all checks passed");
        $finish;
    end

endmodule
