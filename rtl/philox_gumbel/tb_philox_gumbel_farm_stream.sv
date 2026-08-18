`timescale 1ns/1ps

module tb_philox_gumbel_farm_stream;
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
    logic score_valid;
    logic score_ready = 1;
    logic [1:0] score_valid_mask;
    logic [5:0] score_position;
    logic [15:0] score_token_base;
    logic signed [15:0] score0_q10;
    logic signed [15:0] score1_q10;
    logic busy;
    logic done;
    logic [1:0] status;
    logic signed [15:0] expected [0:9];
    integer received;
    integer cycle_count;
    logic was_stalled;
    logic [1:0] stalled_mask;
    logic [5:0] stalled_position;
    logic [15:0] stalled_token_base;
    logic signed [15:0] stalled_score0;
    logic signed [15:0] stalled_score1;

    philox_gumbel_farm_stream dut (.*);
    always #5 clk = ~clk;

    initial begin
        expected[0] = 380;
        expected[1] = 192;
        expected[2] = 2223;
        expected[3] = 1522;
        expected[4] = -432;
        expected[5] = -103;
        expected[6] = 1365;
        expected[7] = 1452;
        expected[8] = 574;
        expected[9] = 2191;

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

        received = 0;
        cycle_count = 0;
        was_stalled = 0;
        while (received < 10) begin
            if (was_stalled) begin
                if (
                    !score_valid || score_valid_mask !== stalled_mask ||
                    score_position !== stalled_position ||
                    score_token_base !== stalled_token_base ||
                    score0_q10 !== stalled_score0 ||
                    score1_q10 !== stalled_score1
                ) begin
                    $fatal(1, "integrated output changed under backpressure");
                end
            end
            score_ready = cycle_count % 5 != 2;
            was_stalled = score_valid && !score_ready;
            if (was_stalled) begin
                stalled_mask = score_valid_mask;
                stalled_position = score_position;
                stalled_token_base = score_token_base;
                stalled_score0 = score0_q10;
                stalled_score1 = score1_q10;
            end
            if (score_valid && score_ready) begin
                if (score_position !== received / 5) begin
                    $fatal(1, "integrated position mismatch at %0d", received);
                end
                if (score_token_base !== received % 5) begin
                    $fatal(1, "integrated token mismatch at %0d", received);
                end
                if (score_valid_mask[0]) begin
                    if (score0_q10 !== expected[received]) begin
                        $fatal(1, "integrated score 0 mismatch at %0d", received);
                    end
                    received = received + 1;
                end
                if (score_valid_mask[1]) begin
                    if (score1_q10 !== expected[received]) begin
                        $fatal(1, "integrated score 1 mismatch at %0d", received);
                    end
                    received = received + 1;
                end
            end
            @(posedge clk);
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end
        score_ready = 1;
        while (!done) begin
            @(posedge clk);
            @(negedge clk);
        end
        if (status != 0 || busy) begin
            $fatal(1, "integrated stream did not finish cleanly");
        end

        @(negedge clk);
        position_count = 3;
        vocabulary_size = 37;
        start_valid = 1;
        @(negedge clk);
        start_valid = 0;
        received = 0;
        cycle_count = 0;
        while (received < 111) begin
            score_ready = cycle_count % 7 != 3;
            if (score_valid && score_ready) begin
                if (score_position !== received / 37) begin
                    $fatal(1, "sustained position mismatch at %0d", received);
                end
                if (score_token_base !== received % 37) begin
                    $fatal(1, "sustained token mismatch at %0d", received);
                end
                if (score_valid_mask[0]) begin
                    received = received + 1;
                end
                if (score_valid_mask[1]) begin
                    received = received + 1;
                end
            end
            @(posedge clk);
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end
        score_ready = 1;
        while (!done) begin
            @(posedge clk);
            @(negedge clk);
        end
        if (status != 0 || busy) begin
            $fatal(1, "sustained stream did not finish cleanly");
        end

        @(negedge clk);
        position_count = 1;
        vocabulary_size = 16'hffff;
        start_valid = 1;
        @(negedge clk);
        start_valid = 0;
        score_ready = 1;
        received = 0;
        while (received < 65535) begin
            if (score_valid) begin
                if (score_position !== 0) begin
                    $fatal(1, "maximum-vocabulary position mismatch");
                end
                if (score_token_base !== received[15:0]) begin
                    $fatal(
                        1,
                        "maximum-vocabulary token mismatch at %0d",
                        received
                    );
                end
                if (score_valid_mask[0]) begin
                    received = received + 1;
                end
                if (score_valid_mask[1]) begin
                    received = received + 1;
                end
            end
            @(posedge clk);
            @(negedge clk);
        end
        while (!done) begin
            @(posedge clk);
            @(negedge clk);
        end
        if (status != 0 || busy) begin
            $fatal(1, "maximum-vocabulary stream did not terminate");
        end

        $display("tb_philox_gumbel_farm_stream: all checks passed");
        $finish;
    end

endmodule
