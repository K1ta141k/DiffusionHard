`timescale 1ns/1ps

module tb_philox_accumulator_argmax_stream;
    logic clk = 0;
    logic rst_n = 0;
    logic start_valid = 0;
    logic start_ready;
    logic [6:0] position_count = 0;
    logic [15:0] vocabulary_size = 0;
    logic [15:0] mask_token_id = 0;
    logic [31:0] evaluation_id = 0;
    logic [31:0] stream_id = 0;
    logic [31:0] seed_low = 0;
    logic [31:0] seed_high = 0;
    logic accumulator_valid = 0;
    logic accumulator_ready;
    logic signed [31:0] accumulator0 = 0;
    logic signed [31:0] accumulator1 = 0;
    logic signed [31:0] multiplier0_q20 = 0;
    logic signed [31:0] multiplier1_q20 = 0;
    logic signed [31:0] bias0_q10 = 0;
    logic signed [31:0] bias1_q10 = 0;
    logic candidate_valid;
    logic candidate_ready = 0;
    logic [5:0] candidate_position;
    logic candidate_id_valid;
    logic [15:0] candidate_id;
    logic signed [33:0] candidate_score_q10;
    logic busy;
    logic done;
    logic [2:0] status;

    philox_accumulator_argmax_stream dut (.*);
    always #5 clk = ~clk;

    task automatic send_accumulator_pair(
        input signed [31:0] value0,
        input signed [31:0] value1
    );
        begin
            accumulator0 = value0;
            accumulator1 = value1;
            multiplier0_q20 = 32'sd1048576;
            multiplier1_q20 = 32'sd1048576;
            bias0_q10 = 0;
            bias1_q10 = 0;
            accumulator_valid = 1;
            @(posedge clk);
            while (!accumulator_ready) @(posedge clk);
            @(negedge clk);
            accumulator_valid = 0;
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        position_count = 1;
        vocabulary_size = 3;
        mask_token_id = 2;
        start_valid = 1;
        @(negedge clk);
        start_valid = 0;

        send_accumulator_pair(100, 0);
        send_accumulator_pair(500, 0);

        while (!candidate_valid) @(negedge clk);
        if (
            !candidate_id_valid || candidate_position !== 0 ||
            candidate_id !== 1 || candidate_score_q10 !== 2176
        ) begin
            $fatal(1, "accumulator-to-candidate result mismatch");
        end
        repeat (2) begin
            @(posedge clk);
            @(negedge clk);
            if (
                !candidate_valid || candidate_id !== 1 ||
                candidate_score_q10 !== 2176
            ) begin
                $fatal(1, "accumulator candidate changed under backpressure");
            end
        end
        candidate_ready = 1;
        @(posedge clk);
        @(negedge clk);
        while (!done) begin
            @(posedge clk);
            @(negedge clk);
        end
        if (busy || status != 0) begin
            $fatal(1, "accumulator argmax stream did not finish cleanly");
        end

        $display("tb_philox_accumulator_argmax_stream: all checks passed");
        $finish;
    end

endmodule
