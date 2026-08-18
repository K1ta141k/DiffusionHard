`timescale 1ns/1ps

module tb_ordered_noisy_argmax_reducer;
    logic clk = 0;
    logic rst_n = 0;
    logic start_valid = 0;
    logic start_ready;
    logic [6:0] position_count = 0;
    logic [15:0] mask_token_id = 16'hffff;
    logic in_valid = 0;
    logic in_ready;
    logic in_last = 0;
    logic [1:0] in_valid_mask = 0;
    logic [5:0] in_position = 0;
    logic [15:0] in_token_base = 0;
    logic signed [32:0] in_model_score0_q10 = 0;
    logic signed [32:0] in_model_score1_q10 = 0;
    logic signed [15:0] in_noise0_q10 = 0;
    logic signed [15:0] in_noise1_q10 = 0;
    logic out_valid;
    logic out_ready = 0;
    logic [5:0] out_position;
    logic out_candidate_valid;
    logic [15:0] out_candidate_id;
    logic signed [33:0] out_candidate_score_q10;
    logic busy;
    logic done;
    logic [1:0] status;
    integer accepted;

    ordered_noisy_argmax_reducer dut (.*);
    always #5 clk = ~clk;

    task automatic send_pair(
        input [5:0] pair_position,
        input signed [32:0] model0,
        input signed [32:0] model1,
        input signed [15:0] noise0,
        input signed [15:0] noise1,
        input last
    );
        begin
            in_position = pair_position;
            in_token_base = 0;
            in_valid_mask = 2'b11;
            in_model_score0_q10 = model0;
            in_model_score1_q10 = model1;
            in_noise0_q10 = noise0;
            in_noise1_q10 = noise1;
            in_last = last;
            in_valid = 1;
            @(posedge clk);
            while (!in_ready) @(posedge clk);
            @(negedge clk);
            in_valid = 0;
            in_last = 0;
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        position_count = 2;
        start_valid = 1;
        @(negedge clk);
        start_valid = 0;

        send_pair(0, 0, 0, 10, 10, 0);
        // This pair both crosses the position boundary and ends the stream.
        send_pair(1, 0, 0, 20, 30, 1);

        while (!out_valid) @(negedge clk);
        out_ready = 1;
        accepted = 0;
        while (accepted < 2) begin
            if (out_valid) begin
                if (!out_candidate_valid || out_position !== accepted) begin
                    $fatal(1, "ordered reducer candidate metadata mismatch");
                end
                if (
                    (accepted == 0 &&
                     (out_candidate_id !== 0 || out_candidate_score_q10 !== 10)) ||
                    (accepted == 1 &&
                     (out_candidate_id !== 1 || out_candidate_score_q10 !== 30))
                ) begin
                    $fatal(1, "ordered reducer score or tie mismatch");
                end
                accepted = accepted + 1;
            end
            @(posedge clk);
            @(negedge clk);
        end
        if (!done || busy || status != 0) begin
            $fatal(1, "ordered reducer did not finish cleanly");
        end

        @(negedge clk);
        position_count = 2;
        start_valid = 1;
        @(negedge clk);
        start_valid = 0;
        send_pair(1, 0, 0, 0, 0, 0);
        if (!done || busy || status != 2) begin
            $fatal(1, "out-of-order stream was not rejected");
        end

        $display("tb_ordered_noisy_argmax_reducer: all checks passed");
        $finish;
    end

endmodule
