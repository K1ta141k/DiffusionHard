`timescale 1ns/1ps

module tb_noisy_argmax_reducer;
    logic clk = 0;
    logic rst_n = 0;
    logic start_valid = 0;
    logic start_ready;
    logic [6:0] position_count = 0;
    logic [15:0] mask_token_id = 0;
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
    integer output_count;

    noisy_argmax_reducer dut (.*);
    always #5 clk = ~clk;

    task automatic send_pair(
        input [5:0] pair_position,
        input [15:0] token_base,
        input [1:0] valid_mask,
        input signed [32:0] model0,
        input signed [32:0] model1,
        input signed [15:0] noise0,
        input signed [15:0] noise1,
        input last
    );
        begin
            while (!in_ready) @(negedge clk);
            in_position = pair_position;
            in_token_base = token_base;
            in_valid_mask = valid_mask;
            in_model_score0_q10 = model0;
            in_model_score1_q10 = model1;
            in_noise0_q10 = noise0;
            in_noise1_q10 = noise1;
            in_last = last;
            in_valid = 1;
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
        mask_token_id = 4;
        start_valid = 1;
        @(negedge clk);
        start_valid = 0;

        // Position 1 arrives first to verify metadata-addressed reduction.
        send_pair(1, 0, 2'b11, 0, 0, 10, 10, 0);
        send_pair(0, 0, 2'b11, 0, 0, 100, 100, 0);
        // Equal scores select the lower token even when the higher token was seen first.
        send_pair(0, 2, 2'b11, 0, 0, 100, 100, 0);
        // Token 4 is masked and the second lane is invalid.
        send_pair(0, 4, 2'b01, 1000, 0, 1000, 0, 0);
        send_pair(1, 2, 2'b11, 0, 0, 20, 30, 0);
        send_pair(1, 4, 2'b01, 1000, 0, 1000, 0, 1);

        out_ready = 1;
        output_count = 0;
        while (output_count < 2) begin
            if (out_valid) begin
                if (!out_candidate_valid) begin
                    $fatal(1, "candidate unexpectedly invalid");
                end
                if (out_position == 0) begin
                    if (out_candidate_id !== 0 || out_candidate_score_q10 !== 100) begin
                        $fatal(1, "position 0 tie or mask handling failed");
                    end
                end else begin
                    if (out_candidate_id !== 3 || out_candidate_score_q10 !== 30) begin
                        $fatal(1, "position 1 reduction failed");
                    end
                end
                output_count = output_count + 1;
            end
            @(posedge clk);
            @(negedge clk);
        end
        if (!done || busy || status != 0) begin
            $fatal(1, "reducer did not finish cleanly");
        end

        $display("tb_noisy_argmax_reducer: all checks passed");
        $finish;
    end

endmodule
