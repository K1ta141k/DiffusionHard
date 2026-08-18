`timescale 1ns/1ps

module tb_philox_noisy_argmax_stream;
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
    logic model_valid = 0;
    logic model_ready;
    logic signed [32:0] model_score0_q10 = 0;
    logic signed [32:0] model_score1_q10 = 0;
    logic candidate_valid;
    logic candidate_ready = 0;
    logic [5:0] candidate_position;
    logic candidate_id_valid;
    logic [15:0] candidate_id;
    logic signed [33:0] candidate_score_q10;
    logic busy;
    logic done;
    logic [2:0] status;
    integer accepted;
    logic [5:0] stalled_position;
    logic [15:0] stalled_id;
    logic signed [33:0] stalled_score;
    integer watchdog = 0;
    integer pairs_sent = 0;

    philox_noisy_argmax_stream dut (.*);
    always #5 clk = ~clk;
    always @(posedge clk) begin
        if (rst_n) begin
            watchdog = watchdog + 1;
            if (watchdog == 500) begin
                $fatal(
                    1,
                    "watchdog: pairs=%0d model_valid=%0d model_ready=%0d rng_valid=%0d rng_busy=%0d reducer_ready=%0d reducer_busy=%0d accepting=%0d current=%0d final_pending=%0d out_valid=%0d accepted=%0d wrapper_busy=%0d wrapper_done=%0d status=%0d",
                    pairs_sent,
                    model_valid,
                    model_ready,
                    dut.rng_score_valid,
                    dut.rng_busy,
                    dut.reducer_in_ready,
                    dut.reducer_busy,
                    dut.reducer.accepting,
                    dut.reducer.current_position,
                    dut.reducer.final_pending,
                    candidate_valid,
                    accepted,
                    busy,
                    done,
                    status
                );
            end
        end
    end

    task automatic send_model_pair(
        input signed [32:0] score0,
        input signed [32:0] score1
    );
        begin
            model_score0_q10 = score0;
            model_score1_q10 = score1;
            model_valid = 1;
            @(posedge clk);
            while (!model_ready) @(posedge clk);
            @(negedge clk);
            pairs_sent = pairs_sent + 1;
            model_valid = 0;
            // Add a deterministic producer bubble.
            @(posedge clk);
            @(negedge clk);
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        position_count = 2;
        vocabulary_size = 5;
        mask_token_id = 4;
        evaluation_id = 3;
        stream_id = 5;
        seed_low = 7;
        seed_high = 11;
        start_valid = 1;
        @(negedge clk);
        start_valid = 0;

        accepted = 0;
        fork
            begin
                send_model_pair(10, -100);
                send_model_pair(0, 100);
                send_model_pair(500, 0);
                send_model_pair(-100, 0);
                send_model_pair(0, 0);
                send_model_pair(0, 0);
            end
            begin
                while (!candidate_valid) @(negedge clk);
                stalled_position = candidate_position;
                stalled_id = candidate_id;
                stalled_score = candidate_score_q10;
                repeat (3) begin
                    @(posedge clk);
                    @(negedge clk);
                    if (
                        !candidate_valid ||
                        candidate_position !== stalled_position ||
                        candidate_id !== stalled_id ||
                        candidate_score_q10 !== stalled_score
                    ) begin
                        $fatal(1, "candidate changed under backpressure");
                    end
                end

                candidate_ready = 1;
                while (accepted < 2) begin
                    if (candidate_valid) begin
                        if (!candidate_id_valid) begin
                            $fatal(
                                1,
                                "integrated candidate unexpectedly invalid"
                            );
                        end
                        if (candidate_position !== accepted) begin
                            $fatal(1, "integrated candidate position mismatch");
                        end
                        if (candidate_id !== 2) begin
                            $fatal(1, "integrated candidate ID mismatch");
                        end
                        if (
                            (accepted == 0 && candidate_score_q10 !== 2223) ||
                            (accepted == 1 && candidate_score_q10 !== 1452)
                        ) begin
                            $fatal(1, "integrated candidate score mismatch");
                        end
                        accepted = accepted + 1;
                    end
                    @(posedge clk);
                    @(negedge clk);
                end
            end
        join
        while (!done) begin
            @(posedge clk);
            @(negedge clk);
        end
        if (busy || status != 0) begin
            $fatal(1, "integrated argmax stream did not finish cleanly");
        end

        $display("tb_philox_noisy_argmax_stream: all checks passed");
        $finish;
    end

endmodule
