`timescale 1ns/1ps

module tb_dual_requantizer_q20;
    logic clk = 0;
    logic rst_n = 0;
    logic input_valid = 0;
    logic input_ready;
    logic signed [31:0] accumulator0 = 0;
    logic signed [31:0] accumulator1 = 0;
    logic signed [31:0] multiplier0_q20 = 0;
    logic signed [31:0] multiplier1_q20 = 0;
    logic signed [31:0] bias0_q10 = 0;
    logic signed [31:0] bias1_q10 = 0;
    logic output_valid;
    logic output_ready = 0;
    logic signed [32:0] model_score0_q10;
    logic signed [32:0] model_score1_q10;

    dual_requantizer_q20 dut (.*);
    always #5 clk = ~clk;

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        accumulator0 = 7;
        accumulator1 = -7;
        multiplier0_q20 = 1 <<< 20;
        multiplier1_q20 = 1 <<< 20;
        bias0_q10 = 3;
        bias1_q10 = -3;
        input_valid = 1;
        @(posedge clk);
        @(negedge clk);
        input_valid = 0;
        if (!output_valid || model_score0_q10 !== 10 || model_score1_q10 !== -10) begin
            $fatal(1, "integer requantization mismatch");
        end
        repeat (3) begin
            @(posedge clk);
            @(negedge clk);
            if (!output_valid || model_score0_q10 !== 10 || model_score1_q10 !== -10) begin
                $fatal(1, "requantized output changed under backpressure");
            end
        end
        output_ready = 1;
        accumulator0 = 3;
        accumulator1 = -3;
        multiplier0_q20 = 1 <<< 19;
        multiplier1_q20 = 1 <<< 19;
        bias0_q10 = 0;
        bias1_q10 = 0;
        input_valid = 1;
        @(posedge clk);
        @(negedge clk);
        input_valid = 0;
        if (!output_valid || model_score0_q10 !== 2 || model_score1_q10 !== -2) begin
            $fatal(1, "half-away-from-zero rounding mismatch");
        end
        @(posedge clk);
        @(negedge clk);
        if (output_valid) begin
            $fatal(1, "accepted requantized output did not clear");
        end

        $display("tb_dual_requantizer_q20: all checks passed");
        $finish;
    end

endmodule
