`timescale 1ns/1ps

module tb_gumbel_q10_dual;
    logic clk = 0;
    logic rst_n = 0;
    logic input_valid = 0;
    logic input_ready;
    logic [31:0] input_word0 = 0;
    logic [31:0] input_word1 = 0;
    logic output_valid;
    logic output_ready = 1;
    logic signed [15:0] output_score0_q10;
    logic signed [15:0] output_score1_q10;
    gumbel_q10_dual dut (.*);
    always #5 clk = ~clk;

    task automatic check_pair(
        input [31:0] word0,
        input [31:0] word1,
        input signed [15:0] expected0,
        input signed [15:0] expected1
    );
        begin
            input_word0 = word0;
            input_word1 = word1;
            while (!input_ready) @(negedge clk);
            input_valid = 1;
            @(posedge clk);
            @(negedge clk);
            input_valid = 0;
            while (!output_valid) begin
                @(posedge clk);
                @(negedge clk);
            end
            if (
                output_score0_q10 !== expected0 ||
                output_score1_q10 !== expected1
            ) begin
                $fatal(1, "dual Gumbel score mismatch");
            end
            @(posedge clk);
            @(negedge clk);
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;

        check_pair(32'h0000_0000, 32'h0000_0001, -1983, -1983);
        check_pair(32'h7fff_ffff, 32'h8000_0000, 372, 377);
        check_pair(32'hffff_ff00, 32'hffff_fffe, 17043, 22718);
        check_pair(32'hffff_ffff, 32'hffff_ffff, 23423, 23423);

        $display("tb_gumbel_q10_dual: all checks passed");
        $finish;
    end

endmodule
