`timescale 1ns/1ps

module tb_philox4x32_iterative;
    logic clk = 0;
    logic rst_n = 0;
    logic input_valid = 0;
    logic input_ready;
    logic [31:0] input_c0 = 0;
    logic [31:0] input_c1 = 0;
    logic [31:0] input_c2 = 0;
    logic [31:0] input_c3 = 0;
    logic [31:0] input_k0 = 0;
    logic [31:0] input_k1 = 0;
    logic output_valid;
    logic output_ready = 0;
    logic [127:0] output_words;
    logic busy;
    logic [127:0] held_words;
    integer cycles;

    philox4x32_iterative dut (.*);

    always #5 clk = ~clk;

    task automatic issue(
        input [31:0] c0,
        input [31:0] c1,
        input [31:0] c2,
        input [31:0] c3,
        input [31:0] k0,
        input [31:0] k1
    );
        begin
            while (!input_ready) @(negedge clk);
            input_c0 = c0;
            input_c1 = c1;
            input_c2 = c2;
            input_c3 = c3;
            input_k0 = k0;
            input_k1 = k1;
            input_valid = 1;
            @(negedge clk);
            input_valid = 0;
        end
    endtask

    task automatic expect_result(input [127:0] expected);
        begin
            cycles = 0;
            while (!output_valid) begin
                @(negedge clk);
                cycles = cycles + 1;
            end
            if (cycles != 9) begin
                $fatal(1, "unexpected iterative latency: %0d", cycles);
            end
            if (output_words !== expected) begin
                $fatal(1, "Philox known-answer mismatch");
            end
            held_words = output_words;
            repeat (3) begin
                @(posedge clk);
                @(negedge clk);
                if (!output_valid || output_words !== held_words) begin
                    $fatal(1, "result changed under backpressure");
                end
            end
            output_ready = 1;
            @(posedge clk);
            @(negedge clk);
            output_ready = 0;
            if (output_valid) begin
                $fatal(1, "accepted result did not clear");
            end
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;

        issue(0, 0, 0, 0, 0, 0);
        expect_result(128'h9b00dbd8_bc57ac4c_e169c58d_6627e8d5);

        issue(
            32'hffff_ffff,
            32'hffff_ffff,
            32'hffff_ffff,
            32'hffff_ffff,
            32'hffff_ffff,
            32'hffff_ffff
        );
        expect_result(128'h6d5451fd_a20bc7c6_41c83b0e_408f276d);

        issue(
            32'h243f6a88,
            32'h85a308d3,
            32'h13198a2e,
            32'h03707344,
            32'ha4093822,
            32'h299f31d0
        );
        expect_result(128'h24126ea1_5001e420_94fdcceb_d16cfe09);

        $display("tb_philox4x32_iterative: all checks passed");
        $finish;
    end

endmodule
