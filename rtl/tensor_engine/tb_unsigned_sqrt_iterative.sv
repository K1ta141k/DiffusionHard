`timescale 1ns/1ps

module tb_unsigned_sqrt_iterative;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg start = 1'b0;
    wire ready;
    reg [43:0] radicand = 0;
    wire busy;
    wire valid_out;
    wire [21:0] root;
    wire [23:0] remainder;
    integer result_count = 0;

    unsigned_sqrt_iterative dut (
        .clk(clk), .rst_n(rst_n), .start(start), .ready(ready),
        .radicand(radicand), .busy(busy), .valid_out(valid_out),
        .root(root), .remainder(remainder)
    );

    always #2 clk = ~clk;

    always @(posedge clk) begin
        #1;
        if (valid_out) begin
            if (result_count == 0) begin
                if (root !== 22'd12345 || remainder !== 0)
                    $fatal(1, "perfect square mismatch");
            end else if (result_count == 1) begin
                if (root !== 22'd111 || remainder !== 24)
                    $fatal(1, "non-square mismatch root=%0d rem=%0d",
                        root, remainder);
            end else begin
                $fatal(1, "unexpected square-root output");
            end
            result_count = result_count + 1;
        end
    end

    task calculate;
        input [43:0] value;
        begin
            wait (ready);
            @(negedge clk);
            radicand = value;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            wait (valid_out);
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        calculate(44'd152399025);
        calculate(44'd12345);
        repeat (2) @(posedge clk);
        if (result_count != 2) $fatal(1, "missing square-root outputs");
        $display("tb_unsigned_sqrt_iterative: PASS");
        $finish;
    end
endmodule
