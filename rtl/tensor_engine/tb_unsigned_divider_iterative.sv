`timescale 1ns/1ps

module tb_unsigned_divider_iterative;
    localparam integer WIDTH = 26;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg start = 1'b0;
    wire ready;
    reg [WIDTH-1:0] dividend = 0;
    reg [WIDTH-1:0] divisor = 0;
    wire busy;
    wire valid_out;
    wire [WIDTH-1:0] quotient;
    wire [WIDTH-1:0] remainder;
    integer result_count = 0;

    unsigned_divider_iterative #(.WIDTH(WIDTH)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .ready(ready),
        .dividend(dividend),
        .divisor(divisor),
        .busy(busy),
        .valid_out(valid_out),
        .quotient(quotient),
        .remainder(remainder)
    );

    always #2 clk = ~clk;

    always @(posedge clk) begin
        #1;
        if (valid_out) begin
            if (result_count == 0) begin
                if (quotient !== 8128 || remainder !== 0)
                    $fatal(1, "power-of-two division mismatch");
            end else if (result_count == 1) begin
                if (quotient !== 97 || remainder !== 26)
                    $fatal(1, "ordinary division mismatch");
            end else begin
                $fatal(1, "unexpected divider output");
            end
            result_count = result_count + 1;
        end
    end

    task divide;
        input [WIDTH-1:0] numerator;
        input [WIDTH-1:0] denominator;
        begin
            wait (ready);
            @(negedge clk);
            dividend = numerator;
            divisor = denominator;
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
        divide(26'd33292288, 26'd4096);
        divide(26'd12345, 26'd127);
        repeat (2) @(posedge clk);
        if (result_count != 2) $fatal(1, "missing divider outputs");
        $display("tb_unsigned_divider_iterative: PASS");
        $finish;
    end
endmodule
