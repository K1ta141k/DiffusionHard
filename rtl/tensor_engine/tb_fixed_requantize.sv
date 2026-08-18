`timescale 1ns/1ps

module tb_fixed_requantize;
    localparam integer LANES = 4;
    localparam integer ACC_WIDTH = 32;
    localparam integer MULTIPLIER_WIDTH = 24;
    localparam integer OUTPUT_WIDTH = 16;
    localparam integer RIGHT_SHIFT = 10;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    reg [LANES*ACC_WIDTH-1:0] accumulators_packed = '0;
    reg [LANES*MULTIPLIER_WIDTH-1:0] multipliers_packed = '0;
    reg [LANES*ACC_WIDTH-1:0] biases_packed = '0;
    wire valid_out;
    wire [LANES*OUTPUT_WIDTH-1:0] outputs_packed;

    fixed_requantize #(
        .LANES(LANES),
        .ACC_WIDTH(ACC_WIDTH),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH),
        .RIGHT_SHIFT(RIGHT_SHIFT)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .accumulators_packed(accumulators_packed),
        .multipliers_packed(multipliers_packed),
        .biases_packed(biases_packed),
        .valid_out(valid_out),
        .outputs_packed(outputs_packed)
    );

    always #5 clk = ~clk;

    task set_lane;
        input integer lane;
        input signed [ACC_WIDTH-1:0] accumulator;
        input [MULTIPLIER_WIDTH-1:0] multiplier;
        input signed [ACC_WIDTH-1:0] bias;
        begin
            accumulators_packed[lane*ACC_WIDTH +: ACC_WIDTH] = accumulator;
            multipliers_packed[lane*MULTIPLIER_WIDTH +: MULTIPLIER_WIDTH] = multiplier;
            biases_packed[lane*ACC_WIDTH +: ACC_WIDTH] = bias;
        end
    endtask

    task expect_lane;
        input integer lane;
        input signed [OUTPUT_WIDTH-1:0] expected;
        reg signed [OUTPUT_WIDTH-1:0] actual;
        begin
            actual = $signed(outputs_packed[lane*OUTPUT_WIDTH +: OUTPUT_WIDTH]);
            if (actual !== expected) begin
                $fatal(1, "lane %0d expected %0d got %0d", lane, expected, actual);
            end
        end
    endtask

    initial begin
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        set_lane(0, 1000, 1024, 10);
        set_lane(1, -1001, 512, -10);
        set_lane(2, 100000, 1024, 0);
        set_lane(3, -100000, 1024, 0);
        valid_in = 1'b1;
        @(posedge clk);
        #1;
        if (!valid_out) $fatal(1, "valid_out missing");
        expect_lane(0, 1010);
        expect_lane(1, -511);
        expect_lane(2, 32767);
        expect_lane(3, -32768);

        @(negedge clk);
        valid_in = 1'b0;
        @(posedge clk);
        #1;
        if (valid_out) $fatal(1, "valid_out did not clear");
        $display("tb_fixed_requantize: PASS");
        $finish;
    end
endmodule
