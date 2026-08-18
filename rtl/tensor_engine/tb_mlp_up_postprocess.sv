`timescale 1ns/1ps

module tb_mlp_up_postprocess;
    localparam integer LANES = 4;
    localparam integer ACC_WIDTH = 32;
    localparam integer MULTIPLIER_WIDTH = 24;
    localparam integer DATA_WIDTH = 16;
    localparam integer RIGHT_SHIFT = 20;
    localparam integer TAG_WIDTH = 8;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    reg [TAG_WIDTH-1:0] tag_in = '0;
    reg [LANES*ACC_WIDTH-1:0] accumulators_packed = '0;
    reg [LANES*MULTIPLIER_WIDTH-1:0] multipliers_packed = '0;
    reg [LANES*ACC_WIDTH-1:0] biases_packed = '0;
    wire valid_out;
    wire [TAG_WIDTH-1:0] tag_out;
    wire [LANES*DATA_WIDTH-1:0] gelu_packed;

    integer output_count = 0;
    reg signed [DATA_WIDTH-1:0] actual;

    mlp_up_postprocess #(
        .LANES(LANES),
        .ACC_WIDTH(ACC_WIDTH),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .RIGHT_SHIFT(RIGHT_SHIFT),
        .TAG_WIDTH(TAG_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .tag_in(tag_in),
        .accumulators_packed(accumulators_packed),
        .multipliers_packed(multipliers_packed),
        .biases_packed(biases_packed),
        .valid_out(valid_out),
        .tag_out(tag_out),
        .gelu_packed(gelu_packed)
    );

    always #5 clk = ~clk;

    task drive_vector;
        input [TAG_WIDTH-1:0] tag;
        input signed [ACC_WIDTH-1:0] lane_0;
        input signed [ACC_WIDTH-1:0] lane_1;
        input signed [ACC_WIDTH-1:0] lane_2;
        input signed [ACC_WIDTH-1:0] lane_3;
        begin
            @(negedge clk);
            tag_in = tag;
            accumulators_packed[0*ACC_WIDTH +: ACC_WIDTH] = lane_0;
            accumulators_packed[1*ACC_WIDTH +: ACC_WIDTH] = lane_1;
            accumulators_packed[2*ACC_WIDTH +: ACC_WIDTH] = lane_2;
            accumulators_packed[3*ACC_WIDTH +: ACC_WIDTH] = lane_3;
            valid_in = 1'b1;
            @(posedge clk);
            #1;
        end
    endtask

    task expect_lane;
        input integer lane;
        input signed [DATA_WIDTH-1:0] expected;
        begin
            actual = $signed(gelu_packed[lane*DATA_WIDTH +: DATA_WIDTH]);
            if (actual !== expected) begin
                $fatal(1, "lane %0d expected %0d got %0d", lane, expected, actual);
            end
        end
    endtask

    always @(posedge clk) begin
        #1;
        if (valid_out) begin
            if (output_count == 0) begin
                if (tag_out !== 8'h31) $fatal(1, "first tag mismatch");
                expect_lane(0, 0);
                expect_lane(1, -163);
                expect_lane(2, 861);
                expect_lane(3, 6144);
            end else if (output_count == 1) begin
                if (tag_out !== 8'h52) $fatal(1, "second tag mismatch");
                expect_lane(0, 0);
                expect_lane(1, 0);
                expect_lane(2, 9216);
                expect_lane(3, 861);
            end else begin
                $fatal(1, "unexpected extra output");
            end
            output_count = output_count + 1;
        end
    end

    initial begin
        multipliers_packed = {
            LANES{24'h100000}
        };
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        drive_vector(8'h31, -9216, -1024, 1024, 6144);
        drive_vector(8'h52, -16384, 0, 9216, 1024);
        @(negedge clk);
        valid_in = 1'b0;

        repeat (5) @(posedge clk);
        #1;
        if (output_count !== 2) begin
            $fatal(1, "expected two outputs, got %0d", output_count);
        end
        $display("tb_mlp_up_postprocess: PASS");
        $finish;
    end
endmodule
