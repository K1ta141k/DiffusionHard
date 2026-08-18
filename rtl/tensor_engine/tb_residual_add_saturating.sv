`timescale 1ns/1ps

module tb_residual_add_saturating;
    localparam integer LANES = 4;
    localparam integer DATA_WIDTH = 24;
    localparam integer TAG_WIDTH = 6;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    reg [TAG_WIDTH-1:0] tag_in = '0;
    reg [LANES*DATA_WIDTH-1:0] values_packed = '0;
    reg [LANES*DATA_WIDTH-1:0] residuals_packed = '0;
    wire valid_out;
    wire [TAG_WIDTH-1:0] tag_out;
    wire [LANES*DATA_WIDTH-1:0] outputs_packed;
    reg signed [DATA_WIDTH-1:0] actual;

    residual_add_saturating #(
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .TAG_WIDTH(TAG_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .tag_in(tag_in),
        .values_packed(values_packed),
        .residuals_packed(residuals_packed),
        .valid_out(valid_out),
        .tag_out(tag_out),
        .outputs_packed(outputs_packed)
    );

    always #5 clk = ~clk;

    task set_lane;
        input integer lane;
        input signed [DATA_WIDTH-1:0] value;
        input signed [DATA_WIDTH-1:0] residual;
        begin
            values_packed[lane*DATA_WIDTH +: DATA_WIDTH] = value;
            residuals_packed[lane*DATA_WIDTH +: DATA_WIDTH] = residual;
        end
    endtask

    task expect_lane;
        input integer lane;
        input signed [DATA_WIDTH-1:0] expected;
        begin
            actual = $signed(outputs_packed[lane*DATA_WIDTH +: DATA_WIDTH]);
            if (actual !== expected) begin
                $fatal(1, "lane %0d expected %0d got %0d", lane, expected, actual);
            end
        end
    endtask

    initial begin
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        tag_in = 6'h2d;
        set_lane(0, 8388600, 100);
        set_lane(1, -8388600, -100);
        set_lane(2, 1234, -234);
        set_lane(3, -2000, 500);
        valid_in = 1'b1;
        @(posedge clk);
        #1;
        if (!valid_out) $fatal(1, "valid_out missing");
        if (tag_out !== 6'h2d) $fatal(1, "tag mismatch");
        expect_lane(0, 8388607);
        expect_lane(1, -8388608);
        expect_lane(2, 1000);
        expect_lane(3, -1500);
        $display("tb_residual_add_saturating: PASS");
        $finish;
    end
endmodule
