`timescale 1ns/1ps

module tb_gelu_q10_lut;
    localparam integer LANES = 6;
    localparam integer DATA_WIDTH = 16;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    reg [LANES*DATA_WIDTH-1:0] inputs_packed = '0;
    wire valid_out;
    wire [LANES*DATA_WIDTH-1:0] outputs_packed;

    gelu_q10_lut #(
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .inputs_packed(inputs_packed),
        .valid_out(valid_out),
        .outputs_packed(outputs_packed)
    );

    always #5 clk = ~clk;

    task set_input;
        input integer lane;
        input signed [DATA_WIDTH-1:0] value;
        begin
            inputs_packed[lane*DATA_WIDTH +: DATA_WIDTH] = value;
        end
    endtask

    task expect_output;
        input integer lane;
        input signed [DATA_WIDTH-1:0] expected;
        reg signed [DATA_WIDTH-1:0] actual;
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
        set_input(0, -9216);
        set_input(1, -1024);
        set_input(2, 0);
        set_input(3, 1024);
        set_input(4, 6144);
        set_input(5, 9216);
        valid_in = 1'b1;
        @(posedge clk);
        #1;
        if (!valid_out) $fatal(1, "valid_out missing");
        expect_output(0, 0);
        expect_output(1, -163);
        expect_output(2, 0);
        expect_output(3, 861);
        expect_output(4, 6144);
        expect_output(5, 9216);

        @(negedge clk);
        valid_in = 1'b0;
        @(posedge clk);
        #1;
        if (valid_out) $fatal(1, "valid_out did not clear");
        $display("tb_gelu_q10_lut: PASS");
        $finish;
    end
endmodule
