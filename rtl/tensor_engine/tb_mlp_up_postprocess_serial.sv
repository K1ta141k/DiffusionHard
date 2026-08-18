`timescale 1ns/1ps

module tb_mlp_up_postprocess_serial;
    localparam integer M_LANES = 2;
    localparam integer N_LANES = 2;
    localparam integer LANES = M_LANES * N_LANES;
    localparam integer ACC_WIDTH = 32;
    localparam integer MULTIPLIER_WIDTH = 24;
    localparam integer DATA_WIDTH = 16;
    localparam integer TAG_WIDTH = 8;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    wire ready_in;
    reg [TAG_WIDTH-1:0] tag_in = '0;
    reg [LANES*ACC_WIDTH-1:0] accumulators_packed = '0;
    reg [M_LANES*16-1:0] token_factors_packed = '0;
    reg [N_LANES*18-1:0] output_factors_packed = '0;
    reg [N_LANES*ACC_WIDTH-1:0] biases_packed = '0;
    reg [N_LANES*24-1:0] sideband_in = '0;
    wire valid_out;
    wire [TAG_WIDTH-1:0] tag_out;
    wire [LANES*DATA_WIDTH-1:0] gelu_packed;
    wire [N_LANES*24-1:0] sideband_out;

    integer output_count = 0;
    reg signed [DATA_WIDTH-1:0] actual;

    mlp_up_postprocess_serial #(
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .ACC_WIDTH(ACC_WIDTH),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .TAG_WIDTH(TAG_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .tag_in(tag_in),
        .accumulators_packed(accumulators_packed),
        .token_factors_packed(token_factors_packed),
        .output_factors_packed(output_factors_packed),
        .biases_packed(biases_packed),
        .sideband_in(sideband_in),
        .valid_out(valid_out),
        .tag_out(tag_out),
        .gelu_packed(gelu_packed),
        .sideband_out(sideband_out)
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
            if (!ready_in) $fatal(1, "serializer input queue full");
            tag_in = tag;
            accumulators_packed[0*ACC_WIDTH +: ACC_WIDTH] = lane_0;
            accumulators_packed[1*ACC_WIDTH +: ACC_WIDTH] = lane_1;
            accumulators_packed[2*ACC_WIDTH +: ACC_WIDTH] = lane_2;
            accumulators_packed[3*ACC_WIDTH +: ACC_WIDTH] = lane_3;
            valid_in = 1'b1;
            @(posedge clk);
            #1;
            @(negedge clk);
            valid_in = 1'b0;
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
                if (sideband_out !== 48'h123456abcdef)
                    $fatal(1, "first sideband mismatch");
            end else if (output_count == 1) begin
                if (tag_out !== 8'h52) $fatal(1, "second tag mismatch");
                expect_lane(0, 0);
                expect_lane(1, 0);
                expect_lane(2, 9216);
                expect_lane(3, 861);
                if (sideband_out !== 48'h654321fedcba)
                    $fatal(1, "second sideband mismatch");
            end else begin
                $fatal(1, "unexpected extra output");
            end
            output_count = output_count + 1;
        end
    end

    initial begin
        token_factors_packed = {M_LANES{16'd4096}};
        output_factors_packed = {N_LANES{18'd65536}};
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        sideband_in = 48'h123456abcdef;
        drive_vector(8'h31, -9216, -1024, 1024, 6144);
        repeat (2) @(posedge clk);
        sideband_in = 48'h654321fedcba;
        drive_vector(8'h52, -16384, 0, 9216, 1024);
        repeat (18) @(posedge clk);
        #1;
        if (output_count !== 2) begin
            $fatal(1, "expected two outputs, got %0d", output_count);
        end
        $display("tb_mlp_up_postprocess_serial: PASS");
        $finish;
    end
endmodule
