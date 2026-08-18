`timescale 1ns/1ps

module tb_int8_mac_tile;
    localparam integer M_LANES = 2;
    localparam integer N_LANES = 2;
    localparam integer K_LANES = 4;
    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH = 32;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    reg clear_accumulators = 1'b0;
    reg last_k_tile = 1'b0;
    reg [M_LANES*K_LANES*DATA_WIDTH-1:0] activations_packed;
    reg [N_LANES*K_LANES*DATA_WIDTH-1:0] weights_packed;
    wire valid_out;
    wire [M_LANES*N_LANES*ACC_WIDTH-1:0] accumulators_packed;

    int8_mac_tile #(
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .K_LANES(K_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .clear_accumulators(clear_accumulators),
        .last_k_tile(last_k_tile),
        .activations_packed(activations_packed),
        .weights_packed(weights_packed),
        .valid_out(valid_out),
        .accumulators_packed(accumulators_packed)
    );

    always #5 clk = ~clk;

    task set_activation;
        input integer m;
        input integer k;
        input signed [DATA_WIDTH-1:0] value;
        begin
            activations_packed[(m*K_LANES+k)*DATA_WIDTH +: DATA_WIDTH] = value;
        end
    endtask

    task set_weight;
        input integer n;
        input integer k;
        input signed [DATA_WIDTH-1:0] value;
        begin
            weights_packed[(n*K_LANES+k)*DATA_WIDTH +: DATA_WIDTH] = value;
        end
    endtask

    function signed [ACC_WIDTH-1:0] accumulator;
        input integer m;
        input integer n;
        begin
            accumulator = $signed(
                accumulators_packed[(m*N_LANES+n)*ACC_WIDTH +: ACC_WIDTH]
            );
        end
    endfunction

    task expect_accumulator;
        input integer m;
        input integer n;
        input signed [ACC_WIDTH-1:0] expected;
        reg signed [ACC_WIDTH-1:0] actual;
        begin
            actual = accumulator(m, n);
            if (actual !== expected) begin
                $display(
                    "FAIL accumulator[%0d,%0d] expected %0d got %0d",
                    m, n, expected, actual
                );
                $fatal(1);
            end
        end
    endtask

    initial begin
        activations_packed = '0;
        weights_packed = '0;
        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        set_activation(0, 0, 1);
        set_activation(0, 1, 2);
        set_activation(0, 2, 3);
        set_activation(0, 3, 4);
        set_activation(1, 0, -1);
        set_activation(1, 1, 0);
        set_activation(1, 2, 2);
        set_activation(1, 3, -2);
        set_weight(0, 0, 2);
        set_weight(0, 1, -1);
        set_weight(0, 2, 1);
        set_weight(0, 3, 3);
        set_weight(1, 0, -2);
        set_weight(1, 1, 4);
        set_weight(1, 2, 0);
        set_weight(1, 3, 1);
        valid_in = 1'b1;
        clear_accumulators = 1'b1;
        last_k_tile = 1'b0;
        @(posedge clk);
        #1;
        if (valid_out !== 1'b0) $fatal(1, "valid_out asserted early");

        set_activation(0, 0, 5);
        set_activation(0, 1, -1);
        set_activation(0, 2, 2);
        set_activation(0, 3, 0);
        set_activation(1, 0, 3);
        set_activation(1, 1, 3);
        set_activation(1, 2, -3);
        set_activation(1, 3, 1);
        set_weight(0, 0, -1);
        set_weight(0, 1, 2);
        set_weight(0, 2, 4);
        set_weight(0, 3, -2);
        set_weight(1, 0, 1);
        set_weight(1, 1, 1);
        set_weight(1, 2, -1);
        set_weight(1, 3, 3);
        clear_accumulators = 1'b0;
        last_k_tile = 1'b1;
        @(posedge clk);
        #1;
        if (valid_out !== 1'b1) $fatal(1, "valid_out missing on final tile");
        expect_accumulator(0, 0, 16);
        expect_accumulator(0, 1, 12);
        expect_accumulator(1, 0, -17);
        expect_accumulator(1, 1, 12);

        valid_in = 1'b0;
        last_k_tile = 1'b0;
        @(posedge clk);
        #1;
        if (valid_out !== 1'b0) $fatal(1, "valid_out did not clear");
        expect_accumulator(0, 0, 16);
        expect_accumulator(1, 0, -17);

        $display("tb_int8_mac_tile: PASS");
        $finish;
    end
endmodule
