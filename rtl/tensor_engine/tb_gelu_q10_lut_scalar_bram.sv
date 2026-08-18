`timescale 1ns/1ps

module tb_gelu_q10_lut_scalar_bram;
    localparam integer DATA_WIDTH = 16;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    reg [DATA_WIDTH-1:0] input_value = '0;
    wire valid_out;
    wire [DATA_WIDTH-1:0] output_value;
    integer output_count = 0;
    reg signed [DATA_WIDTH-1:0] actual;

    gelu_q10_lut_scalar_bram dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .input_value(input_value),
        .valid_out(valid_out),
        .output_value(output_value)
    );

    always #5 clk = ~clk;

    task drive;
        input signed [DATA_WIDTH-1:0] value;
        begin
            @(negedge clk);
            input_value = value;
            valid_in = 1'b1;
            @(posedge clk);
            #1;
        end
    endtask

    always @(posedge clk) begin
        #1;
        if (valid_out) begin
            actual = $signed(output_value);
            case (output_count)
                0: if (actual !== 0) $fatal(1, "-9 output mismatch");
                1: if (actual !== -163) $fatal(1, "-1 output mismatch");
                2: if (actual !== 0) $fatal(1, "zero output mismatch");
                3: if (actual !== 861) $fatal(1, "+1 output mismatch");
                4: if (actual !== 6144) $fatal(1, "+6 output mismatch");
                5: if (actual !== 9216) $fatal(1, "+9 output mismatch");
                default: $fatal(1, "unexpected output");
            endcase
            output_count = output_count + 1;
        end
    end

    initial begin
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        drive(-9216);
        drive(-1024);
        drive(0);
        drive(1024);
        drive(6144);
        drive(9216);
        @(negedge clk);
        valid_in = 1'b0;
        repeat (6) @(posedge clk);
        #1;
        if (output_count !== 6) $fatal(1, "expected six outputs");
        $display("tb_gelu_q10_lut_scalar_bram: PASS");
        $finish;
    end
endmodule
