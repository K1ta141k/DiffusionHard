`timescale 1ns/1ps

module tb_smoothquant_int8_vector_serial;
    localparam integer LANES = 6;
    localparam integer TAG_WIDTH = 8;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    wire ready_in;
    reg [TAG_WIDTH-1:0] tag_in = 0;
    reg [LANES*16-1:0] inputs_packed = 0;
    reg [LANES*24-1:0] multipliers_packed = 0;
    wire valid_out;
    wire [TAG_WIDTH-1:0] tag_out;
    wire [LANES*8-1:0] outputs_packed;
    integer output_count = 0;

    smoothquant_int8_vector_serial #(
        .LANES(LANES),
        .TAG_WIDTH(TAG_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .tag_in(tag_in),
        .inputs_packed(inputs_packed),
        .multipliers_packed(multipliers_packed),
        .valid_out(valid_out),
        .tag_out(tag_out),
        .outputs_packed(outputs_packed)
    );

    always #2 clk = ~clk;

    task set_lane;
        input integer lane;
        input signed [15:0] value;
        input [23:0] multiplier;
        begin
            inputs_packed[lane*16 +: 16] = value;
            multipliers_packed[lane*24 +: 24] = multiplier;
        end
    endtask

    always @(posedge clk) begin
        if (valid_out) begin
            if (output_count == 0) begin
                if (tag_out !== 8'h31) $fatal(1, "first tag mismatch");
                if ($signed(outputs_packed[0*8 +: 8]) !== 64)
                    $fatal(1, "positive conversion mismatch");
                if ($signed(outputs_packed[1*8 +: 8]) !== -64)
                    $fatal(1, "negative conversion mismatch");
                if ($signed(outputs_packed[2*8 +: 8]) !== 127)
                    $fatal(1, "positive saturation mismatch");
                if ($signed(outputs_packed[3*8 +: 8]) !== -127)
                    $fatal(1, "negative saturation mismatch");
                if ($signed(outputs_packed[4*8 +: 8]) !== 0)
                    $fatal(1, "small value mismatch");
                if ($signed(outputs_packed[5*8 +: 8]) !== 0)
                    $fatal(1, "dead channel mismatch");
            end else if (output_count == 1) begin
                if (tag_out !== 8'h52) $fatal(1, "second tag mismatch");
                if ($signed(outputs_packed[0*8 +: 8]) !== 1)
                    $fatal(1, "positive half rounding mismatch");
                if ($signed(outputs_packed[1*8 +: 8]) !== -1)
                    $fatal(1, "negative half rounding mismatch");
            end else begin
                $fatal(1, "unexpected extra output");
            end
            output_count = output_count + 1;
        end
    end

    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        tag_in = 8'h31;
        set_lane(0, 16'sd1024, 24'd65536);
        set_lane(1, -16'sd1024, 24'd65536);
        set_lane(2, 16'sd32767, 24'd65536);
        set_lane(3, -16'sd32768, 24'd65536);
        set_lane(4, 16'sd512, 24'd1);
        set_lane(5, 16'sd4096, 24'd0);
        valid_in = 1'b1;
        @(negedge clk);

        tag_in = 8'h52;
        inputs_packed = 0;
        multipliers_packed = 0;
        set_lane(0, 16'sd8, 24'd65536);
        set_lane(1, -16'sd8, 24'd65536);
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        repeat (20) @(posedge clk);
        if (output_count != 2) $fatal(1, "missing outputs");
        $display("tb_smoothquant_int8_vector_serial: PASS");
        $finish;
    end
endmodule
