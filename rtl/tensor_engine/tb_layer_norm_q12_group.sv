`timescale 1ns/1ps

module tb_layer_norm_q12_group;
    localparam integer INPUT_SIZE = 768;
    localparam integer INPUT_WIDTH = 24;
    localparam integer OUTPUT_WIDTH = 18;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg start = 1'b0;
    reg [3:0] group_in = 4'd0;
    wire start_ready;
    reg start_replay = 1'b0;
    reg final_replay = 1'b0;
    wire replay_ready;
    reg input_valid = 1'b0;
    wire input_ready;
    reg [4*INPUT_WIDTH-1:0] input_q10_packed = 0;
    wire output_valid;
    wire [3:0] output_group;
    wire [9:0] output_channel;
    wire [4*OUTPUT_WIDTH-1:0] output_q12_packed;
    wire busy;
    wire done;

    integer channel;
    integer output_count = 0;
    integer replay_count = 0;
    integer token;
    integer expected;

    layer_norm_q12_group #(
        .INPUT_SIZE(INPUT_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .group_in(group_in),
        .start_ready(start_ready),
        .start_replay(start_replay),
        .final_replay(final_replay),
        .replay_ready(replay_ready),
        .input_valid(input_valid),
        .input_ready(input_ready),
        .input_q10_packed(input_q10_packed),
        .output_valid(output_valid),
        .output_group(output_group),
        .output_channel(output_channel),
        .output_q12_packed(output_q12_packed),
        .busy(busy),
        .done(done)
    );

    always #2 clk = ~clk;

    task stream_input;
        input integer check_output;
        begin
            for (channel = 0; channel < INPUT_SIZE;
                 channel = channel + 1) begin
                @(negedge clk);
                if (!input_ready)
                    $fatal(1, "input not ready at channel %0d", channel);
                input_valid = 1'b1;
                input_q10_packed[0*INPUT_WIDTH +: INPUT_WIDTH] =
                    channel[0] ? 24'sd1024 : -24'sd1024;
                input_q10_packed[1*INPUT_WIDTH +: INPUT_WIDTH] =
                    channel[0] ? 24'sd2048 : -24'sd2048;
                input_q10_packed[2*INPUT_WIDTH +: INPUT_WIDTH] =
                    channel[0] ? 24'sd3072 : -24'sd3072;
                input_q10_packed[3*INPUT_WIDTH +: INPUT_WIDTH] =
                    channel[0] ? 24'sd4096 : -24'sd4096;
            end
            @(negedge clk);
            input_valid = 1'b0;
        end
    endtask

    always @(posedge clk) begin
        #1;
        if (output_valid) begin
            if (output_group !== 4'd9)
                $fatal(1, "output group mismatch");
            if (output_channel !== (output_count % INPUT_SIZE))
                $fatal(1, "channel tag mismatch at output %0d", output_count);
            expected = (output_channel[0]) ? 4096 : -4096;
            for (token = 0; token < 4; token = token + 1) begin
                if ($signed(output_q12_packed[
                        token*OUTPUT_WIDTH +: OUTPUT_WIDTH
                    ]) !== expected)
                    $fatal(
                        1,
                        "normalized value mismatch at output %0d token %0d",
                        output_count,
                        token
                    );
            end
            output_count = output_count + 1;
            if ((output_count % INPUT_SIZE) == 0)
                replay_count = replay_count + 1;
        end
    end

    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        group_in = 4'd9;
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        stream_input(0);

        wait (replay_ready);
        @(negedge clk);
        final_replay = 1'b0;
        start_replay = 1'b1;
        @(negedge clk);
        start_replay = 1'b0;
        stream_input(1);

        wait (replay_ready);
        @(negedge clk);
        final_replay = 1'b1;
        start_replay = 1'b1;
        @(negedge clk);
        start_replay = 1'b0;
        stream_input(1);

        wait (done);
        repeat (2) @(posedge clk);
        if (output_count != 2*INPUT_SIZE)
            $fatal(1, "expected %0d outputs, got %0d", 2*INPUT_SIZE,
                   output_count);
        if (replay_count != 2)
            $fatal(1, "expected two complete replays");
        if (busy)
            $fatal(1, "busy remained asserted after final replay");
        if (!start_ready)
            $fatal(1, "core did not return to idle");
        $display("tb_layer_norm_q12_group: PASS");
        $finish;
    end

    initial begin
        repeat (10000) @(posedge clk);
        $fatal(1, "timeout");
    end

endmodule
