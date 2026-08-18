`timescale 1ns/1ps

module tb_candidate_reveal_stream;
    localparam integer MAX_POSITIONS = 256;
    localparam integer COUNT_WIDTH = 9;
    localparam integer INDEX_WIDTH = 8;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start_valid = 1'b0;
    logic start_ready;
    logic [COUNT_WIDTH-1:0] position_count = '0;
    logic [32:0] reveal_threshold_q32 = '0;
    logic in_valid = 1'b0;
    logic in_ready;
    logic in_active = 1'b0;
    logic in_candidate_valid = 1'b0;
    logic [15:0] in_token_id = '0;
    logic [15:0] in_candidate_id = '0;
    logic [31:0] in_random_word = '0;
    logic out_valid;
    logic out_ready = 1'b1;
    logic [INDEX_WIDTH-1:0] out_position;
    logic out_active;
    logic out_candidate_valid;
    logic [15:0] out_token_id;
    logic out_changed;
    logic busy;
    logic done;
    logic invalidate_all;
    logic [COUNT_WIDTH-1:0] changed_count;
    logic [1:0] status;

    integer expected_changes;
    integer position;
    integer test_index;
    logic expected_changed;
    logic [31:0] lfsr;

    always #5 clk = ~clk;

    candidate_reveal_stream dut (
        .clk,
        .rst_n,
        .start_valid,
        .start_ready,
        .position_count,
        .reveal_threshold_q32,
        .in_valid,
        .in_ready,
        .in_active,
        .in_candidate_valid,
        .in_token_id,
        .in_candidate_id,
        .in_random_word,
        .out_valid,
        .out_ready,
        .out_position,
        .out_active,
        .out_candidate_valid,
        .out_token_id,
        .out_changed,
        .busy,
        .done,
        .invalidate_all,
        .changed_count,
        .status
    );

    task automatic begin_command(
        input logic [COUNT_WIDTH-1:0] count,
        input logic [32:0] threshold
    );
        begin
            @(negedge clk);
            position_count = count;
            reveal_threshold_q32 = threshold;
            start_valid = 1'b1;
            @(posedge clk);
            #1;
            start_valid = 1'b0;
        end
    endtask

    task automatic send_position(
        input integer index,
        input logic active,
        input logic candidate_valid,
        input logic [31:0] random_word,
        input logic expected_reveal
    );
        begin
            @(negedge clk);
            in_active = active;
            in_candidate_valid = candidate_valid;
            in_token_id = 16'd50257;
            in_candidate_id = 16'(100 + index);
            in_random_word = random_word;
            in_valid = 1'b1;
            #1;
            if (!in_ready || !out_valid) $fatal(1, "stream was not ready");
            if (out_position != index[INDEX_WIDTH-1:0]) $fatal(1, "bad index");
            if (out_changed != expected_reveal) $fatal(1, "bad reveal result");
            if (out_active != (active && !expected_reveal)) $fatal(1, "bad active");
            if (out_candidate_valid != (candidate_valid && !expected_reveal))
                $fatal(1, "bad candidate validity");
            if (out_token_id != (expected_reveal ? 16'(100 + index) : 16'd50257))
                $fatal(1, "bad output token");
            @(posedge clk);
            #1;
            in_valid = 1'b0;
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        begin_command(9'd8, 33'd0);
        for (position = 0; position < 8; position = position + 1)
            send_position(position, 1'b1, 1'b1, 32'd0, 1'b0);
        if (!done || changed_count != 0 || invalidate_all) $fatal(1, "zero case");

        begin_command(9'd8, 33'h1_0000_0000);
        for (position = 0; position < 8; position = position + 1)
            send_position(position, 1'b1, 1'b1, 32'hffff_ffff, 1'b1);
        if (!done || changed_count != 8 || !invalidate_all) $fatal(1, "one case");

        begin_command(9'd4, 33'h0_8000_0000);
        @(negedge clk);
        out_ready = 1'b0;
        in_valid = 1'b1;
        in_active = 1'b1;
        in_candidate_valid = 1'b1;
        in_random_word = 32'd0;
        #1;
        if (in_ready) $fatal(1, "backpressure failure");
        @(negedge clk);
        out_ready = 1'b1;
        in_valid = 1'b0;
        for (position = 0; position < 4; position = position + 1)
            send_position(position, 1'b1, 1'b1,
                          position[0] ? 32'hffff_ffff : 32'd0,
                          !position[0]);
        if (!done || changed_count != 2 || !invalidate_all) $fatal(1, "partial case");

        lfsr = 32'h1ace_b00c;
        for (test_index = 0; test_index < 16; test_index = test_index + 1) begin
            expected_changes = 0;
            begin_command(9'd64, {1'b0, test_index[3:0], 28'd0});
            for (position = 0; position < 64; position = position + 1) begin
                lfsr = {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
                expected_changed = lfsr < {test_index[3:0], 28'd0};
                if (expected_changed) expected_changes = expected_changes + 1;
                send_position(position, 1'b1, 1'b1, lfsr, expected_changed);
            end
            if (!done || changed_count != expected_changes) $fatal(1, "random count");
            if (invalidate_all != (expected_changes != 0)) $fatal(1, "random invalidate");
        end

        begin_command(9'd0, 33'd0);
        if (!done || status != 2'd1 || busy) $fatal(1, "invalid count");

        begin_command(9'd8, 33'h1_0000_0001);
        if (!done || status != 2'd2 || busy) $fatal(1, "invalid threshold");

        $display("tb_candidate_reveal_stream: all checks passed");
        $finish;
    end

endmodule
