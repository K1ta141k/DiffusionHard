`timescale 1ns/1ps

module tb_mlp_up_activation_quantizer;
    localparam integer INPUT_SIZE = 64;
    localparam integer M_LANES = 4;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg start = 1'b0;
    reg [3:0] group_in = 4'd3;
    wire start_ready;
    reg start_pass2 = 1'b0;
    wire pass2_ready;
    reg input_valid = 1'b0;
    wire input_ready;
    reg [M_LANES*18-1:0] normalized_q12_packed = 0;
    reg [17:0] smoothing_reciprocal_q15 = 18'd32768;
    wire token_factor_valid;
    wire [3:0] token_factor_group;
    wire [M_LANES*16-1:0] token_factors_packed;
    wire activation_load_valid;
    wire [3:0] activation_load_group;
    wire [0:0] activation_load_k_tile;
    wire [M_LANES*32*8-1:0] activation_load_data;
    wire busy;
    wire done;
    integer channel;
    integer token;
    integer tile_count = 0;
    integer maximum;
    integer multiplier;
    integer product;
    integer expected;

    mlp_up_activation_quantizer #(
        .INPUT_SIZE(INPUT_SIZE),
        .M_LANES(M_LANES),
        .GROUP_WIDTH(4),
        .K_TILE_WIDTH(1)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .group_in(group_in),
        .start_ready(start_ready),
        .start_pass2(start_pass2),
        .pass2_ready(pass2_ready),
        .input_valid(input_valid),
        .input_ready(input_ready),
        .normalized_q12_packed(normalized_q12_packed),
        .smoothing_reciprocal_q15(smoothing_reciprocal_q15),
        .token_factor_valid(token_factor_valid),
        .token_factor_group(token_factor_group),
        .token_factors_packed(token_factors_packed),
        .activation_load_valid(activation_load_valid),
        .activation_load_group(activation_load_group),
        .activation_load_k_tile(activation_load_k_tile),
        .activation_load_data(activation_load_data),
        .busy(busy),
        .done(done)
    );

    always #2 clk = ~clk;

    task stream_pass;
        begin
            for (channel = 0; channel < INPUT_SIZE; channel = channel + 1) begin
                @(negedge clk);
                if (!input_ready) $fatal(1, "quantizer input not ready");
                for (token = 0; token < M_LANES; token = token + 1)
                    normalized_q12_packed[token*18 +: 18] =
                        (channel + 1) * (token + 1) * 64;
                input_valid = 1'b1;
            end
            @(negedge clk);
            input_valid = 1'b0;
        end
    endtask

    always @(posedge clk) begin
        #1;
        if (token_factor_valid) begin
            if (token_factor_group !== 3) $fatal(1, "factor group mismatch");
            for (token = 0; token < M_LANES; token = token + 1) begin
                maximum = INPUT_SIZE * (token + 1) * 64;
                expected = (maximum*64 + 63) / 127;
                if (token_factors_packed[token*16 +: 16] !== expected)
                    $fatal(1, "token factor mismatch");
            end
        end
        if (activation_load_valid) begin
            if (activation_load_group !== 3) $fatal(1, "tile group mismatch");
            if (activation_load_k_tile !== tile_count)
                $fatal(1, "tile index mismatch");
            for (token = 0; token < M_LANES; token = token + 1) begin
                maximum = INPUT_SIZE * (token + 1) * 64;
                multiplier = (33292288 + maximum/2) / maximum;
                for (channel = 0; channel < 32; channel = channel + 1) begin
                    product = (tile_count*32 + channel + 1)
                        * (token + 1) * 64 * multiplier;
                    expected = (product + 131072) / 262144;
                    if ($signed(activation_load_data[
                        (token*32 + channel)*8 +: 8
                    ]) !== expected)
                        $fatal(1, "activation tile mismatch tile=%0d token=%0d channel=%0d expected=%0d actual=%0d",
                            tile_count, token, channel, expected,
                            $signed(activation_load_data[
                                (token*32 + channel)*8 +: 8
                            ]));
                end
            end
            tile_count = tile_count + 1;
        end
    end

    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        stream_pass();
        wait (pass2_ready);
        @(negedge clk);
        start_pass2 = 1'b1;
        @(negedge clk);
        start_pass2 = 1'b0;
        stream_pass();
        wait (done);
        repeat (2) @(posedge clk);
        if (tile_count != 2) $fatal(1, "expected two activation tiles");
        $display("tb_mlp_up_activation_quantizer: PASS");
        $finish;
    end
endmodule
