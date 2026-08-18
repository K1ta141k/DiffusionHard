`timescale 1ns/1ps

module tb_mlp_interstage_tile_bridge;
    localparam integer TOKENS = 4;
    localparam integer M_LANES = 2;
    localparam integer N_LANES = 6;
    localparam integer INPUT_SIZE = 96;
    localparam integer GROUP_WIDTH = 1;
    localparam integer K_TILE_WIDTH = 2;
    localparam integer OUTPUT_TILE_TAG_WIDTH = 5;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    reg [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile_in = 0;
    reg [GROUP_WIDTH-1:0] group_in = 0;
    reg [M_LANES*N_LANES*8-1:0] values_packed = 0;
    wire activation_load_valid;
    wire [GROUP_WIDTH-1:0] activation_load_group;
    wire [K_TILE_WIDTH-1:0] activation_load_k_tile;
    wire [M_LANES*32*8-1:0] activation_load_data;
    wire done;
    integer output_count = 0;
    integer output_tile;
    integer group;
    integer token;
    integer channel;
    integer expected;

    mlp_interstage_tile_bridge #(
        .TOKENS(TOKENS),
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .INPUT_SIZE(INPUT_SIZE),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .GROUP_WIDTH(GROUP_WIDTH),
        .K_TILE_WIDTH(K_TILE_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .output_tile_in(output_tile_in),
        .group_in(group_in),
        .values_packed(values_packed),
        .activation_load_valid(activation_load_valid),
        .activation_load_group(activation_load_group),
        .activation_load_k_tile(activation_load_k_tile),
        .activation_load_data(activation_load_data),
        .done(done)
    );

    always #2 clk = ~clk;

    always @(posedge clk) begin
        #1;
        if (activation_load_valid) begin
            if (activation_load_group !== (output_count % 2))
                $fatal(1, "group order mismatch");
            if (activation_load_k_tile !== (output_count / 2))
                $fatal(1, "K tile order mismatch");
            for (token = 0; token < M_LANES; token = token + 1) begin
                for (channel = 0; channel < 32; channel = channel + 1) begin
                    expected = activation_load_group*10 + token
                        + activation_load_k_tile*32 + channel;
                    if ($signed(activation_load_data[
                        (token*32 + channel)*8 +: 8
                    ]) !== expected) begin
                        $fatal(1, "data mismatch group=%0d k=%0d token=%0d channel=%0d",
                            activation_load_group, activation_load_k_tile,
                            token, channel);
                    end
                end
            end
            output_count = output_count + 1;
        end
    end

    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        for (output_tile = 0; output_tile <= 10;
             output_tile = output_tile + 1) begin
            for (group = 0; group < 2; group = group + 1) begin
                @(negedge clk);
                output_tile_in = output_tile;
                group_in = group;
                for (token = 0; token < M_LANES; token = token + 1) begin
                    for (channel = 0; channel < N_LANES;
                         channel = channel + 1) begin
                        values_packed[
                            (token*N_LANES + channel)*8 +: 8
                        ] = group*10 + token + output_tile*N_LANES + channel;
                    end
                end
                valid_in = 1'b1;
            end
        end
        @(negedge clk);
        valid_in = 1'b0;
        repeat (3) @(posedge clk);
        #1;
        if (output_count != 4) $fatal(1, "expected four completed tiles");
        if (done) $fatal(1, "partial test must not assert full-tensor done");
        $display("tb_mlp_interstage_tile_bridge: PASS");
        $finish;
    end
endmodule
