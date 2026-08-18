`timescale 1ns/1ps

module tb_mlp_interstage_pipeline;
    localparam integer TOKENS = 8;
    localparam integer M_LANES = 4;
    localparam integer N_LANES = 6;
    localparam integer INPUT_SIZE = 96;
    localparam integer LANES = M_LANES * N_LANES;
    localparam integer GROUP_WIDTH = 1;
    localparam integer K_TILE_WIDTH = 2;
    localparam integer OUTPUT_TILE_TAG_WIDTH = 5;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    wire ready_in;
    reg [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile_in = 0;
    reg [GROUP_WIDTH-1:0] group_in = 0;
    reg [LANES*16-1:0] gelu_q10_packed = 0;
    reg [N_LANES*24-1:0] multipliers_packed = 0;
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
    integer lane;
    integer expected;

    mlp_interstage_pipeline #(
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
        .ready_in(ready_in),
        .output_tile_in(output_tile_in),
        .group_in(group_in),
        .gelu_q10_packed(gelu_q10_packed),
        .multipliers_packed(multipliers_packed),
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
                $fatal(1, "connected group order mismatch");
            if (activation_load_k_tile !== (output_count / 2))
                $fatal(1, "connected K tile order mismatch");
            for (token = 0; token < M_LANES; token = token + 1) begin
                for (channel = 0; channel < 32; channel = channel + 1) begin
                    expected = activation_load_group*10 + token
                        + activation_load_k_tile*32 + channel;
                    if ($signed(activation_load_data[
                        (token*32 + channel)*8 +: 8
                    ]) !== expected) begin
                        $fatal(1, "connected data mismatch group=%0d k=%0d token=%0d channel=%0d",
                            activation_load_group, activation_load_k_tile,
                            token, channel);
                    end
                end
            end
            output_count = output_count + 1;
        end
    end

    task send_vector;
        input integer tile_value;
        input integer group_value;
        begin
            while (!ready_in) @(negedge clk);
            output_tile_in = tile_value;
            group_in = group_value;
            for (token = 0; token < M_LANES; token = token + 1) begin
                for (channel = 0; channel < N_LANES;
                     channel = channel + 1) begin
                    lane = token*N_LANES + channel;
                    gelu_q10_packed[lane*16 +: 16] = 16 * (
                        group_value*10 + token + tile_value*N_LANES + channel
                    );
                    multipliers_packed[channel*24 +: 24] = 24'd65536;
                end
            end
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        for (output_tile = 0; output_tile <= 10;
             output_tile = output_tile + 1) begin
            for (group = 0; group < 2; group = group + 1) begin
                send_vector(output_tile, group);
            end
        end
        wait (output_count == 4);
        repeat (2) @(posedge clk);
        if (done) $fatal(1, "partial connected test must not assert done");
        $display("tb_mlp_interstage_pipeline: PASS");
        $finish;
    end
endmodule
