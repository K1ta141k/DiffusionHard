`timescale 1ns/1ps

module tb_mlp_down_pingpong_pipeline;
    localparam integer TOKENS = 4;
    localparam integer INPUT_SIZE = 64;
    localparam integer OUTPUT_SIZE = 4;
    localparam integer M_LANES = 1;
    localparam integer N_LANES = 2;
    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH = 32;
    localparam integer MULTIPLIER_WIDTH = 24;
    localparam integer OUTPUT_WIDTH = 24;
    localparam integer OUTPUT_TILE_TAG_WIDTH = 2;
    localparam integer GROUP_WIDTH = 2;
    localparam integer K_TILE_WIDTH = 1;
    localparam integer LANES = M_LANES * N_LANES;
    localparam integer K_TILES = INPUT_SIZE / 32;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg activation_load_valid = 1'b0;
    reg [GROUP_WIDTH-1:0] activation_load_group = '0;
    reg [K_TILE_WIDTH-1:0] activation_load_k_tile = '0;
    reg [M_LANES*32*DATA_WIDTH-1:0] activation_load_data = '0;
    reg weight_load_valid = 1'b0;
    reg weight_load_bank = 1'b0;
    reg [K_TILE_WIDTH-1:0] weight_load_k_tile = '0;
    reg [N_LANES*32*DATA_WIDTH-1:0] weight_load_data = '0;
    wire weight_load_ready;
    reg metadata_load_valid = 1'b0;
    reg metadata_load_bank = 1'b0;
    reg [LANES*MULTIPLIER_WIDTH-1:0] metadata_load_multipliers = '0;
    reg [LANES*ACC_WIDTH-1:0] metadata_load_biases = '0;
    wire metadata_load_ready;
    reg residual_load_valid = 1'b0;
    reg [GROUP_WIDTH-1:0] residual_load_group = '0;
    reg [OUTPUT_TILE_TAG_WIDTH-1:0] residual_load_output_tile = '0;
    reg [LANES*OUTPUT_WIDTH-1:0] residual_load_data = '0;
    wire residual_load_ready;
    reg start = 1'b0;
    reg start_bank = 1'b0;
    reg [OUTPUT_TILE_TAG_WIDTH-1:0] start_output_tile = '0;
    wire start_ready;
    wire busy;
    wire valid_out;
    wire bank_out;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile_out;
    wire [GROUP_WIDTH-1:0] group_out;
    wire [LANES*OUTPUT_WIDTH-1:0] outputs_packed;
    wire done;

    integer group_index;
    integer tile_index;
    integer output_count = 0;
    integer done_count = 0;
    integer expected_bank;
    integer expected_group;
    reg signed [OUTPUT_WIDTH-1:0] actual;

    mlp_down_pingpong_pipeline #(
        .TOKENS(TOKENS),
        .INPUT_SIZE(INPUT_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .GROUP_WIDTH(GROUP_WIDTH),
        .K_TILE_WIDTH(K_TILE_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .activation_load_valid(activation_load_valid),
        .activation_load_group(activation_load_group),
        .activation_load_k_tile(activation_load_k_tile),
        .activation_load_data(activation_load_data),
        .weight_load_valid(weight_load_valid),
        .weight_load_bank(weight_load_bank),
        .weight_load_k_tile(weight_load_k_tile),
        .weight_load_data(weight_load_data),
        .weight_load_ready(weight_load_ready),
        .metadata_load_valid(metadata_load_valid),
        .metadata_load_bank(metadata_load_bank),
        .metadata_load_multipliers(metadata_load_multipliers),
        .metadata_load_biases(metadata_load_biases),
        .metadata_load_ready(metadata_load_ready),
        .residual_load_valid(residual_load_valid),
        .residual_load_group(residual_load_group),
        .residual_load_output_tile(residual_load_output_tile),
        .residual_load_data(residual_load_data),
        .residual_load_ready(residual_load_ready),
        .start(start),
        .start_bank(start_bank),
        .start_output_tile(start_output_tile),
        .start_ready(start_ready),
        .busy(busy),
        .valid_out(valid_out),
        .bank_out(bank_out),
        .output_tile_out(output_tile_out),
        .group_out(group_out),
        .outputs_packed(outputs_packed),
        .done(done)
    );

    always #5 clk = ~clk;

    task load_bank;
        input integer bank;
        input signed [ACC_WIDTH-1:0] bias_0;
        input signed [ACC_WIDTH-1:0] bias_1;
        begin
            for (tile_index = 0; tile_index < K_TILES; tile_index = tile_index + 1) begin
                @(negedge clk);
                weight_load_bank = bank;
                weight_load_k_tile = tile_index;
                weight_load_data = '0;
                weight_load_valid = 1'b1;
                #1;
                if (!weight_load_ready) $fatal(1, "weight bank blocked");
                @(posedge clk);
                #1;
            end
            @(negedge clk);
            weight_load_valid = 1'b0;
            metadata_load_bank = bank;
            metadata_load_multipliers = {LANES{24'h100000}};
            metadata_load_biases[0*ACC_WIDTH +: ACC_WIDTH] = bias_0;
            metadata_load_biases[1*ACC_WIDTH +: ACC_WIDTH] = bias_1;
            metadata_load_valid = 1'b1;
            #1;
            if (!metadata_load_ready) $fatal(1, "metadata bank blocked");
            @(posedge clk);
            #1;
            @(negedge clk);
            metadata_load_valid = 1'b0;
        end
    endtask

    task launch;
        input integer bank;
        input integer output_tile;
        begin
            @(negedge clk);
            start_bank = bank;
            start_output_tile = output_tile;
            start = 1'b1;
            #1;
            if (!start_ready) $fatal(1, "start blocked");
            @(posedge clk);
            #1;
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    always @(posedge clk) begin
        #1;
        if (valid_out) begin
            expected_bank = output_count / TOKENS;
            expected_group = output_count % TOKENS;
            if (bank_out !== expected_bank[0]) $fatal(1, "bank tag mismatch");
            if (output_tile_out !== expected_bank[OUTPUT_TILE_TAG_WIDTH-1:0]) begin
                $fatal(1, "output tile tag mismatch");
            end
            if (group_out !== expected_group[GROUP_WIDTH-1:0]) begin
                $fatal(1, "group tag mismatch");
            end
            actual = $signed(outputs_packed[0*OUTPUT_WIDTH +: OUTPUT_WIDTH]);
            if (actual !== (expected_bank ? 900+expected_group : 110+expected_group)) begin
                $fatal(1, "lane 0 result mismatch: %0d", actual);
            end
            actual = $signed(outputs_packed[1*OUTPUT_WIDTH +: OUTPUT_WIDTH]);
            if (actual !== (expected_bank ? 200-expected_group : 220+expected_group)) begin
                $fatal(1, "lane 1 result mismatch: %0d", actual);
            end
            output_count = output_count + 1;
        end
        if (done) done_count = done_count + 1;
    end

    initial begin
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        for (group_index = 0; group_index < TOKENS; group_index = group_index + 1) begin
            for (tile_index = 0; tile_index < K_TILES; tile_index = tile_index + 1) begin
                @(negedge clk);
                activation_load_group = group_index;
                activation_load_k_tile = tile_index;
                activation_load_data = '0;
                activation_load_valid = 1'b1;
                @(posedge clk);
                #1;
            end
        end
        @(negedge clk);
        activation_load_valid = 1'b0;

        for (group_index = 0; group_index < TOKENS; group_index = group_index + 1) begin
            @(negedge clk);
            residual_load_group = group_index;
            residual_load_output_tile = 0;
            residual_load_data[0*OUTPUT_WIDTH +: OUTPUT_WIDTH] = 10 + group_index;
            residual_load_data[1*OUTPUT_WIDTH +: OUTPUT_WIDTH] = 20 + group_index;
            residual_load_valid = 1'b1;
            @(posedge clk);
            #1;
            @(negedge clk);
            residual_load_output_tile = 1;
            residual_load_data[0*OUTPUT_WIDTH +: OUTPUT_WIDTH] = 1000 + group_index;
            residual_load_data[1*OUTPUT_WIDTH +: OUTPUT_WIDTH] = -100 - group_index;
            @(posedge clk);
            #1;
        end
        @(negedge clk);
        residual_load_valid = 1'b0;

        load_bank(0, 100, 200);
        launch(0, 0);
        load_bank(1, -100, 300);
        wait (!busy);
        launch(1, 1);
        wait (!busy);
        repeat (18) @(posedge clk);
        #1;

        if (output_count !== 2*TOKENS) begin
            $fatal(1, "expected %0d outputs, got %0d", 2*TOKENS, output_count);
        end
        if (done_count !== 2) begin
            $fatal(1, "expected two done pulses, got %0d", done_count);
        end
        $display("tb_mlp_down_pingpong_pipeline: PASS");
        $finish;
    end
endmodule
