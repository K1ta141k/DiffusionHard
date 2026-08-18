`timescale 1ns/1ps

module tb_mlp_tile_controller;
    localparam integer TOKENS = 8;
    localparam integer INPUT_SIZE = 64;
    localparam integer M_LANES = 2;
    localparam integer N_LANES = 2;
    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH = 32;
    localparam integer GROUPS = TOKENS / M_LANES;
    localparam integer K_TILES = INPUT_SIZE / 32;
    localparam integer GROUP_WIDTH = $clog2(GROUPS);
    localparam integer K_TILE_WIDTH = $clog2(K_TILES);

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg activation_load_valid = 1'b0;
    reg [GROUP_WIDTH-1:0] activation_load_group = '0;
    reg [K_TILE_WIDTH-1:0] activation_load_k_tile = '0;
    reg [M_LANES*32*DATA_WIDTH-1:0] activation_load_data = '0;
    reg weight_load_valid = 1'b0;
    reg [K_TILE_WIDTH-1:0] weight_load_k_tile = '0;
    reg [N_LANES*32*DATA_WIDTH-1:0] weight_load_data = '0;
    reg start = 1'b0;
    wire busy;
    wire result_valid;
    wire [GROUP_WIDTH-1:0] result_group;
    wire [M_LANES*N_LANES*ACC_WIDTH-1:0] result_accumulators;
    wire done;

    integer expected [0:GROUPS-1][0:M_LANES-1][0:N_LANES-1];
    integer activation_values [0:GROUPS-1][0:M_LANES-1][0:INPUT_SIZE-1];
    integer weight_values [0:N_LANES-1][0:INPUT_SIZE-1];
    integer group_index;
    integer tile_index;
    integer m_index;
    integer n_index;
    integer k_index;
    integer observed_groups = 0;
    integer done_pulses = 0;
    reg signed [DATA_WIDTH-1:0] data_value;
    reg signed [ACC_WIDTH-1:0] actual_value;

    mlp_tile_controller #(
        .TOKENS(TOKENS),
        .INPUT_SIZE(INPUT_SIZE),
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .activation_load_valid(activation_load_valid),
        .activation_load_group(activation_load_group),
        .activation_load_k_tile(activation_load_k_tile),
        .activation_load_data(activation_load_data),
        .weight_load_valid(weight_load_valid),
        .weight_load_k_tile(weight_load_k_tile),
        .weight_load_data(weight_load_data),
        .start(start),
        .busy(busy),
        .result_valid(result_valid),
        .result_group(result_group),
        .result_accumulators(result_accumulators),
        .done(done)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        #1;
        if (result_valid) begin
            if (result_group !== observed_groups[GROUP_WIDTH-1:0]) begin
                $fatal(1, "expected result group %0d got %0d", observed_groups, result_group);
            end
            for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
                for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                    actual_value = $signed(
                        result_accumulators[(m_index*N_LANES+n_index)*ACC_WIDTH +: ACC_WIDTH]
                    );
                    if (actual_value !== expected[observed_groups][m_index][n_index]) begin
                        $fatal(
                            1,
                            "group %0d accumulator[%0d,%0d] expected %0d got %0d",
                            observed_groups,
                            m_index,
                            n_index,
                            expected[observed_groups][m_index][n_index],
                            actual_value
                        );
                    end
                end
            end
            observed_groups = observed_groups + 1;
        end
        if (done) begin
            done_pulses = done_pulses + 1;
        end
    end

    initial begin
        for (group_index = 0; group_index < GROUPS; group_index = group_index + 1) begin
            for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
                for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                    expected[group_index][m_index][n_index] = 0;
                end
                for (k_index = 0; k_index < INPUT_SIZE; k_index = k_index + 1) begin
                    activation_values[group_index][m_index][k_index] =
                        ((group_index+1)*5 + m_index*3 + k_index) % 15 - 7;
                end
            end
        end
        for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
            for (k_index = 0; k_index < INPUT_SIZE; k_index = k_index + 1) begin
                weight_values[n_index][k_index] = (n_index*7 + k_index*3) % 13 - 6;
            end
        end
        for (group_index = 0; group_index < GROUPS; group_index = group_index + 1) begin
            for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
                for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                    for (k_index = 0; k_index < INPUT_SIZE; k_index = k_index + 1) begin
                        expected[group_index][m_index][n_index] =
                            expected[group_index][m_index][n_index]
                            + activation_values[group_index][m_index][k_index]
                            * weight_values[n_index][k_index];
                    end
                end
            end
        end

        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        for (tile_index = 0; tile_index < K_TILES; tile_index = tile_index + 1) begin
            @(negedge clk);
            weight_load_data = '0;
            for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                for (k_index = 0; k_index < 32; k_index = k_index + 1) begin
                    data_value = weight_values[n_index][tile_index*32+k_index];
                    weight_load_data[(n_index*32+k_index)*DATA_WIDTH +: DATA_WIDTH] = data_value;
                end
            end
            weight_load_k_tile = tile_index;
            weight_load_valid = 1'b1;
            @(posedge clk);
            #1;
        end
        @(negedge clk);
        weight_load_valid = 1'b0;

        for (group_index = 0; group_index < GROUPS; group_index = group_index + 1) begin
            for (tile_index = 0; tile_index < K_TILES; tile_index = tile_index + 1) begin
                @(negedge clk);
                activation_load_data = '0;
                for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
                    for (k_index = 0; k_index < 32; k_index = k_index + 1) begin
                        data_value = activation_values[group_index][m_index][tile_index*32+k_index];
                        activation_load_data[(m_index*32+k_index)*DATA_WIDTH +: DATA_WIDTH] = data_value;
                    end
                end
                activation_load_group = group_index;
                activation_load_k_tile = tile_index;
                activation_load_valid = 1'b1;
                @(posedge clk);
                #1;
            end
        end
        @(negedge clk);
        activation_load_valid = 1'b0;
        start = 1'b1;
        @(posedge clk);
        #1;
        @(negedge clk);
        start = 1'b0;

        repeat (24) @(posedge clk);
        #1;
        if (observed_groups !== GROUPS) begin
            $fatal(1, "expected %0d result groups, got %0d", GROUPS, observed_groups);
        end
        if (done_pulses !== 1) begin
            $fatal(1, "expected one done pulse, got %0d", done_pulses);
        end
        if (busy !== 1'b0) begin
            $fatal(1, "controller remained busy");
        end

        $display("tb_mlp_tile_controller: PASS");
        $finish;
    end
endmodule
