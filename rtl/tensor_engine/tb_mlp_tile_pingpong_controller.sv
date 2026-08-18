`timescale 1ns/1ps

module tb_mlp_tile_pingpong_controller;
    parameter integer INTERNAL_MAC = 1;
    parameter integer SYNC_ACTIVATION_MEMORY = 0;
    localparam integer TOKENS = 8;
    localparam integer INPUT_SIZE = 64;
    localparam integer M_LANES = 2;
    localparam integer N_LANES = 2;
    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH = 32;
    localparam integer OUTPUT_TILE_TAG_WIDTH = 4;
    localparam integer GROUPS = TOKENS / M_LANES;
    localparam integer K_TILES = INPUT_SIZE / 32;
    localparam integer GROUP_WIDTH = $clog2(GROUPS);
    localparam integer K_TILE_WIDTH = $clog2(K_TILES);
    localparam integer TAG_WIDTH = 1 + OUTPUT_TILE_TAG_WIDTH + GROUP_WIDTH;

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
    reg start = 1'b0;
    reg start_bank = 1'b0;
    reg [OUTPUT_TILE_TAG_WIDTH-1:0] start_output_tile = '0;
    wire start_ready;
    wire busy;
    wire active_bank;
    wire result_valid;
    wire result_bank;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] result_output_tile;
    wire [GROUP_WIDTH-1:0] result_group;
    wire [M_LANES*N_LANES*ACC_WIDTH-1:0] result_accumulators;
    wire array_request_valid;
    wire array_request_clear;
    wire array_request_last;
    wire [TAG_WIDTH-1:0] array_request_tag;
    wire [M_LANES*32*DATA_WIDTH-1:0] array_request_activations;
    wire [N_LANES*32*DATA_WIDTH-1:0] array_request_weights;
    wire array_response_valid;
    wire [TAG_WIDTH-1:0] array_response_tag;
    wire [M_LANES*N_LANES*ACC_WIDTH-1:0] array_response_accumulators;
    wire done;

    integer activation_values [0:GROUPS-1][0:M_LANES-1][0:INPUT_SIZE-1];
    integer weight_values [0:1][0:N_LANES-1][0:INPUT_SIZE-1];
    integer expected [0:1][0:GROUPS-1][0:M_LANES-1][0:N_LANES-1];
    integer bank_index;
    integer group_index;
    integer tile_index;
    integer m_index;
    integer n_index;
    integer k_index;
    integer output_count = 0;
    integer done_count = 0;
    integer expected_bank;
    integer expected_group;
    reg signed [DATA_WIDTH-1:0] data_value;
    reg signed [ACC_WIDTH-1:0] actual_value;

    mlp_tile_pingpong_controller #(
        .TOKENS(TOKENS),
        .INPUT_SIZE(INPUT_SIZE),
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .INTERNAL_MAC(INTERNAL_MAC),
        .SYNC_ACTIVATION_MEMORY(SYNC_ACTIVATION_MEMORY)
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
        .start(start),
        .start_bank(start_bank),
        .start_output_tile(start_output_tile),
        .start_ready(start_ready),
        .busy(busy),
        .active_bank(active_bank),
        .result_valid(result_valid),
        .result_bank(result_bank),
        .result_output_tile(result_output_tile),
        .result_group(result_group),
        .result_accumulators(result_accumulators),
        .array_request_valid(array_request_valid),
        .array_request_clear(array_request_clear),
        .array_request_last(array_request_last),
        .array_request_tag(array_request_tag),
        .array_request_activations(array_request_activations),
        .array_request_weights(array_request_weights),
        .array_response_valid(array_response_valid),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators),
        .done(done)
    );

    generate
        if (!INTERNAL_MAC) begin : external_array
            int8_mac_tile_pipelined #(
                .M_LANES(M_LANES), .N_LANES(N_LANES),
                .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
                .TAG_WIDTH(TAG_WIDTH)
            ) mac (
                .clk(clk), .rst_n(rst_n), .valid_in(array_request_valid),
                .clear_accumulators(array_request_clear),
                .last_k_tile(array_request_last), .tag_in(array_request_tag),
                .activations_packed(array_request_activations),
                .weights_packed(array_request_weights),
                .valid_out(array_response_valid), .tag_out(array_response_tag),
                .accumulators_packed(array_response_accumulators)
            );
        end else begin : no_external_array
            assign array_response_valid = 1'b0;
            assign array_response_tag = {TAG_WIDTH{1'b0}};
            assign array_response_accumulators =
                {M_LANES*N_LANES*ACC_WIDTH{1'b0}};
        end
    endgenerate

    always #5 clk = ~clk;

    task load_weight_tile;
        input integer bank;
        input integer tile;
        begin
            @(negedge clk);
            weight_load_data = '0;
            for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                for (k_index = 0; k_index < 32; k_index = k_index + 1) begin
                    data_value = weight_values[bank][n_index][tile*32+k_index];
                    weight_load_data[(n_index*32+k_index)*DATA_WIDTH +: DATA_WIDTH] = data_value;
                end
            end
            weight_load_bank = bank;
            weight_load_k_tile = tile;
            weight_load_valid = 1'b1;
            #1;
            if (!weight_load_ready) $fatal(1, "weight load unexpectedly blocked");
            @(posedge clk);
            #1;
        end
    endtask

    task launch;
        input integer bank;
        input integer output_tile;
        begin
            @(negedge clk);
            weight_load_valid = 1'b0;
            start_bank = bank;
            start_output_tile = output_tile;
            start = 1'b1;
            #1;
            if (!start_ready) $fatal(1, "start unexpectedly blocked");
            @(posedge clk);
            #1;
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    always @(posedge clk) begin
        #1;
        if (result_valid) begin
            expected_bank = output_count / GROUPS;
            expected_group = output_count % GROUPS;
            if (result_bank !== expected_bank[0]) begin
                $fatal(1, "expected bank %0d got %0d", expected_bank, result_bank);
            end
            if (result_group !== expected_group[GROUP_WIDTH-1:0]) begin
                $fatal(1, "expected group %0d got %0d", expected_group, result_group);
            end
            if (result_output_tile !== (expected_bank ? 4 : 3)) begin
                $fatal(1, "unexpected output tile tag");
            end
            for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
                for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                    actual_value = $signed(
                        result_accumulators[(m_index*N_LANES+n_index)*ACC_WIDTH +: ACC_WIDTH]
                    );
                    if (actual_value !== expected[expected_bank][expected_group][m_index][n_index]) begin
                        $fatal(
                            1,
                            "bank %0d group %0d accumulator[%0d,%0d] expected %0d got %0d",
                            expected_bank,
                            expected_group,
                            m_index,
                            n_index,
                            expected[expected_bank][expected_group][m_index][n_index],
                            actual_value
                        );
                    end
                end
            end
            output_count = output_count + 1;
        end
        if (done) done_count = done_count + 1;
    end

    initial begin
        for (group_index = 0; group_index < GROUPS; group_index = group_index + 1) begin
            for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
                for (k_index = 0; k_index < INPUT_SIZE; k_index = k_index + 1) begin
                    activation_values[group_index][m_index][k_index] =
                        ((group_index+1)*5 + m_index*3 + k_index) % 15 - 7;
                end
            end
        end
        for (bank_index = 0; bank_index < 2; bank_index = bank_index + 1) begin
            for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                for (k_index = 0; k_index < INPUT_SIZE; k_index = k_index + 1) begin
                    weight_values[bank_index][n_index][k_index] =
                        ((bank_index+1)*4 + n_index*7 + k_index*3) % 13 - 6;
                end
            end
            for (group_index = 0; group_index < GROUPS; group_index = group_index + 1) begin
                for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
                    for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                        expected[bank_index][group_index][m_index][n_index] = 0;
                        for (k_index = 0; k_index < INPUT_SIZE; k_index = k_index + 1) begin
                            expected[bank_index][group_index][m_index][n_index] =
                                expected[bank_index][group_index][m_index][n_index]
                                + activation_values[group_index][m_index][k_index]
                                * weight_values[bank_index][n_index][k_index];
                        end
                    end
                end
            end
        end

        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

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

        load_weight_tile(0, 0);
        load_weight_tile(0, 1);
        launch(0, 3);

        @(negedge clk);
        weight_load_bank = 1'b0;
        weight_load_valid = 1'b1;
        #1;
        if (weight_load_ready) $fatal(1, "active bank overwrite was not blocked");
        @(posedge clk);
        #1;
        weight_load_valid = 1'b0;

        load_weight_tile(1, 0);
        load_weight_tile(1, 1);

        wait (!busy);
        launch(1, 4);
        wait (!busy);
        repeat (10) @(posedge clk);
        #1;

        if (output_count !== 2*GROUPS) begin
            $fatal(1, "expected %0d outputs, got %0d", 2*GROUPS, output_count);
        end
        if (done_count !== 2) begin
            $fatal(1, "expected two done pulses, got %0d", done_count);
        end
        $display("tb_mlp_tile_pingpong_controller: PASS");
        $finish;
    end
endmodule
