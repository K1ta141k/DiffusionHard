`timescale 1ns/1ps

module mlp_up_pingpong_pipeline #(
    parameter integer TOKENS = 64,
    parameter integer INPUT_SIZE = 768,
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer TOKEN_FACTOR_WIDTH = 16,
    parameter integer OUTPUT_FACTOR_WIDTH = 18,
    parameter integer FACTOR_SHIFT = 8,
    parameter integer OUTPUT_WIDTH = 16,
    parameter integer RIGHT_SHIFT = 20,
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10,
    parameter integer INTERNAL_MAC = 1,
    parameter integer POSTPROCESS_PARALLEL4 = 0,
    parameter integer GROUP_WIDTH = ((TOKENS / M_LANES) <= 1)
        ? 1 : $clog2(TOKENS / M_LANES),
    parameter integer K_TILE_WIDTH = ((INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(INPUT_SIZE / 32)
) (
    input  wire clk,
    input  wire rst_n,

    input  wire activation_load_valid,
    input  wire [GROUP_WIDTH-1:0] activation_load_group,
    input  wire [K_TILE_WIDTH-1:0] activation_load_k_tile,
    input  wire [M_LANES*32*DATA_WIDTH-1:0] activation_load_data,

    input  wire weight_load_valid,
    input  wire weight_load_bank,
    input  wire [K_TILE_WIDTH-1:0] weight_load_k_tile,
    input  wire [N_LANES*32*DATA_WIDTH-1:0] weight_load_data,
    output wire weight_load_ready,

    input  wire metadata_load_valid,
    input  wire metadata_load_bank,
    input  wire [N_LANES*OUTPUT_FACTOR_WIDTH-1:0]
        metadata_load_output_factors,
    input  wire [N_LANES*ACC_WIDTH-1:0] metadata_load_biases,
    input  wire [N_LANES*MULTIPLIER_WIDTH-1:0]
        metadata_load_interstage_multipliers,
    output wire metadata_load_ready,

    input  wire token_factor_load_valid,
    input  wire [GROUP_WIDTH-1:0] token_factor_load_group,
    input  wire [M_LANES*TOKEN_FACTOR_WIDTH-1:0]
        token_factor_load_factors,
    output wire token_factor_load_ready,

    input  wire start,
    input  wire start_bank,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] start_output_tile,
    output wire start_ready,
    output wire busy,

    output wire valid_out,
    output wire bank_out,
    output wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile_out,
    output wire [GROUP_WIDTH-1:0] group_out,
    output wire [M_LANES*N_LANES*OUTPUT_WIDTH-1:0] gelu_packed,
    output wire [N_LANES*MULTIPLIER_WIDTH-1:0]
        interstage_multipliers_out,
    output wire array_request_valid,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [OUTPUT_TILE_TAG_WIDTH+GROUP_WIDTH:0] array_request_tag,
    output wire [M_LANES*32*DATA_WIDTH-1:0] array_request_activations,
    output wire [N_LANES*32*DATA_WIDTH-1:0] array_request_weights,
    input  wire array_response_valid,
    input  wire [OUTPUT_TILE_TAG_WIDTH+GROUP_WIDTH:0] array_response_tag,
    input  wire [M_LANES*N_LANES*ACC_WIDTH-1:0]
        array_response_accumulators,
    output wire done
);

    localparam integer LANES = M_LANES * N_LANES;
    localparam integer POST_TAG_WIDTH = 1 + OUTPUT_TILE_TAG_WIDTH + GROUP_WIDTH;
    localparam integer TOKEN_GROUPS = TOKENS / M_LANES;

    wire raw_valid;
    wire raw_bank;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] raw_output_tile;
    wire [GROUP_WIDTH-1:0] raw_group;
    wire [LANES*ACC_WIDTH-1:0] raw_accumulators;
    wire raw_done;
    wire active_bank_unused;
    wire bank_0_load_ready;
    wire bank_1_load_ready;
    wire [POST_TAG_WIDTH-1:0] post_tag;
    wire [M_LANES*TOKEN_FACTOR_WIDTH-1:0] selected_token_factors;
    wire [N_LANES*OUTPUT_FACTOR_WIDTH-1:0] selected_output_factors;
    wire [N_LANES*ACC_WIDTH-1:0] selected_biases;
    wire [N_LANES*MULTIPLIER_WIDTH-1:0]
        selected_interstage_multipliers;
    wire postprocess_ready;

    reg [M_LANES*TOKEN_FACTOR_WIDTH-1:0] token_factor_groups
        [0:TOKEN_GROUPS-1];
    reg [N_LANES*OUTPUT_FACTOR_WIDTH-1:0] output_factor_bank_0;
    reg [N_LANES*OUTPUT_FACTOR_WIDTH-1:0] output_factor_bank_1;
    reg [N_LANES*ACC_WIDTH-1:0] bias_bank_0;
    reg [N_LANES*ACC_WIDTH-1:0] bias_bank_1;
    reg [N_LANES*MULTIPLIER_WIDTH-1:0] interstage_bank_0;
    reg [N_LANES*MULTIPLIER_WIDTH-1:0] interstage_bank_1;
    integer metadata_group_index;

    assign metadata_load_ready = metadata_load_bank
        ? bank_1_load_ready : bank_0_load_ready;
    assign token_factor_load_ready = !busy;
    assign selected_token_factors = token_factor_groups[raw_group];
    assign selected_output_factors = raw_bank
        ? output_factor_bank_1 : output_factor_bank_0;
    assign selected_biases = raw_bank ? bias_bank_1 : bias_bank_0;
    assign selected_interstage_multipliers = raw_bank
        ? interstage_bank_1 : interstage_bank_0;
    assign bank_out = post_tag[POST_TAG_WIDTH-1];
    assign output_tile_out = post_tag[
        GROUP_WIDTH +: OUTPUT_TILE_TAG_WIDTH
    ];
    assign group_out = post_tag[GROUP_WIDTH-1:0];
    assign done = valid_out && (group_out == TOKEN_GROUPS-1);

    initial begin
        if ((INPUT_SIZE / 32) < (POSTPROCESS_PARALLEL4
            ? ((LANES + 3) / 4) : LANES)) begin
            $error("MLP up postprocess cannot drain before the next MAC result");
        end
    end

    mlp_tile_pingpong_controller #(
        .TOKENS(TOKENS),
        .INPUT_SIZE(INPUT_SIZE),
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .INTERNAL_MAC(INTERNAL_MAC),
        .GROUP_WIDTH(GROUP_WIDTH),
        .K_TILE_WIDTH(K_TILE_WIDTH)
    ) controller (
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
        .bank_0_load_ready(bank_0_load_ready),
        .bank_1_load_ready(bank_1_load_ready),
        .start(start),
        .start_bank(start_bank),
        .start_output_tile(start_output_tile),
        .start_ready(start_ready),
        .busy(busy),
        .active_bank(active_bank_unused),
        .result_valid(raw_valid),
        .result_bank(raw_bank),
        .result_output_tile(raw_output_tile),
        .result_group(raw_group),
        .result_accumulators(raw_accumulators),
        .array_request_valid(array_request_valid),
        .array_request_clear(array_request_clear),
        .array_request_last(array_request_last),
        .array_request_tag(array_request_tag),
        .array_request_activations(array_request_activations),
        .array_request_weights(array_request_weights),
        .array_response_valid(array_response_valid),
        .array_response_tag(array_response_tag),
        .array_response_accumulators(array_response_accumulators),
        .done(raw_done)
    );

    generate
        if (POSTPROCESS_PARALLEL4) begin : parallel4_postprocess
            mlp_up_postprocess_parallel4 #(
                .M_LANES(M_LANES), .N_LANES(N_LANES),
                .ACC_WIDTH(ACC_WIDTH),
                .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
                .TOKEN_FACTOR_WIDTH(TOKEN_FACTOR_WIDTH),
                .OUTPUT_FACTOR_WIDTH(OUTPUT_FACTOR_WIDTH),
                .FACTOR_SHIFT(FACTOR_SHIFT), .DATA_WIDTH(OUTPUT_WIDTH),
                .RIGHT_SHIFT(RIGHT_SHIFT), .TAG_WIDTH(POST_TAG_WIDTH)
            ) postprocess (
                .clk(clk), .rst_n(rst_n), .valid_in(raw_valid),
                .ready_in(postprocess_ready),
                .tag_in({raw_bank, raw_output_tile, raw_group}),
                .accumulators_packed(raw_accumulators),
                .token_factors_packed(selected_token_factors),
                .output_factors_packed(selected_output_factors),
                .biases_packed(selected_biases),
                .sideband_in(selected_interstage_multipliers),
                .valid_out(valid_out), .tag_out(post_tag),
                .gelu_packed(gelu_packed),
                .sideband_out(interstage_multipliers_out)
            );
        end else begin : serial_postprocess
            mlp_up_postprocess_serial #(
                .M_LANES(M_LANES), .N_LANES(N_LANES),
                .ACC_WIDTH(ACC_WIDTH),
                .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
                .TOKEN_FACTOR_WIDTH(TOKEN_FACTOR_WIDTH),
                .OUTPUT_FACTOR_WIDTH(OUTPUT_FACTOR_WIDTH),
                .FACTOR_SHIFT(FACTOR_SHIFT), .DATA_WIDTH(OUTPUT_WIDTH),
                .RIGHT_SHIFT(RIGHT_SHIFT), .TAG_WIDTH(POST_TAG_WIDTH)
            ) postprocess (
                .clk(clk), .rst_n(rst_n), .valid_in(raw_valid),
                .ready_in(postprocess_ready),
                .tag_in({raw_bank, raw_output_tile, raw_group}),
                .accumulators_packed(raw_accumulators),
                .token_factors_packed(selected_token_factors),
                .output_factors_packed(selected_output_factors),
                .biases_packed(selected_biases),
                .sideband_in(selected_interstage_multipliers),
                .valid_out(valid_out), .tag_out(post_tag),
                .gelu_packed(gelu_packed),
                .sideband_out(interstage_multipliers_out)
            );
        end
    endgenerate

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && raw_valid && !postprocess_ready) begin
            $error("serialized postprocess input queue overflow");
        end
`endif
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            for (metadata_group_index = 0;
                 metadata_group_index < TOKEN_GROUPS;
                 metadata_group_index = metadata_group_index + 1) begin
                token_factor_groups[metadata_group_index] <=
                    {M_LANES*TOKEN_FACTOR_WIDTH{1'b0}};
            end
            output_factor_bank_0 <= {N_LANES*OUTPUT_FACTOR_WIDTH{1'b0}};
            output_factor_bank_1 <= {N_LANES*OUTPUT_FACTOR_WIDTH{1'b0}};
            bias_bank_0 <= {N_LANES*ACC_WIDTH{1'b0}};
            bias_bank_1 <= {N_LANES*ACC_WIDTH{1'b0}};
            interstage_bank_0 <= {N_LANES*MULTIPLIER_WIDTH{1'b0}};
            interstage_bank_1 <= {N_LANES*MULTIPLIER_WIDTH{1'b0}};
        end else begin
            if (metadata_load_valid && metadata_load_ready) begin
                if (metadata_load_bank) begin
                    output_factor_bank_1 <= metadata_load_output_factors;
                    bias_bank_1 <= metadata_load_biases;
                    interstage_bank_1 <= metadata_load_interstage_multipliers;
                end else begin
                    output_factor_bank_0 <= metadata_load_output_factors;
                    bias_bank_0 <= metadata_load_biases;
                    interstage_bank_0 <= metadata_load_interstage_multipliers;
                end
            end
            if (token_factor_load_valid && token_factor_load_ready) begin
                token_factor_groups[token_factor_load_group] <=
                    token_factor_load_factors;
            end
        end
    end

endmodule
