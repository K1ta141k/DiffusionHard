`timescale 1ns/1ps

module mlp_down_pingpong_pipeline #(
    parameter integer TOKENS = 64,
    parameter integer INPUT_SIZE = 3072,
    parameter integer OUTPUT_SIZE = 768,
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer OUTPUT_WIDTH = 24,
    parameter integer RIGHT_SHIFT = 20,
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10,
    parameter integer INTERNAL_MAC = 1,
    parameter integer SYNC_ACTIVATION_MEMORY = 0,
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
    input  wire [M_LANES*N_LANES*MULTIPLIER_WIDTH-1:0]
        metadata_load_multipliers,
    input  wire [M_LANES*N_LANES*ACC_WIDTH-1:0] metadata_load_biases,
    output wire metadata_load_ready,

    input  wire residual_load_valid,
    input  wire [GROUP_WIDTH-1:0] residual_load_group,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] residual_load_output_tile,
    input  wire [M_LANES*N_LANES*OUTPUT_WIDTH-1:0] residual_load_data,
    output wire residual_load_ready,

    input  wire start,
    input  wire start_bank,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] start_output_tile,
    output wire start_ready,
    output wire busy,

    output wire valid_out,
    output wire bank_out,
    output wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile_out,
    output wire [GROUP_WIDTH-1:0] group_out,
    output wire [M_LANES*N_LANES*OUTPUT_WIDTH-1:0] outputs_packed,
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
    localparam integer TOKEN_GROUPS = TOKENS / M_LANES;
    localparam integer OUTPUT_TILES = OUTPUT_SIZE / N_LANES;
    localparam integer POST_TAG_WIDTH = 1 + OUTPUT_TILE_TAG_WIDTH + GROUP_WIDTH;

    wire raw_valid;
    wire raw_bank;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] raw_output_tile;
    wire [GROUP_WIDTH-1:0] raw_group;
    wire [LANES*ACC_WIDTH-1:0] raw_accumulators;
    wire raw_done;
    wire active_bank_unused;
    wire bank_0_load_ready;
    wire bank_1_load_ready;
    wire postprocess_ready;
    wire postprocess_valid;
    wire [POST_TAG_WIDTH-1:0] postprocess_tag;
    wire [LANES*OUTPUT_WIDTH-1:0] postprocess_values;
    wire postprocess_bank;
    wire [OUTPUT_TILE_TAG_WIDTH-1:0] postprocess_output_tile;
    wire [GROUP_WIDTH-1:0] postprocess_group;
    wire [LANES*OUTPUT_WIDTH-1:0] selected_residuals;
    wire [POST_TAG_WIDTH-1:0] output_tag;

    reg [LANES*MULTIPLIER_WIDTH-1:0] multiplier_bank_0;
    reg [LANES*MULTIPLIER_WIDTH-1:0] multiplier_bank_1;
    reg [LANES*ACC_WIDTH-1:0] bias_bank_0;
    reg [LANES*ACC_WIDTH-1:0] bias_bank_1;
    reg [LANES*OUTPUT_WIDTH-1:0] residual_tiles_bank_0
        [0:TOKEN_GROUPS-1];
    reg [LANES*OUTPUT_WIDTH-1:0] residual_tiles_bank_1
        [0:TOKEN_GROUPS-1];

    assign metadata_load_ready = metadata_load_bank
        ? bank_1_load_ready : bank_0_load_ready;
    assign residual_load_ready = !busy;
    assign postprocess_bank = postprocess_tag[POST_TAG_WIDTH-1];
    assign postprocess_output_tile = postprocess_tag[
        GROUP_WIDTH +: OUTPUT_TILE_TAG_WIDTH
    ];
    assign postprocess_group = postprocess_tag[GROUP_WIDTH-1:0];
    assign selected_residuals = postprocess_bank
        ? residual_tiles_bank_1[postprocess_group]
        : residual_tiles_bank_0[postprocess_group];
    assign bank_out = output_tag[POST_TAG_WIDTH-1];
    assign output_tile_out = output_tag[
        GROUP_WIDTH +: OUTPUT_TILE_TAG_WIDTH
    ];
    assign group_out = output_tag[GROUP_WIDTH-1:0];
    assign done = valid_out && (group_out == TOKEN_GROUPS-1);

    initial begin
        if (OUTPUT_SIZE % N_LANES != 0) begin
            $error("OUTPUT_SIZE must be divisible by N_LANES");
        end
        if ((INPUT_SIZE / 32) < LANES) begin
            $error("serialized postprocess requires K tiles per group >= output lanes");
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
        .SYNC_ACTIVATION_MEMORY(SYNC_ACTIVATION_MEMORY),
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

    fixed_requantize_vector_serial #(
        .LANES(LANES),
        .ACC_WIDTH(ACC_WIDTH),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH),
        .RIGHT_SHIFT(RIGHT_SHIFT),
        .TAG_WIDTH(POST_TAG_WIDTH)
    ) postprocess (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(raw_valid),
        .ready_in(postprocess_ready),
        .tag_in({raw_bank, raw_output_tile, raw_group}),
        .accumulators_packed(raw_accumulators),
        .multipliers_packed(
            raw_bank ? multiplier_bank_1 : multiplier_bank_0
        ),
        .biases_packed(raw_bank ? bias_bank_1 : bias_bank_0),
        .valid_out(postprocess_valid),
        .tag_out(postprocess_tag),
        .outputs_packed(postprocess_values)
    );

    residual_add_saturating #(
        .LANES(LANES),
        .DATA_WIDTH(OUTPUT_WIDTH),
        .TAG_WIDTH(POST_TAG_WIDTH)
    ) residual_adder (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(postprocess_valid),
        .tag_in(postprocess_tag),
        .values_packed(postprocess_values),
        .residuals_packed(selected_residuals),
        .valid_out(valid_out),
        .tag_out(output_tag),
        .outputs_packed(outputs_packed)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            multiplier_bank_0 <= {LANES*MULTIPLIER_WIDTH{1'b0}};
            multiplier_bank_1 <= {LANES*MULTIPLIER_WIDTH{1'b0}};
            bias_bank_0 <= {LANES*ACC_WIDTH{1'b0}};
            bias_bank_1 <= {LANES*ACC_WIDTH{1'b0}};
        end else begin
            if (metadata_load_valid && metadata_load_ready) begin
                if (metadata_load_bank) begin
                    multiplier_bank_1 <= metadata_load_multipliers;
                    bias_bank_1 <= metadata_load_biases;
                end else begin
                    multiplier_bank_0 <= metadata_load_multipliers;
                    bias_bank_0 <= metadata_load_biases;
                end
            end
            if (residual_load_valid && residual_load_ready) begin
                if (residual_load_output_tile[0])
                    residual_tiles_bank_1[residual_load_group] <=
                        residual_load_data;
                else
                    residual_tiles_bank_0[residual_load_group] <=
                        residual_load_data;
            end
            if (raw_valid && !postprocess_ready) begin
`ifndef SYNTHESIS
                $error("serialized down postprocess input queue overflow");
`endif
            end
        end
    end

endmodule
