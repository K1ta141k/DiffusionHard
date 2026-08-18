`timescale 1ns/1ps

module mlp_tile_pingpong_controller #(
    parameter integer TOKENS = 64,
    parameter integer INPUT_SIZE = 768,
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
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
    output wire bank_0_load_ready,
    output wire bank_1_load_ready,

    input  wire start,
    input  wire start_bank,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] start_output_tile,
    output wire start_ready,
    output reg  busy,
    output reg  active_bank,

    output wire result_valid,
    output wire result_bank,
    output wire [OUTPUT_TILE_TAG_WIDTH-1:0] result_output_tile,
    output wire [GROUP_WIDTH-1:0] result_group,
    output wire [M_LANES*N_LANES*ACC_WIDTH-1:0] result_accumulators,
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

    localparam integer K_LANES = 32;
    localparam integer TOKEN_GROUPS = TOKENS / M_LANES;
    localparam integer K_TILES = INPUT_SIZE / K_LANES;
    localparam integer CORE_TAG_WIDTH = 1 + OUTPUT_TILE_TAG_WIDTH + GROUP_WIDTH;
    localparam integer ACTIVATION_DEPTH = TOKEN_GROUPS * K_TILES;
    localparam integer ACTIVATION_ADDR_WIDTH = (ACTIVATION_DEPTH <= 1)
        ? 1 : $clog2(ACTIVATION_DEPTH);

    reg [N_LANES*K_LANES*DATA_WIDTH-1:0] weight_tiles_bank_0
        [0:K_TILES-1];
    reg [N_LANES*K_LANES*DATA_WIDTH-1:0] weight_tiles_bank_1
        [0:K_TILES-1];

    reg [GROUP_WIDTH-1:0] group_counter;
    reg [K_TILE_WIDTH-1:0] k_tile_counter;
    reg [OUTPUT_TILE_TAG_WIDTH-1:0] active_output_tile;
    reg [1:0] bank_pending;
    wire [M_LANES*K_LANES*DATA_WIDTH-1:0] core_activations;
    wire [N_LANES*K_LANES*DATA_WIDTH-1:0] core_weights;
    wire internal_result_valid;
    wire [CORE_TAG_WIDTH-1:0] internal_result_tag;
    wire [M_LANES*N_LANES*ACC_WIDTH-1:0]
        internal_result_accumulators;
    wire selected_result_valid = INTERNAL_MAC
        ? internal_result_valid : array_response_valid;
    wire [CORE_TAG_WIDTH-1:0] selected_result_tag = INTERNAL_MAC
        ? internal_result_tag : array_response_tag;
    wire [M_LANES*N_LANES*ACC_WIDTH-1:0] selected_result_accumulators =
        INTERNAL_MAC ? internal_result_accumulators
                     : array_response_accumulators;
    wire core_clear;
    wire core_last;
    wire sync_activation_valid;
    wire [M_LANES*K_LANES*DATA_WIDTH-1:0] sync_activation_data;
    reg [N_LANES*K_LANES*DATA_WIDTH-1:0] sync_weight_data;
    reg sync_clear;
    reg sync_last;
    reg [CORE_TAG_WIDTH-1:0] sync_tag;
    wire [ACTIVATION_ADDR_WIDTH-1:0] activation_write_address =
        activation_load_group * K_TILES + activation_load_k_tile;
    wire [ACTIVATION_ADDR_WIDTH-1:0] activation_read_address =
        group_counter * K_TILES + k_tile_counter;

    initial begin
        if (TOKENS % M_LANES != 0) begin
            $error("TOKENS must be divisible by M_LANES");
        end
        if (INPUT_SIZE % K_LANES != 0) begin
            $error("INPUT_SIZE must be divisible by 32");
        end
    end

    generate
        if (SYNC_ACTIVATION_MEMORY) begin : synchronous_activation_memory
            wide_synchronous_uram #(
                .WIDTH(M_LANES*K_LANES*DATA_WIDTH),
                .DEPTH(ACTIVATION_DEPTH),
                .ADDR_WIDTH(ACTIVATION_ADDR_WIDTH)
            ) activations (
                .clk(clk),
                .write_valid(activation_load_valid && !busy),
                .write_address(activation_write_address),
                .write_data(activation_load_data),
                .read_valid(busy),
                .read_address(activation_read_address),
                .read_data_valid(sync_activation_valid),
                .read_data(sync_activation_data)
            );
            assign core_activations = sync_activation_data;
        end else begin : asynchronous_activation_memory
            reg [M_LANES*K_LANES*DATA_WIDTH-1:0] activation_tiles
                [0:TOKEN_GROUPS-1][0:K_TILES-1];
            always @(posedge clk) begin
                if (activation_load_valid && !busy)
                    activation_tiles[activation_load_group][
                        activation_load_k_tile
                    ] <= activation_load_data;
            end
            assign core_activations = activation_tiles[
                group_counter
            ][k_tile_counter];
            assign sync_activation_valid = 1'b0;
            assign sync_activation_data =
                {M_LANES*K_LANES*DATA_WIDTH{1'b0}};
        end
    endgenerate
    assign core_weights = active_bank
        ? weight_tiles_bank_1[k_tile_counter]
        : weight_tiles_bank_0[k_tile_counter];
    assign core_clear = (k_tile_counter == {K_TILE_WIDTH{1'b0}});
    assign core_last = (k_tile_counter == K_TILES-1);
    assign array_request_valid = SYNC_ACTIVATION_MEMORY
        ? sync_activation_valid : busy;
    assign array_request_clear = SYNC_ACTIVATION_MEMORY
        ? sync_clear : core_clear;
    assign array_request_last = SYNC_ACTIVATION_MEMORY
        ? sync_last : core_last;
    assign array_request_tag = SYNC_ACTIVATION_MEMORY
        ? sync_tag : {active_bank, active_output_tile, group_counter};
    assign array_request_activations = core_activations;
    assign array_request_weights = SYNC_ACTIVATION_MEMORY
        ? sync_weight_data : core_weights;

    assign start_ready = !busy && !bank_pending[start_bank];
    assign weight_load_ready = !bank_pending[weight_load_bank];
    assign bank_0_load_ready = !bank_pending[0];
    assign bank_1_load_ready = !bank_pending[1];
    assign result_valid = selected_result_valid;
    assign result_accumulators = selected_result_accumulators;
    assign result_group = selected_result_tag[GROUP_WIDTH-1:0];
    assign result_output_tile = selected_result_tag[
        GROUP_WIDTH +: OUTPUT_TILE_TAG_WIDTH
    ];
    assign result_bank = selected_result_tag[CORE_TAG_WIDTH-1];
    assign done = result_valid && (result_group == TOKEN_GROUPS-1);

    generate
        if (INTERNAL_MAC) begin : internal_mac
            int8_mac_tile_pipelined #(
                .M_LANES(M_LANES),
                .N_LANES(N_LANES),
                .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH(ACC_WIDTH),
                .TAG_WIDTH(CORE_TAG_WIDTH)
            ) compute_core (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(array_request_valid),
                .clear_accumulators(array_request_clear),
                .last_k_tile(array_request_last),
                .tag_in(array_request_tag),
                .activations_packed(array_request_activations),
                .weights_packed(array_request_weights),
                .valid_out(internal_result_valid),
                .tag_out(internal_result_tag),
                .accumulators_packed(internal_result_accumulators)
            );
        end else begin : no_internal_mac
            assign internal_result_valid = 1'b0;
            assign internal_result_tag = {CORE_TAG_WIDTH{1'b0}};
            assign internal_result_accumulators =
                {M_LANES*N_LANES*ACC_WIDTH{1'b0}};
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0;
            active_bank <= 1'b0;
            active_output_tile <= {OUTPUT_TILE_TAG_WIDTH{1'b0}};
            bank_pending <= 2'b0;
            group_counter <= {GROUP_WIDTH{1'b0}};
            k_tile_counter <= {K_TILE_WIDTH{1'b0}};
            sync_weight_data <= {N_LANES*K_LANES*DATA_WIDTH{1'b0}};
            sync_clear <= 1'b0;
            sync_last <= 1'b0;
            sync_tag <= {CORE_TAG_WIDTH{1'b0}};
        end else begin
            if (weight_load_valid && weight_load_ready) begin
                if (weight_load_bank) begin
                    weight_tiles_bank_1[weight_load_k_tile] <= weight_load_data;
                end else begin
                    weight_tiles_bank_0[weight_load_k_tile] <= weight_load_data;
                end
            end

            if (SYNC_ACTIVATION_MEMORY && busy) begin
                sync_weight_data <= core_weights;
                sync_clear <= core_clear;
                sync_last <= core_last;
                sync_tag <= {active_bank, active_output_tile, group_counter};
            end

            if (start && start_ready) begin
                busy <= 1'b1;
                active_bank <= start_bank;
                active_output_tile <= start_output_tile;
                bank_pending[start_bank] <= 1'b1;
                group_counter <= {GROUP_WIDTH{1'b0}};
                k_tile_counter <= {K_TILE_WIDTH{1'b0}};
            end else if (busy) begin
                if (k_tile_counter == K_TILES-1) begin
                    k_tile_counter <= {K_TILE_WIDTH{1'b0}};
                    if (group_counter == TOKEN_GROUPS-1) begin
                        busy <= 1'b0;
                    end else begin
                        group_counter <= group_counter + 1'b1;
                    end
                end else begin
                    k_tile_counter <= k_tile_counter + 1'b1;
                end
            end
            if (done) begin
                bank_pending[result_bank] <= 1'b0;
            end
        end
    end

endmodule
