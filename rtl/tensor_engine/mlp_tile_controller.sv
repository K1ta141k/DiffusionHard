`timescale 1ns/1ps

module mlp_tile_controller #(
    parameter integer TOKENS = 64,
    parameter integer INPUT_SIZE = 768,
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
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
    input  wire [K_TILE_WIDTH-1:0] weight_load_k_tile,
    input  wire [N_LANES*32*DATA_WIDTH-1:0] weight_load_data,

    input  wire start,
    output reg  busy,
    output wire result_valid,
    output wire [GROUP_WIDTH-1:0] result_group,
    output wire [M_LANES*N_LANES*ACC_WIDTH-1:0] result_accumulators,
    output wire done
);

    localparam integer K_LANES = 32;
    localparam integer TOKEN_GROUPS = TOKENS / M_LANES;
    localparam integer K_TILES = INPUT_SIZE / K_LANES;
    reg [M_LANES*K_LANES*DATA_WIDTH-1:0] activation_tiles
        [0:TOKEN_GROUPS-1][0:K_TILES-1];
    reg [N_LANES*K_LANES*DATA_WIDTH-1:0] weight_tiles
        [0:K_TILES-1];

    reg [GROUP_WIDTH-1:0] group_counter;
    reg [K_TILE_WIDTH-1:0] k_tile_counter;
    wire [M_LANES*K_LANES*DATA_WIDTH-1:0] core_activations;
    wire [N_LANES*K_LANES*DATA_WIDTH-1:0] core_weights;
    wire core_clear;
    wire core_last;

    initial begin
        if (TOKENS % M_LANES != 0) begin
            $error("TOKENS must be divisible by M_LANES");
        end
        if (INPUT_SIZE % K_LANES != 0) begin
            $error("INPUT_SIZE must be divisible by 32");
        end
    end

    assign core_activations = activation_tiles[group_counter][k_tile_counter];
    assign core_weights = weight_tiles[k_tile_counter];

    assign core_clear = (k_tile_counter == {K_TILE_WIDTH{1'b0}});
    assign core_last = (k_tile_counter == K_TILES-1);
    assign done = result_valid && (result_group == TOKEN_GROUPS-1);

    int8_mac_tile_pipelined #(
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .TAG_WIDTH(GROUP_WIDTH)
    ) compute_core (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(busy),
        .clear_accumulators(core_clear),
        .last_k_tile(core_last),
        .tag_in(group_counter),
        .activations_packed(core_activations),
        .weights_packed(core_weights),
        .valid_out(result_valid),
        .tag_out(result_group),
        .accumulators_packed(result_accumulators)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0;
            group_counter <= {GROUP_WIDTH{1'b0}};
            k_tile_counter <= {K_TILE_WIDTH{1'b0}};
        end else begin
            if (activation_load_valid && !busy) begin
                activation_tiles[activation_load_group][activation_load_k_tile] <=
                    activation_load_data;
            end
            if (weight_load_valid && !busy) begin
                weight_tiles[weight_load_k_tile] <= weight_load_data;
            end

            if (start && !busy) begin
                busy <= 1'b1;
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
        end
    end

endmodule
