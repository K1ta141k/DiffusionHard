`timescale 1ns/1ps

// Rejected wide-register baseline retained for resource comparison.
// The active implementation is mlp_interstage_tile_bridge_bram.

module mlp_interstage_tile_bridge #(
    parameter integer TOKENS = 64,
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer INPUT_SIZE = 3072,
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10,
    parameter integer GROUP_WIDTH = ((TOKENS / M_LANES) <= 1)
        ? 1 : $clog2(TOKENS / M_LANES),
    parameter integer K_TILE_WIDTH = ((INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(INPUT_SIZE / 32)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile_in,
    input  wire [GROUP_WIDTH-1:0] group_in,
    input  wire [M_LANES*N_LANES*8-1:0] values_packed,
    output reg  activation_load_valid,
    output reg  [GROUP_WIDTH-1:0] activation_load_group,
    output reg  [K_TILE_WIDTH-1:0] activation_load_k_tile,
    output reg  [M_LANES*32*8-1:0] activation_load_data,
    output wire done
);

    localparam integer TOKEN_GROUPS = TOKENS / M_LANES;
    localparam integer OUTPUT_TILES = INPUT_SIZE / N_LANES;
    localparam integer TILE_BITS = M_LANES * 32 * 8;

    reg [TILE_BITS-1:0] partial_tiles [0:TOKEN_GROUPS-1];
    reg [K_TILE_WIDTH-1:0] current_k_tiles [0:TOKEN_GROUPS-1];
    reg [TILE_BITS-1:0] completed_candidate;
    reg [TILE_BITS-1:0] spill_candidate;
    reg [OUTPUT_TILE_TAG_WIDTH+$clog2(N_LANES)-1:0] base_channel;
    reg [K_TILE_WIDTH-1:0] base_k_tile;
    reg [4:0] base_offset;
    reg completes_tile;
    integer token_index;
    integer channel_index;
    integer channel_offset;
    integer group_index;

    assign done = activation_load_valid
        && (activation_load_group == TOKEN_GROUPS-1)
        && (activation_load_k_tile == (INPUT_SIZE / 32)-1);

    initial begin
        if (TOKENS % M_LANES != 0) begin
            $error("TOKENS must be divisible by M_LANES");
        end
        if (INPUT_SIZE % 32 != 0 || INPUT_SIZE % N_LANES != 0) begin
            $error("INPUT_SIZE must be divisible by 32 and N_LANES");
        end
        if (N_LANES > 32) begin
            $error("N_LANES must not exceed one down activation tile");
        end
    end

    always @* begin
        base_channel = output_tile_in * N_LANES;
        base_k_tile = base_channel / 32;
        base_offset = base_channel % 32;
        completed_candidate = partial_tiles[group_in];
        spill_candidate = {TILE_BITS{1'b0}};
        completes_tile = (base_offset + N_LANES) >= 32;
        for (token_index = 0; token_index < M_LANES;
             token_index = token_index + 1) begin
            for (channel_index = 0; channel_index < N_LANES;
                 channel_index = channel_index + 1) begin
                channel_offset = base_offset + channel_index;
                if (channel_offset < 32) begin
                    completed_candidate[
                        (token_index*32 + channel_offset)*8 +: 8
                    ] = values_packed[
                        (token_index*N_LANES + channel_index)*8 +: 8
                    ];
                end else begin
                    spill_candidate[
                        (token_index*32 + channel_offset-32)*8 +: 8
                    ] = values_packed[
                        (token_index*N_LANES + channel_index)*8 +: 8
                    ];
                end
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            activation_load_valid <= 1'b0;
            activation_load_group <= {GROUP_WIDTH{1'b0}};
            activation_load_k_tile <= {K_TILE_WIDTH{1'b0}};
            activation_load_data <= {TILE_BITS{1'b0}};
            for (group_index = 0; group_index < TOKEN_GROUPS;
                 group_index = group_index + 1) begin
                partial_tiles[group_index] <= {TILE_BITS{1'b0}};
                current_k_tiles[group_index] <= {K_TILE_WIDTH{1'b0}};
            end
        end else begin
            activation_load_valid <= 1'b0;
            if (valid_in) begin
`ifndef SYNTHESIS
                if (base_k_tile != current_k_tiles[group_in]) begin
                    $error("interstage output tiles arrived out of order");
                end
`endif
                if (completes_tile) begin
                    activation_load_valid <= 1'b1;
                    activation_load_group <= group_in;
                    activation_load_k_tile <= base_k_tile;
                    activation_load_data <= completed_candidate;
                    partial_tiles[group_in] <= spill_candidate;
                    current_k_tiles[group_in] <= base_k_tile + 1'b1;
                end else begin
                    partial_tiles[group_in] <= completed_candidate;
                end
            end
        end
    end

endmodule
