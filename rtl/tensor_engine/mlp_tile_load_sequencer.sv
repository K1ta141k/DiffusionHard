`timescale 1ns/1ps

module mlp_tile_load_sequencer #(
    parameter integer INPUT_SIZE = 768,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 8,
    parameter integer METADATA_WIDTH = 444,
    parameter integer K_TILE_WIDTH = ((INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(INPUT_SIZE / 32)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire command_valid,
    input  wire command_bank,
    output wire command_ready,

    input  wire weight_stream_valid,
    output wire weight_stream_ready,
    input  wire [N_LANES*32*DATA_WIDTH-1:0] weight_stream_data,
    input  wire metadata_stream_valid,
    output wire metadata_stream_ready,
    input  wire [METADATA_WIDTH-1:0] metadata_stream_data,

    output wire weight_load_valid,
    output wire weight_load_bank,
    output wire [K_TILE_WIDTH-1:0] weight_load_k_tile,
    output wire [N_LANES*32*DATA_WIDTH-1:0] weight_load_data,
    input  wire weight_load_ready,
    output wire metadata_load_valid,
    output wire metadata_load_bank,
    output wire [METADATA_WIDTH-1:0] metadata_load_data,
    input  wire metadata_load_ready,

    output reg  busy,
    output reg  done
);

    localparam integer K_TILES = INPUT_SIZE / 32;

    reg active_bank;
    reg [K_TILE_WIDTH-1:0] weight_tile_counter;
    reg weights_complete;
    reg metadata_complete;
    wire weight_accept = weight_load_valid && weight_load_ready;
    wire metadata_accept = metadata_load_valid && metadata_load_ready;
    wire accepting_last_weight = weight_accept
        && weight_tile_counter == K_TILES-1;
    wire all_weights = weights_complete || accepting_last_weight;
    wire all_metadata = metadata_complete || metadata_accept;

    assign command_ready = !busy && !done;
    assign weight_stream_ready = busy && !weights_complete
        && weight_load_ready;
    assign metadata_stream_ready = busy && !metadata_complete
        && metadata_load_ready;
    assign weight_load_valid = busy && !weights_complete
        && weight_stream_valid;
    assign weight_load_bank = active_bank;
    assign weight_load_k_tile = weight_tile_counter;
    assign weight_load_data = weight_stream_data;
    assign metadata_load_valid = busy && !metadata_complete
        && metadata_stream_valid;
    assign metadata_load_bank = active_bank;
    assign metadata_load_data = metadata_stream_data;

    always @(posedge clk) begin
        if (!rst_n) begin
            active_bank <= 1'b0;
            weight_tile_counter <= {K_TILE_WIDTH{1'b0}};
            weights_complete <= 1'b0;
            metadata_complete <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (command_valid && command_ready) begin
                active_bank <= command_bank;
                weight_tile_counter <= {K_TILE_WIDTH{1'b0}};
                weights_complete <= 1'b0;
                metadata_complete <= 1'b0;
                busy <= 1'b1;
            end else if (busy) begin
                if (weight_accept) begin
                    if (weight_tile_counter == K_TILES-1)
                        weights_complete <= 1'b1;
                    else
                        weight_tile_counter <= weight_tile_counter + 1'b1;
                end
                if (metadata_accept)
                    metadata_complete <= 1'b1;
                if (all_weights && all_metadata) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end

    initial begin
        if (INPUT_SIZE < 32 || INPUT_SIZE % 32 != 0)
            $error("MLP tile loader input size must be divisible by 32");
        if (METADATA_WIDTH < 1)
            $error("MLP tile loader metadata width must be positive");
    end

endmodule
