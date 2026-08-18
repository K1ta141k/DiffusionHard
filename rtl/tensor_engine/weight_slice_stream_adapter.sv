`timescale 1ns/1ps

module weight_slice_stream_adapter #(
    parameter integer INPUT_SIZE = 768,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 8,
    parameter integer STREAM_WIDTH = 512,
    parameter integer K_TILE_WIDTH = ((INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(INPUT_SIZE / 32),
    parameter integer BEAT_WIDTH = (((N_LANES*32*DATA_WIDTH) / STREAM_WIDTH) <= 1)
        ? 1 : $clog2((N_LANES*32*DATA_WIDTH) / STREAM_WIDTH)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire command_valid,
    input  wire command_bank,
    output wire command_ready,

    input  wire stream_valid,
    output wire stream_ready,
    input  wire [STREAM_WIDTH-1:0] stream_data,
    input  wire stream_last,

    output wire weight_load_valid,
    output wire weight_load_bank,
    output wire [K_TILE_WIDTH-1:0] weight_load_k_tile,
    output wire [N_LANES*32*DATA_WIDTH-1:0] weight_load_data,
    input  wire weight_load_ready,

    output reg  busy,
    output reg  done,
    output reg  protocol_error
);

    localparam integer TILE_WIDTH = N_LANES * 32 * DATA_WIDTH;
    localparam integer K_TILES = INPUT_SIZE / 32;
    localparam integer BEATS_PER_TILE = TILE_WIDTH / STREAM_WIDTH;

    reg active_bank;
    reg [K_TILE_WIDTH-1:0] k_tile_counter;
    reg [BEAT_WIDTH-1:0] beat_counter;
    reg [STREAM_WIDTH-1:0] beat_buffer [0:BEATS_PER_TILE-2];
    wire accepting_beat;
    wire final_beat_of_tile;
    wire final_beat_of_slice;

    initial begin
        if (INPUT_SIZE % 32 != 0) begin
            $error("INPUT_SIZE must be divisible by 32");
        end
        if (TILE_WIDTH % STREAM_WIDTH != 0) begin
            $error("weight tile width must be divisible by stream width");
        end
    end

    genvar beat_index;
    generate
        for (beat_index = 0; beat_index < BEATS_PER_TILE-1; beat_index = beat_index + 1) begin : pack_buffered_beats
            assign weight_load_data[
                beat_index*STREAM_WIDTH +: STREAM_WIDTH
            ] = beat_buffer[beat_index];
        end
    endgenerate
    assign weight_load_data[
        (BEATS_PER_TILE-1)*STREAM_WIDTH +: STREAM_WIDTH
    ] = stream_data;

    assign command_ready = !busy;
    assign final_beat_of_tile = (beat_counter == BEATS_PER_TILE-1);
    assign final_beat_of_slice = final_beat_of_tile
        && (k_tile_counter == K_TILES-1);
    assign stream_ready = busy
        && (!final_beat_of_tile || weight_load_ready);
    assign accepting_beat = stream_valid && stream_ready;
    assign weight_load_valid = accepting_beat && final_beat_of_tile;
    assign weight_load_bank = active_bank;
    assign weight_load_k_tile = k_tile_counter;

    integer reset_index;

    always @(posedge clk) begin
        if (!rst_n) begin
            active_bank <= 1'b0;
            k_tile_counter <= {K_TILE_WIDTH{1'b0}};
            beat_counter <= {BEAT_WIDTH{1'b0}};
            for (reset_index = 0; reset_index < BEATS_PER_TILE-1; reset_index = reset_index + 1) begin
                beat_buffer[reset_index] <= {STREAM_WIDTH{1'b0}};
            end
            busy <= 1'b0;
            done <= 1'b0;
            protocol_error <= 1'b0;
        end else begin
            done <= 1'b0;
            if (command_valid && command_ready) begin
                active_bank <= command_bank;
                k_tile_counter <= {K_TILE_WIDTH{1'b0}};
                beat_counter <= {BEAT_WIDTH{1'b0}};
                busy <= 1'b1;
                protocol_error <= 1'b0;
            end

            if (accepting_beat) begin
                if (stream_last != final_beat_of_slice) begin
                    protocol_error <= 1'b1;
                end
                if (final_beat_of_tile) begin
                    beat_counter <= {BEAT_WIDTH{1'b0}};
                    if (k_tile_counter == K_TILES-1) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                    end else begin
                        k_tile_counter <= k_tile_counter + 1'b1;
                    end
                end else begin
                    beat_buffer[beat_counter] <= stream_data;
                    beat_counter <= beat_counter + 1'b1;
                end
            end
        end
    end

endmodule
