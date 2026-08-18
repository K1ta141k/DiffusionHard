`timescale 1ns/1ps

module qkv_head_tile_controller #(
    parameter integer KINDS = 3,
    parameter integer CHANNEL_TILES = 11,
    parameter integer LAST_TILE_VALID_CHANNELS = 4
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output wire start_ready,
    input  wire [3:0] head_in,
    output wire metadata_enable,
    input  wire metadata_fire,
    input  wire tile_start_ready,
    output wire tile_start,
    input  wire tile_done,
    output wire [3:0] active_head,
    output wire [1:0] active_kind,
    output wire [3:0] active_channel_tile,
    output wire [2:0] active_valid_channels,
    output wire [11:0] active_global_row,
    output wire busy,
    output reg  done
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_METADATA = 2'd1;
    localparam [1:0] STATE_START = 2'd2;
    localparam [1:0] STATE_RUN = 2'd3;

    reg [1:0] state;
    reg [3:0] head;
    reg [1:0] kind;
    reg [3:0] channel_tile;
    wire [6:0] channel_offset = (channel_tile << 2) + (channel_tile << 1);
    wire [11:0] kind_base = (kind == 0) ? 0
        : (kind == 1) ? 12'd768 : 12'd1536;

    assign start_ready = (state == STATE_IDLE);
    assign metadata_enable = (state == STATE_METADATA);
    assign tile_start = (state == STATE_START) && tile_start_ready;
    assign active_head = head;
    assign active_kind = kind;
    assign active_channel_tile = channel_tile;
    assign active_valid_channels = (channel_tile == CHANNEL_TILES-1)
        ? LAST_TILE_VALID_CHANNELS : 3'd6;
    assign active_global_row = kind_base + {head, 6'b0} + channel_offset;
    assign busy = (state != STATE_IDLE);

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            head <= 0;
            kind <= 0;
            channel_tile <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && start) begin
                head <= head_in;
                kind <= 0;
                channel_tile <= 0;
                state <= STATE_METADATA;
            end else if (state == STATE_METADATA && metadata_fire) begin
                state <= STATE_START;
            end else if (state == STATE_START && tile_start) begin
                state <= STATE_RUN;
            end else if (state == STATE_RUN && tile_done) begin
                if (channel_tile == CHANNEL_TILES-1) begin
                    channel_tile <= 0;
                    if (kind == KINDS-1) begin
                        state <= STATE_IDLE;
                        done <= 1'b1;
                    end else begin
                        kind <= kind + 1'b1;
                        state <= STATE_METADATA;
                    end
                end else begin
                    channel_tile <= channel_tile + 1'b1;
                    state <= STATE_METADATA;
                end
            end
        end
    end

    initial begin
        if (KINDS < 1 || KINDS > 3)
            $error("QKV controller supports one through three kinds");
        if (CHANNEL_TILES < 1 || CHANNEL_TILES > 11)
            $error("QKV controller supports one through eleven channel tiles");
    end

endmodule
