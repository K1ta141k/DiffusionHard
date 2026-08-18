`timescale 1ns/1ps

module mlp_block_controller #(
    parameter integer FRONTEND_GROUPS = 16,
    parameter integer UP_OUTPUT_TILES = 512,
    parameter integer DOWN_OUTPUT_TILES = 128,
    parameter integer TILE_WIDTH = 10
) (
    input  wire clk,
    input  wire rst_n,
    input  wire block_start,
    output wire block_start_ready,
    output wire frontend_start,
    output wire [3:0] frontend_group,
    input  wire frontend_start_ready,
    input  wire frontend_done,
    output wire up_load_enable,
    output wire [TILE_WIDTH-1:0] up_load_tile,
    output wire up_load_bank,
    input  wire up_load_done,
    output wire up_start,
    output wire [TILE_WIDTH-1:0] up_start_tile,
    output wire up_start_bank,
    input  wire up_start_ready,
    input  wire up_tile_done,
    input  wire up_all_activations_done,
    output wire down_load_enable,
    output wire [TILE_WIDTH-1:0] down_load_tile,
    output wire down_load_bank,
    input  wire down_load_done,
    output wire down_start,
    output wire [TILE_WIDTH-1:0] down_start_tile,
    output wire down_start_bank,
    input  wire down_start_ready,
    input  wire down_tile_done,
    output wire busy,
    output reg  done
);

    localparam [3:0] STATE_IDLE = 4'd0;
    localparam [3:0] STATE_FRONTEND_START = 4'd1;
    localparam [3:0] STATE_FRONTEND_RUN = 4'd2;
    localparam [3:0] STATE_UP_LOAD = 4'd3;
    localparam [3:0] STATE_UP_START = 4'd4;
    localparam [3:0] STATE_UP_RUN = 4'd5;
    localparam [3:0] STATE_UP_DRAIN = 4'd6;
    localparam [3:0] STATE_DOWN_LOAD = 4'd7;
    localparam [3:0] STATE_DOWN_START = 4'd8;
    localparam [3:0] STATE_DOWN_RUN = 4'd9;

    reg [3:0] state;
    reg [3:0] active_frontend_group;
    reg [TILE_WIDTH-1:0] active_up_tile;
    reg [TILE_WIDTH-1:0] active_down_tile;
    reg next_tile_loaded;
    wire up_has_next = active_up_tile < UP_OUTPUT_TILES-1;
    wire down_has_next = active_down_tile < DOWN_OUTPUT_TILES-1;
    wire up_loading_next = state == STATE_UP_RUN && up_has_next
        && !next_tile_loaded;
    wire down_loading_next = state == STATE_DOWN_RUN && down_has_next
        && !next_tile_loaded;

    assign block_start_ready = state == STATE_IDLE;
    assign busy = state != STATE_IDLE;
    assign frontend_start = state == STATE_FRONTEND_START
        && frontend_start_ready;
    assign frontend_group = active_frontend_group;
    assign up_load_enable = state == STATE_UP_LOAD || up_loading_next;
    assign up_load_tile = up_loading_next
        ? active_up_tile + 1'b1 : active_up_tile;
    assign up_load_bank = up_load_tile[0];
    assign up_start = state == STATE_UP_START && up_start_ready;
    assign up_start_tile = active_up_tile;
    assign up_start_bank = active_up_tile[0];
    assign down_load_enable = state == STATE_DOWN_LOAD || down_loading_next;
    assign down_load_tile = down_loading_next
        ? active_down_tile + 1'b1 : active_down_tile;
    assign down_load_bank = down_load_tile[0];
    assign down_start = state == STATE_DOWN_START && down_start_ready;
    assign down_start_tile = active_down_tile;
    assign down_start_bank = active_down_tile[0];

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_frontend_group <= 0;
            active_up_tile <= 0;
            active_down_tile <= 0;
            next_tile_loaded <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && block_start) begin
                active_frontend_group <= 0;
                state <= STATE_FRONTEND_START;
            end else if (state == STATE_FRONTEND_START && frontend_start) begin
                state <= STATE_FRONTEND_RUN;
            end else if (state == STATE_FRONTEND_RUN && frontend_done) begin
                if (active_frontend_group == FRONTEND_GROUPS-1) begin
                    active_up_tile <= 0;
                    state <= STATE_UP_LOAD;
                end else begin
                    active_frontend_group <= active_frontend_group + 1'b1;
                    state <= STATE_FRONTEND_START;
                end
            end else if (state == STATE_UP_LOAD && up_load_done) begin
                state <= STATE_UP_START;
            end else if (state == STATE_UP_START && up_start) begin
                next_tile_loaded <= 1'b0;
                state <= STATE_UP_RUN;
            end else if (state == STATE_UP_RUN) begin
                if (up_loading_next && up_load_done)
                    next_tile_loaded <= 1'b1;
                if (up_tile_done) begin
                    if (!up_has_next) begin
                        next_tile_loaded <= 1'b0;
                        if (up_all_activations_done) begin
                            active_down_tile <= 0;
                            state <= STATE_DOWN_LOAD;
                        end else begin
                            state <= STATE_UP_DRAIN;
                        end
                    end else begin
                        active_up_tile <= active_up_tile + 1'b1;
                        next_tile_loaded <= 1'b0;
                        if (next_tile_loaded
                            || (up_loading_next && up_load_done))
                            state <= STATE_UP_START;
                        else
                            state <= STATE_UP_LOAD;
                    end
                end
            end else if (state == STATE_UP_DRAIN
                         && up_all_activations_done) begin
                active_down_tile <= 0;
                state <= STATE_DOWN_LOAD;
            end else if (state == STATE_DOWN_LOAD && down_load_done) begin
                state <= STATE_DOWN_START;
            end else if (state == STATE_DOWN_START && down_start) begin
                next_tile_loaded <= 1'b0;
                state <= STATE_DOWN_RUN;
            end else if (state == STATE_DOWN_RUN) begin
                if (down_loading_next && down_load_done)
                    next_tile_loaded <= 1'b1;
                if (down_tile_done) begin
                    if (!down_has_next) begin
                        state <= STATE_IDLE;
                        next_tile_loaded <= 1'b0;
                        done <= 1'b1;
                    end else begin
                        active_down_tile <= active_down_tile + 1'b1;
                        next_tile_loaded <= 1'b0;
                        if (next_tile_loaded
                            || (down_loading_next && down_load_done))
                            state <= STATE_DOWN_START;
                        else
                            state <= STATE_DOWN_LOAD;
                    end
                end
            end
        end
    end

    initial begin
        if (FRONTEND_GROUPS < 1 || FRONTEND_GROUPS > 16)
            $error("MLP controller supports one through sixteen groups");
        if (UP_OUTPUT_TILES < 1 || DOWN_OUTPUT_TILES < 1)
            $error("MLP controller requires positive tile counts");
    end

endmodule
