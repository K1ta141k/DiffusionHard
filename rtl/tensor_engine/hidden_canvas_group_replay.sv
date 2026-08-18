`timescale 1ns/1ps

module hidden_canvas_group_replay (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [3:0] group_in,
    output wire start_ready,
    output wire canvas_read_valid,
    output wire [3:0] canvas_read_group,
    output wire [6:0] canvas_read_output_tile,
    input  wire canvas_read_data_valid,
    input  wire [4*6*24-1:0] canvas_read_q10_packed,
    output wire output_valid,
    input  wire output_ready,
    output wire [3:0] output_group,
    output wire [9:0] output_channel,
    output wire [4*24-1:0] output_q10_packed,
    output wire busy,
    output reg  done
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_READ = 2'd1;
    localparam [1:0] STATE_EMIT = 2'd2;

    reg [1:0] state;
    reg [3:0] active_group;
    reg [6:0] active_output_tile;
    reg [2:0] active_lane;
    reg [4*6*24-1:0] tile_buffer;
    reg read_issued;
    genvar token_lane;

    assign start_ready = state == STATE_IDLE;
    assign canvas_read_valid = state == STATE_READ && !read_issued;
    assign canvas_read_group = active_group;
    assign canvas_read_output_tile = active_output_tile;
    assign output_valid = state == STATE_EMIT;
    assign output_group = active_group;
    assign output_channel = active_output_tile * 6 + active_lane;
    assign busy = state != STATE_IDLE;

    generate
        for (token_lane = 0; token_lane < 4;
             token_lane = token_lane + 1) begin : select_tokens
            assign output_q10_packed[token_lane*24 +: 24] = tile_buffer[
                (token_lane*6+active_lane)*24 +: 24
            ];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_group <= 0;
            active_output_tile <= 0;
            active_lane <= 0;
            tile_buffer <= 0;
            read_issued <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && start) begin
                active_group <= group_in;
                active_output_tile <= 0;
                active_lane <= 0;
                read_issued <= 1'b0;
                state <= STATE_READ;
            end else if (state == STATE_READ && !read_issued) begin
                read_issued <= 1'b1;
            end
            if (state == STATE_READ && canvas_read_data_valid) begin
                tile_buffer <= canvas_read_q10_packed;
                active_lane <= 0;
                read_issued <= 1'b0;
                state <= STATE_EMIT;
            end else if (state == STATE_EMIT && output_ready) begin
                if (active_lane == 5) begin
                    active_lane <= 0;
                    if (active_output_tile == 127) begin
                        state <= STATE_IDLE;
                        done <= 1'b1;
                    end else begin
                        active_output_tile <= active_output_tile + 1'b1;
                        read_issued <= 1'b0;
                        state <= STATE_READ;
                    end
                end else begin
                    active_lane <= active_lane + 1'b1;
                end
            end
        end
    end

endmodule
