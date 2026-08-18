`timescale 1ns/1ps

module qkv_head_output_router (
    input  wire clk,
    input  wire rst_n,
    input  wire tile_valid,
    output wire tile_ready,
    input  wire [3:0] tile_head,
    input  wire [1:0] tile_kind,
    input  wire [3:0] tile_group,
    input  wire [3:0] tile_channel_tile,
    input  wire [2:0] tile_valid_channels,
    input  wire [4*6*18-1:0] tile_q12_packed,
    output wire query_load_valid,
    output wire key_load_valid,
    output wire value_load_valid,
    output wire [3:0] load_head,
    output wire [5:0] load_token,
    output wire [5:0] load_channel,
    output wire signed [17:0] load_q12,
    output reg  tile_done,
    output reg  [3:0] done_head,
    output reg  [1:0] done_kind,
    output reg  [3:0] done_group,
    output reg  [3:0] done_channel_tile,
    output wire busy
);

    reg active;
    reg [3:0] active_head;
    reg [1:0] active_kind;
    reg [3:0] active_group;
    reg [3:0] active_channel_tile;
    reg [2:0] active_valid_channels;
    reg [1:0] token_lane;
    reg [2:0] channel_lane;
    reg [4*6*18-1:0] tile_buffer;
    wire lane_valid = channel_lane < active_valid_channels;
    wire [6:0] channel_base = (active_channel_tile << 2)
        + (active_channel_tile << 1);

    assign tile_ready = !active;
    assign busy = active;
    assign query_load_valid = active && lane_valid && active_kind == 0;
    assign key_load_valid = active && lane_valid && active_kind == 1;
    assign value_load_valid = active && lane_valid && active_kind == 2;
    assign load_head = active_head;
    assign load_token = {active_group, token_lane};
    assign load_channel = channel_base + channel_lane;
    assign load_q12 = tile_buffer[
        (token_lane*6+channel_lane)*18 +: 18
    ];

    always @(posedge clk) begin
        if (!rst_n) begin
            active <= 1'b0;
            active_head <= 0;
            active_kind <= 0;
            active_group <= 0;
            active_channel_tile <= 0;
            active_valid_channels <= 0;
            token_lane <= 0;
            channel_lane <= 0;
            tile_buffer <= 0;
            tile_done <= 1'b0;
            done_head <= 0;
            done_kind <= 0;
            done_group <= 0;
            done_channel_tile <= 0;
        end else begin
            tile_done <= 1'b0;
            if (tile_valid && tile_ready) begin
                active <= 1'b1;
                active_head <= tile_head;
                active_kind <= tile_kind;
                active_group <= tile_group;
                active_channel_tile <= tile_channel_tile;
                active_valid_channels <= tile_valid_channels;
                token_lane <= 0;
                channel_lane <= 0;
                tile_buffer <= tile_q12_packed;
            end else if (active) begin
                if (channel_lane == 5) begin
                    channel_lane <= 0;
                    if (token_lane == 3) begin
                        token_lane <= 0;
                        active <= 1'b0;
                        tile_done <= 1'b1;
                        done_head <= active_head;
                        done_kind <= active_kind;
                        done_group <= active_group;
                        done_channel_tile <= active_channel_tile;
                    end else begin
                        token_lane <= token_lane + 1'b1;
                    end
                end else begin
                    channel_lane <= channel_lane + 1'b1;
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && tile_valid && tile_ready && tile_kind > 2)
            $error("QKV output router received an invalid kind");
`endif
    end

endmodule

module qkv_head_output_router_parallel (
    input  wire clk,
    input  wire rst_n,
    input  wire tile_valid,
    output wire tile_ready,
    input  wire [3:0] tile_head,
    input  wire [1:0] tile_kind,
    input  wire [3:0] tile_group,
    input  wire [3:0] tile_channel_tile,
    input  wire [2:0] tile_valid_channels,
    input  wire [4*6*18-1:0] tile_q12_packed,
    output wire qk_vector_load_valid,
    output wire qk_vector_load_is_key,
    output wire [5:0] qk_vector_load_token,
    output wire [5:0] qk_vector_load_channel_base,
    output wire [2:0] qk_vector_load_valid_channels,
    output wire [6*18-1:0] qk_vector_load_q12_packed,
    output wire value_vector_load_valid,
    output wire [3:0] value_vector_load_group,
    output wire [5:0] value_vector_load_channel,
    output wire [4*18-1:0] value_vector_load_q12_packed,
    output reg  tile_done,
    output reg  [3:0] done_head,
    output reg  [1:0] done_kind,
    output reg  [3:0] done_group,
    output reg  [3:0] done_channel_tile,
    output wire busy
);

    reg active;
    reg [3:0] active_head;
    reg [1:0] active_kind;
    reg [3:0] active_group;
    reg [3:0] active_channel_tile;
    reg [2:0] active_valid_channels;
    reg [2:0] step;
    reg [17:0] tile_values [0:3][0:5];
    integer capture_token;
    integer capture_channel;
    wire [6:0] channel_base_wide = (active_channel_tile << 2)
        + (active_channel_tile << 1);
    wire [5:0] channel_base = channel_base_wide[5:0];
    wire last_step = active_kind == 2
        ? step == active_valid_channels-1'b1 : step == 3;

    genvar qk_lane;
    genvar value_lane;

    assign tile_ready = !active;
    assign busy = active;
    assign qk_vector_load_valid = active && active_kind < 2;
    assign qk_vector_load_is_key = active_kind == 1;
    assign qk_vector_load_token = {active_group, step[1:0]};
    assign qk_vector_load_channel_base = channel_base;
    assign qk_vector_load_valid_channels = active_valid_channels;
    assign value_vector_load_valid = active && active_kind == 2;
    assign value_vector_load_group = active_group;
    assign value_vector_load_channel = channel_base + step;

    generate
        for (qk_lane = 0; qk_lane < 6; qk_lane = qk_lane + 1) begin : qk_pack
            assign qk_vector_load_q12_packed[qk_lane*18 +: 18] =
                tile_values[step[1:0]][qk_lane];
        end
        for (value_lane = 0; value_lane < 4;
             value_lane = value_lane + 1) begin : value_pack
            assign value_vector_load_q12_packed[value_lane*18 +: 18] =
                tile_values[value_lane][step];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            active <= 1'b0;
            active_head <= 0;
            active_kind <= 0;
            active_group <= 0;
            active_channel_tile <= 0;
            active_valid_channels <= 0;
            step <= 0;
            tile_done <= 1'b0;
            done_head <= 0;
            done_kind <= 0;
            done_group <= 0;
            done_channel_tile <= 0;
        end else begin
            tile_done <= 1'b0;
            if (tile_valid && tile_ready) begin
                active <= 1'b1;
                active_head <= tile_head;
                active_kind <= tile_kind;
                active_group <= tile_group;
                active_channel_tile <= tile_channel_tile;
                active_valid_channels <= tile_valid_channels;
                step <= 0;
                for (capture_token = 0; capture_token < 4;
                     capture_token = capture_token + 1)
                    for (capture_channel = 0; capture_channel < 6;
                         capture_channel = capture_channel + 1)
                        tile_values[capture_token][capture_channel] <=
                            tile_q12_packed[
                                (capture_token*6+capture_channel)*18 +: 18
                            ];
            end else if (active) begin
                if (last_step) begin
                    active <= 1'b0;
                    step <= 0;
                    tile_done <= 1'b1;
                    done_head <= active_head;
                    done_kind <= active_kind;
                    done_group <= active_group;
                    done_channel_tile <= active_channel_tile;
                end else begin
                    step <= step + 1'b1;
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && tile_valid && tile_ready && tile_kind > 2)
            $error("parallel QKV output router received an invalid kind");
        if (rst_n && tile_valid && tile_ready
            && (tile_valid_channels < 1 || tile_valid_channels > 6))
            $error("parallel QKV output router received invalid channel count");
`endif
    end

endmodule
