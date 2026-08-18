`timescale 1ns/1ps

module attention_qk_group_scheduler #(
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 18,
    parameter integer ACC_WIDTH = 48,
    parameter integer GROUP_WIDTH = 4,
    parameter integer KEY_TILE_WIDTH = 4,
    parameter integer INTERNAL_MAC = 1
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [GROUP_WIDTH-1:0] group_in,
    output wire start_ready,
    output wire query_read_valid,
    output wire [5:0] query_read_token,
    output wire [1:0] query_read_channel_block,
    input  wire query_data_valid,
    input  wire [16*DATA_WIDTH-1:0] query_data_packed,
    output wire key_read_valid,
    output wire [5:0] key_read_token,
    output wire [1:0] key_read_channel_block,
    input  wire key_data_valid,
    input  wire [16*DATA_WIDTH-1:0] key_data_packed,
    output wire score_valid,
    input  wire score_ready,
    output wire [GROUP_WIDTH-1:0] score_group,
    output wire [KEY_TILE_WIDTH-1:0] score_key_tile,
    output wire [2:0] score_valid_keys,
    output wire [M_LANES*N_LANES*DATA_WIDTH-1:0] scores_q10_packed,
    output wire mac_request_valid,
    output wire mac_request_clear,
    output wire mac_request_last,
    output wire [KEY_TILE_WIDTH-1:0] mac_request_tag,
    output wire [M_LANES*32*DATA_WIDTH-1:0] mac_request_activations,
    output wire [N_LANES*32*DATA_WIDTH-1:0] mac_request_weights,
    input  wire mac_response_valid,
    input  wire [KEY_TILE_WIDTH-1:0] mac_response_tag,
    input  wire [M_LANES*N_LANES*ACC_WIDTH-1:0]
        mac_response_accumulators,
    output wire busy,
    output reg  done
);

    localparam [3:0] STATE_IDLE = 4'd0;
    localparam [3:0] STATE_QUERY_LOAD = 4'd1;
    localparam [3:0] STATE_KEY_LOAD = 4'd2;
    localparam [3:0] STATE_MAC_FIRST = 4'd3;
    localparam [3:0] STATE_MAC_SECOND = 4'd4;
    localparam [3:0] STATE_MAC_WAIT = 4'd5;
    localparam [3:0] STATE_SCORE_OUTPUT = 4'd6;

    reg [3:0] state;
    reg [GROUP_WIDTH-1:0] active_group;
    reg [KEY_TILE_WIDTH-1:0] active_key_tile;
    reg [5:0] query_issue_count;
    reg [5:0] query_capture_count;
    reg [5:0] key_issue_count;
    reg [5:0] key_capture_count;
    reg signed [DATA_WIDTH-1:0] query_buffer [0:M_LANES-1][0:63];
    reg signed [DATA_WIDTH-1:0] key_buffer [0:N_LANES-1][0:63];
    reg [M_LANES*N_LANES*DATA_WIDTH-1:0] score_buffer;

    reg [M_LANES*32*DATA_WIDTH-1:0] mac_activations;
    reg [N_LANES*32*DATA_WIDTH-1:0] mac_weights;
    wire mac_valid_in = (state == STATE_MAC_FIRST)
        || (state == STATE_MAC_SECOND);
    wire mac_valid_out;
    wire [KEY_TILE_WIDTH-1:0] mac_tag_out;
    wire [M_LANES*N_LANES*ACC_WIDTH-1:0] mac_accumulators;
    wire [5:0] key_read_limit = (active_key_tile == 10) ? 16 : 24;

    integer lane_index;
    integer channel_index;
    integer output_index;
    reg signed [ACC_WIDTH-1:0] dot_product;
    reg signed [ACC_WIDTH-1:0] rounded_score;
    reg signed [DATA_WIDTH-1:0] saturated_score;

    assign start_ready = (state == STATE_IDLE);
    assign busy = (state != STATE_IDLE);
    assign query_read_valid = (state == STATE_QUERY_LOAD)
        && (query_issue_count < 16);
    assign query_read_token = {active_group, 2'b00}
        + (query_issue_count >> 2);
    assign query_read_channel_block = query_issue_count[1:0];
    assign key_read_valid = (state == STATE_KEY_LOAD)
        && (key_issue_count < key_read_limit);
    assign key_read_token = active_key_tile * 6 + (key_issue_count >> 2);
    assign key_read_channel_block = key_issue_count[1:0];
    assign score_valid = (state == STATE_SCORE_OUTPUT);
    assign score_group = active_group;
    assign score_key_tile = active_key_tile;
    assign score_valid_keys = (active_key_tile == 10) ? 3'd4 : 3'd6;
    assign scores_q10_packed = score_buffer;
    assign mac_request_valid = mac_valid_in;
    assign mac_request_clear = (state == STATE_MAC_FIRST);
    assign mac_request_last = (state == STATE_MAC_SECOND);
    assign mac_request_tag = active_key_tile;
    assign mac_request_activations = mac_activations;
    assign mac_request_weights = mac_weights;

    function automatic signed [ACC_WIDTH-1:0] round_score_q10;
        input signed [ACC_WIDTH-1:0] value;
        begin
            if (value >= 0)
                round_score_q10 = (value + (48'sd1 << 16)) >>> 17;
            else
                round_score_q10 = -(((-value) + (48'sd1 << 16)) >>> 17);
        end
    endfunction

    always @* begin
        mac_activations = 0;
        mac_weights = 0;
        for (lane_index = 0; lane_index < M_LANES;
             lane_index = lane_index + 1)
            for (channel_index = 0; channel_index < 32;
                 channel_index = channel_index + 1)
                mac_activations[
                    (lane_index*32+channel_index)*DATA_WIDTH +: DATA_WIDTH
                ] = query_buffer[lane_index][
                    channel_index + ((state == STATE_MAC_SECOND) ? 32 : 0)
                ];
        for (lane_index = 0; lane_index < N_LANES;
             lane_index = lane_index + 1)
            for (channel_index = 0; channel_index < 32;
                 channel_index = channel_index + 1)
                mac_weights[
                    (lane_index*32+channel_index)*DATA_WIDTH +: DATA_WIDTH
                ] = key_buffer[lane_index][
                    channel_index + ((state == STATE_MAC_SECOND) ? 32 : 0)
                ];
    end

    generate
        if (INTERNAL_MAC) begin : internal_mac
            mixed_precision_mac_tile_pipelined #(
                .M_LANES(M_LANES),
                .N_LANES(N_LANES),
                .STORAGE_WIDTH(DATA_WIDTH),
                .ACC_WIDTH(ACC_WIDTH),
                .TAG_WIDTH(KEY_TILE_WIDTH)
            ) implementation (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(mac_valid_in),
                .narrow_int8_mode(1'b0),
                .clear_accumulators(state == STATE_MAC_FIRST),
                .last_k_tile(state == STATE_MAC_SECOND),
                .tag_in(active_key_tile),
                .activations_packed(mac_activations),
                .weights_packed(mac_weights),
                .valid_out(mac_valid_out),
                .tag_out(mac_tag_out),
                .accumulators_packed(mac_accumulators)
            );
        end else begin : external_mac
            assign mac_valid_out = mac_response_valid;
            assign mac_tag_out = mac_response_tag;
            assign mac_accumulators = mac_response_accumulators;
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_group <= 0;
            active_key_tile <= 0;
            query_issue_count <= 0;
            query_capture_count <= 0;
            key_issue_count <= 0;
            key_capture_count <= 0;
            score_buffer <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start && start_ready) begin
                active_group <= group_in;
                active_key_tile <= 0;
                query_issue_count <= 0;
                query_capture_count <= 0;
                state <= STATE_QUERY_LOAD;
            end else if (state == STATE_QUERY_LOAD) begin
                if (query_read_valid)
                    query_issue_count <= query_issue_count + 1'b1;
                if (query_data_valid) begin
                    for (lane_index = 0; lane_index < 16;
                         lane_index = lane_index + 1)
                        query_buffer[query_capture_count >> 2][
                            query_capture_count[1:0]*16 + lane_index
                        ] <= query_data_packed[
                            lane_index*DATA_WIDTH +: DATA_WIDTH
                        ];
                    if (query_capture_count == 15) begin
                        key_issue_count <= 0;
                        key_capture_count <= 0;
                        for (lane_index = 0; lane_index < N_LANES;
                             lane_index = lane_index + 1)
                            for (channel_index = 0; channel_index < 64;
                                 channel_index = channel_index + 1)
                                key_buffer[lane_index][channel_index] <= 0;
                        state <= STATE_KEY_LOAD;
                    end else begin
                        query_capture_count <= query_capture_count + 1'b1;
                    end
                end
            end else if (state == STATE_KEY_LOAD) begin
                if (key_read_valid)
                    key_issue_count <= key_issue_count + 1'b1;
                if (key_data_valid) begin
                    for (lane_index = 0; lane_index < 16;
                         lane_index = lane_index + 1)
                        key_buffer[key_capture_count >> 2][
                            key_capture_count[1:0]*16 + lane_index
                        ] <= key_data_packed[
                            lane_index*DATA_WIDTH +: DATA_WIDTH
                        ];
                    if (key_capture_count == key_read_limit-1)
                        state <= STATE_MAC_FIRST;
                    else
                        key_capture_count <= key_capture_count + 1'b1;
                end
            end else if (state == STATE_MAC_FIRST) begin
                state <= STATE_MAC_SECOND;
            end else if (state == STATE_MAC_SECOND) begin
                state <= STATE_MAC_WAIT;
            end else if (state == STATE_MAC_WAIT && mac_valid_out) begin
                for (output_index = 0;
                     output_index < M_LANES*N_LANES;
                     output_index = output_index + 1) begin
                    dot_product = $signed(mac_accumulators[
                        output_index*ACC_WIDTH +: ACC_WIDTH
                    ]);
                    rounded_score = round_score_q10(dot_product);
                    if (rounded_score > 131071)
                        saturated_score = 18'sd131071;
                    else if (rounded_score < -131072)
                        saturated_score = -18'sd131072;
                    else
                        saturated_score = rounded_score[DATA_WIDTH-1:0];
                    score_buffer[
                        output_index*DATA_WIDTH +: DATA_WIDTH
                    ] <= saturated_score;
                end
                state <= STATE_SCORE_OUTPUT;
            end else if (state == STATE_SCORE_OUTPUT && score_ready) begin
                if (active_key_tile == 10) begin
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else begin
                    active_key_tile <= active_key_tile + 1'b1;
                    key_issue_count <= 0;
                    key_capture_count <= 0;
                    for (lane_index = 0; lane_index < N_LANES;
                         lane_index = lane_index + 1)
                        for (channel_index = 0; channel_index < 64;
                             channel_index = channel_index + 1)
                            key_buffer[lane_index][channel_index] <= 0;
                    state <= STATE_KEY_LOAD;
                end
            end
        end
    end

endmodule
