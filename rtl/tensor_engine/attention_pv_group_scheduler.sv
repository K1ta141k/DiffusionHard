`timescale 1ns/1ps

module attention_pv_group_scheduler #(
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 18,
    parameter integer ACC_WIDTH = 48,
    parameter integer GROUP_WIDTH = 4,
    parameter integer OUTPUT_TILE_WIDTH = 4,
    parameter integer INTERNAL_MAC = 1,
    parameter integer ARRAY_BACKPRESSURE = 0
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [GROUP_WIDTH-1:0] group_in,
    input  wire [M_LANES*64*16-1:0] probabilities_q16_packed,
    output wire start_ready,
    output wire value_read_valid,
    output wire [1:0] value_read_key_block,
    output wire [5:0] value_read_channel,
    input  wire value_data_valid,
    input  wire [16*DATA_WIDTH-1:0] value_data_packed,
    output wire attention_tile_valid,
    input  wire attention_tile_ready,
    output wire [GROUP_WIDTH-1:0] attention_group,
    output wire [OUTPUT_TILE_WIDTH-1:0] attention_output_tile,
    output wire [2:0] attention_valid_channels,
    output wire [M_LANES*N_LANES*DATA_WIDTH-1:0]
        attention_q12_packed,
    output wire mac_request_valid,
    input  wire mac_request_ready,
    output wire mac_request_clear,
    output wire mac_request_last,
    output wire [OUTPUT_TILE_WIDTH-1:0] mac_request_tag,
    output wire [M_LANES*32*DATA_WIDTH-1:0] mac_request_activations,
    output wire [N_LANES*32*DATA_WIDTH-1:0] mac_request_weights,
    input  wire mac_response_valid,
    input  wire [OUTPUT_TILE_WIDTH-1:0] mac_response_tag,
    input  wire [M_LANES*N_LANES*ACC_WIDTH-1:0]
        mac_response_accumulators,
    output wire busy,
    output reg  done
);

    localparam [2:0] STATE_IDLE = 3'd0;
    localparam [2:0] STATE_VALUE_LOAD = 3'd1;
    localparam [2:0] STATE_MAC_WAIT = 3'd2;
    localparam [2:0] STATE_OUTPUT = 3'd3;

    reg [2:0] state;
    reg [GROUP_WIDTH-1:0] active_group;
    reg [OUTPUT_TILE_WIDTH-1:0] active_output_tile;
    reg [2:0] value_issue_channel;
    reg [2:0] value_issue_key_block;
    reg [2:0] value_capture_channel;
    reg [1:0] value_capture_key_block;
    reg [M_LANES*64*16-1:0] probability_buffer;
    reg signed [DATA_WIDTH-1:0] value_buffer [0:N_LANES-1][0:15];
    reg [M_LANES*N_LANES*DATA_WIDTH-1:0] attention_buffer;
    reg mac_issue_pending;
    reg [1:0] mac_issue_key_block;

    reg [M_LANES*32*DATA_WIDTH-1:0] mac_activations;
    reg [N_LANES*32*DATA_WIDTH-1:0] mac_weights;
    wire mac_valid_in = mac_issue_pending;
    wire selected_mac_ready = INTERNAL_MAC || !ARRAY_BACKPRESSURE
        || mac_request_ready;
    wire mac_valid_out;
    wire [OUTPUT_TILE_WIDTH-1:0] mac_tag_out;
    wire [M_LANES*N_LANES*ACC_WIDTH-1:0] mac_accumulators;
    wire [2:0] active_valid_channels =
        (active_output_tile == 10) ? 3'd4 : 3'd6;

    integer lane_index;
    integer key_index;
    integer output_index;
    reg signed [ACC_WIDTH-1:0] weighted_sum;
    reg signed [ACC_WIDTH-1:0] rounded_output;
    reg signed [DATA_WIDTH-1:0] saturated_output;

    assign start_ready = (state == STATE_IDLE);
    assign busy = (state != STATE_IDLE);
    assign value_read_valid = (state == STATE_VALUE_LOAD)
        && value_issue_key_block < 4
        && (!ARRAY_BACKPRESSURE || !mac_issue_pending);
    assign value_read_channel = active_output_tile * 6
        + value_issue_channel;
    assign value_read_key_block = value_issue_key_block[1:0];
    assign attention_tile_valid = (state == STATE_OUTPUT);
    assign attention_group = active_group;
    assign attention_output_tile = active_output_tile;
    assign attention_valid_channels = active_valid_channels;
    assign attention_q12_packed = attention_buffer;
    assign mac_request_valid = mac_valid_in;
    assign mac_request_clear = mac_issue_key_block == 0;
    assign mac_request_last = mac_issue_key_block == 3;
    assign mac_request_tag = active_output_tile;
    assign mac_request_activations = mac_activations;
    assign mac_request_weights = mac_weights;

    function automatic signed [ACC_WIDTH-1:0] round_attention_q12;
        input signed [ACC_WIDTH-1:0] value;
        begin
            if (value >= 0)
                round_attention_q12 = (value + (48'sd1 << 15)) >>> 16;
            else
                round_attention_q12 = -(((-value) + (48'sd1 << 15)) >>> 16);
        end
    endfunction

    always @* begin
        mac_activations = 0;
        mac_weights = 0;
        for (lane_index = 0; lane_index < M_LANES;
             lane_index = lane_index + 1)
            for (key_index = 0; key_index < 16;
                 key_index = key_index + 1)
                mac_activations[
                    (lane_index*32+key_index)*DATA_WIDTH +: DATA_WIDTH
                ] = {
                    2'b00,
                    probability_buffer[
                        (lane_index*64 + mac_issue_key_block*16
                         + key_index)*16 +: 16
                    ]
                };
        for (lane_index = 0; lane_index < N_LANES;
             lane_index = lane_index + 1)
            for (key_index = 0; key_index < 16;
                 key_index = key_index + 1)
                if (lane_index < active_valid_channels)
                    mac_weights[
                        (lane_index*32+key_index)*DATA_WIDTH +: DATA_WIDTH
                    ] = value_buffer[lane_index][key_index];
    end

    generate
        if (INTERNAL_MAC) begin : internal_mac
            mixed_precision_mac_tile_pipelined #(
                .M_LANES(M_LANES),
                .N_LANES(N_LANES),
                .STORAGE_WIDTH(DATA_WIDTH),
                .ACC_WIDTH(ACC_WIDTH),
                .TAG_WIDTH(OUTPUT_TILE_WIDTH)
            ) implementation (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(mac_valid_in),
                .narrow_int8_mode(1'b0),
                .clear_accumulators(mac_issue_key_block == 0),
                .last_k_tile(mac_issue_key_block == 3),
                .tag_in(active_output_tile),
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
            active_output_tile <= 0;
            value_issue_channel <= 0;
            value_issue_key_block <= 0;
            value_capture_channel <= 0;
            value_capture_key_block <= 0;
            probability_buffer <= 0;
            attention_buffer <= 0;
            mac_issue_pending <= 1'b0;
            mac_issue_key_block <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (mac_valid_in && selected_mac_ready)
                mac_issue_pending <= 1'b0;
            if (start && start_ready) begin
                active_group <= group_in;
                active_output_tile <= 0;
                probability_buffer <= probabilities_q16_packed;
                value_issue_channel <= 0;
                value_issue_key_block <= 0;
                value_capture_channel <= 0;
                value_capture_key_block <= 0;
                mac_issue_pending <= 1'b0;
                state <= STATE_VALUE_LOAD;
            end else if (state == STATE_VALUE_LOAD) begin
                if (value_read_valid) begin
                    if (value_issue_channel == active_valid_channels-1) begin
                        value_issue_channel <= 0;
                        value_issue_key_block <= value_issue_key_block + 1'b1;
                    end else begin
                        value_issue_channel <= value_issue_channel + 1'b1;
                    end
                end
                if (value_data_valid) begin
                    for (lane_index = 0; lane_index < 16;
                         lane_index = lane_index + 1)
                        value_buffer[value_capture_channel][lane_index] <=
                            value_data_packed[
                            lane_index*DATA_WIDTH +: DATA_WIDTH
                        ];
                    if (value_capture_channel == active_valid_channels-1) begin
                        value_capture_channel <= 0;
                        mac_issue_pending <= 1'b1;
                        mac_issue_key_block <= value_capture_key_block;
                        if (value_capture_key_block == 3)
                            state <= STATE_MAC_WAIT;
                        else
                            value_capture_key_block <=
                                value_capture_key_block + 1'b1;
                    end else begin
                        value_capture_channel <= value_capture_channel + 1'b1;
                    end
                end
            end else if (state == STATE_MAC_WAIT && mac_valid_out) begin
                for (output_index = 0;
                     output_index < M_LANES*N_LANES;
                     output_index = output_index + 1) begin
                    weighted_sum = $signed(mac_accumulators[
                        output_index*ACC_WIDTH +: ACC_WIDTH
                    ]);
                    rounded_output = round_attention_q12(weighted_sum);
                    if (rounded_output > 131071)
                        saturated_output = 18'sd131071;
                    else if (rounded_output < -131072)
                        saturated_output = -18'sd131072;
                    else
                        saturated_output = rounded_output[DATA_WIDTH-1:0];
                    attention_buffer[
                        output_index*DATA_WIDTH +: DATA_WIDTH
                    ] <= saturated_output;
                end
                state <= STATE_OUTPUT;
            end else if (state == STATE_OUTPUT && attention_tile_ready) begin
                if (active_output_tile == 10) begin
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else begin
                    active_output_tile <= active_output_tile + 1'b1;
                    value_issue_channel <= 0;
                    value_issue_key_block <= 0;
                    value_capture_channel <= 0;
                    value_capture_key_block <= 0;
                    mac_issue_pending <= 1'b0;
                    state <= STATE_VALUE_LOAD;
                end
            end
        end
    end

endmodule
