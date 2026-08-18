`timescale 1ns/1ps

module attention_qk_group_pair_scheduler_packed_m8 #(
    parameter integer N_LANES = 6,
    parameter integer INTERNAL_MAC = 1,
    parameter integer ARRAY_BACKPRESSURE = 0,
    parameter integer TAG_WIDTH = 6,
    parameter integer OUTPUTS = 8 * N_LANES,
    parameter integer KEY_TILES = (64 + N_LANES - 1) / N_LANES
) (
    input  wire clk,
    input  wire rst_n,
    input  wire scale_load_valid,
    input  wire [5:0] scale_load_token,
    input  wire [17:0] scale_load_query_maximum,
    input  wire [17:0] scale_load_key_maximum,
    input  wire [23:0] scale_load_query_multiplier_q17,
    input  wire [23:0] scale_load_key_multiplier_q17,
    input  wire start,
    input  wire [2:0] group_pair_in,
    output wire start_ready,
    output wire query_read_valid,
    output wire [5:0] query_read_token,
    output wire [1:0] query_read_channel_block,
    input  wire query_data_valid,
    input  wire [16*18-1:0] query_data_q12_packed,
    output wire key_read_valid,
    output wire [5:0] key_read_token,
    output wire [1:0] key_read_channel_block,
    input  wire key_data_valid,
    input  wire [16*18-1:0] key_data_q12_packed,
    output wire score_pair_valid,
    input  wire score_pair_ready,
    output wire [2:0] score_group_pair,
    output wire [5:0] score_key_tile,
    output wire [2:0] score_valid_keys,
    output wire [OUTPUTS*18-1:0] scores_q10_packed,
    output wire mac_request_valid,
    input  wire mac_request_ready,
    output wire mac_request_clear,
    output wire mac_request_last,
    output wire [TAG_WIDTH-1:0] mac_request_tag,
    output wire [8*32*8-1:0] mac_request_activations_int8,
    output wire [N_LANES*32*8-1:0] mac_request_weights_int8,
    input  wire mac_response_valid,
    input  wire [TAG_WIDTH-1:0] mac_response_tag,
    input  wire [OUTPUTS*32-1:0] mac_response_accumulators,
    output wire busy,
    output reg  done
);

    localparam [3:0] STATE_IDLE = 4'd0;
    localparam [3:0] STATE_QUERY_LOAD = 4'd1;
    localparam [3:0] STATE_KEY_CLEAR = 4'd2;
    localparam [3:0] STATE_KEY_LOAD = 4'd3;
    localparam [3:0] STATE_MAC_FIRST = 4'd4;
    localparam [3:0] STATE_MAC_SECOND = 4'd5;
    localparam [3:0] STATE_MAC_WAIT = 4'd6;
    localparam [3:0] STATE_REQUANTIZE = 4'd7;
    localparam [3:0] STATE_SCORE_OUTPUT = 4'd8;
    localparam [3:0] STATE_WAIT_PREFETCH = 4'd9;

    reg [3:0] state;
    reg [2:0] active_group_pair;
    reg [5:0] active_key_tile;
    reg [5:0] query_issue_count;
    reg [5:0] query_response_count;
    reg [5:0] key_issue_count;
    reg [5:0] key_response_count;
    reg [5:0] requant_issue_pair;
    reg prefetch_active;
    reg prefetch_done;
    reg signed [7:0] query_buffer [0:1][0:7][0:31];
    reg signed [7:0] key_buffer [0:1][0:N_LANES-1][0:31];
    reg signed [20:0] dot_buffer [0:OUTPUTS-1];
    reg signed [17:0] score_buffer [0:OUTPUTS-1];
    reg [17:0] query_maximum_table [0:63];
    reg [17:0] key_maximum_table [0:63];
    reg [23:0] query_multiplier_table [0:63];
    reg [23:0] key_multiplier_table [0:63];
    reg [17:0] query_maximum_buffer [0:7];
    reg [17:0] key_maximum_buffer [0:N_LANES-1];
    reg [17:0] key_maximum_prefetch_buffer [0:N_LANES-1];

    wire [5:0] key_count = (active_key_tile == KEY_TILES-1)
        ? (64 - (KEY_TILES-1)*N_LANES) : N_LANES;
    wire [5:0] key_read_limit = key_count * 4;
    wire [5:0] next_key_tile = active_key_tile + 1'b1;
    wire [5:0] next_key_count = (next_key_tile == KEY_TILES-1)
        ? (64 - (KEY_TILES-1)*N_LANES) : N_LANES;
    wire [5:0] next_key_read_limit = next_key_count * 4;
    wire [5:0] query_base_token = {active_group_pair, 3'b000};
    wire prefetch_context = state == STATE_REQUANTIZE
        || state == STATE_SCORE_OUTPUT || state == STATE_WAIT_PREFETCH;

    wire quantizer_valid_in =
        (state == STATE_QUERY_LOAD && query_data_valid)
        || ((state == STATE_KEY_LOAD || prefetch_context) && key_data_valid);
    wire [5:0] quantizer_tag_in = (state == STATE_QUERY_LOAD)
        ? query_response_count : key_response_count;
    wire [16*18-1:0] quantizer_values = (state == STATE_QUERY_LOAD)
        ? query_data_q12_packed : key_data_q12_packed;
    wire [5:0] quantizer_token = (state == STATE_QUERY_LOAD)
        ? query_base_token + (query_response_count >> 2)
        : ((prefetch_context ? next_key_tile : active_key_tile)*N_LANES)
            + (key_response_count >> 2);
    wire [23:0] quantizer_multiplier = (state == STATE_QUERY_LOAD)
        ? query_multiplier_table[quantizer_token]
        : key_multiplier_table[quantizer_token];
    wire quantizer_valid_out;
    wire [5:0] quantizer_tag_out;
    wire [16*8-1:0] quantizer_values_int8;

    reg [8*32*8-1:0] mac_activations;
    reg [N_LANES*32*8-1:0] mac_weights;
    wire internal_mac_valid;
    wire internal_mac_mode;
    wire [TAG_WIDTH-1:0] internal_mac_tag;
    wire [OUTPUTS*32-1:0] internal_mac_accumulators;
    wire selected_mac_valid = INTERNAL_MAC
        ? internal_mac_valid : mac_response_valid;
    wire selected_mac_ready = INTERNAL_MAC || !ARRAY_BACKPRESSURE
        || mac_request_ready;
    wire [TAG_WIDTH-1:0] selected_mac_tag = INTERNAL_MAC
        ? internal_mac_tag : mac_response_tag;
    wire [OUTPUTS*32-1:0] selected_mac_accumulators = INTERNAL_MAC
        ? internal_mac_accumulators : mac_response_accumulators;

    wire requant_inputs_valid = (state == STATE_REQUANTIZE)
        && (requant_issue_pair < OUTPUTS/2);
    wire [5:0] requant_index_0 = requant_issue_pair * 2;
    wire [5:0] requant_index_1 = requant_issue_pair * 2 + 1'b1;
    wire [2:0] requant_row_0 = requant_index_0 / N_LANES;
    wire [2:0] requant_row_1 = requant_index_1 / N_LANES;
    wire [2:0] requant_key_lane_0 = requant_index_0 % N_LANES;
    wire [2:0] requant_key_lane_1 = requant_index_1 % N_LANES;
    wire [5:0] requant_query_token_0 = query_base_token + requant_row_0;
    wire [5:0] requant_query_token_1 = query_base_token + requant_row_1;
    wire [5:0] requant_key_token_0 =
        active_key_tile*N_LANES + requant_key_lane_0;
    wire [5:0] requant_key_token_1 =
        active_key_tile*N_LANES + requant_key_lane_1;
    wire requant_key_valid_0 = requant_key_lane_0 < key_count;
    wire requant_key_valid_1 = requant_key_lane_1 < key_count;
    wire requant_valid_0;
    wire requant_valid_1;
    wire [5:0] requant_tag_0;
    wire [5:0] requant_tag_1;
    wire signed [17:0] requant_score_0;
    wire signed [17:0] requant_score_1;

    integer lane;
    integer channel;
    integer output_index;
    genvar packed_output;

    assign start_ready = state == STATE_IDLE;
    assign busy = state != STATE_IDLE;
    assign query_read_valid = state == STATE_QUERY_LOAD
        && query_issue_count < 32;
    assign query_read_token = query_base_token + (query_issue_count >> 2);
    assign query_read_channel_block = query_issue_count[1:0];
    assign key_read_valid =
        ((state == STATE_KEY_LOAD) && key_issue_count < key_read_limit)
        || (prefetch_context && prefetch_active
            && key_issue_count < next_key_read_limit);
    assign key_read_token = (prefetch_context ? next_key_tile : active_key_tile)
        * N_LANES + (key_issue_count >> 2);
    assign key_read_channel_block = key_issue_count[1:0];
    assign score_pair_valid = state == STATE_SCORE_OUTPUT;
    assign score_group_pair = active_group_pair;
    assign score_key_tile = active_key_tile;
    assign score_valid_keys = key_count[2:0];
    assign mac_request_valid = state == STATE_MAC_FIRST
        || state == STATE_MAC_SECOND;
    assign mac_request_clear = state == STATE_MAC_FIRST;
    assign mac_request_last = state == STATE_MAC_SECOND;
    assign mac_request_tag = active_key_tile[TAG_WIDTH-1:0];
    assign mac_request_activations_int8 = mac_activations;
    assign mac_request_weights_int8 = mac_weights;

    generate
        for (packed_output = 0; packed_output < OUTPUTS;
             packed_output = packed_output + 1) begin : pack_scores
            assign scores_q10_packed[packed_output*18 +: 18] =
                score_buffer[packed_output];
        end
    endgenerate

    attention_dynamic_int8_quantizer_16 #(
        .TAG_WIDTH(6)
    ) quantizer (
        .clk(clk), .rst_n(rst_n), .valid_in(quantizer_valid_in),
        .tag_in(quantizer_tag_in), .values_q12_packed(quantizer_values),
        .multiplier_q17(quantizer_multiplier),
        .valid_out(quantizer_valid_out), .tag_out(quantizer_tag_out),
        .values_int8_packed(quantizer_values_int8)
    );

    attention_dynamic_score_requantizer_q10 #(
        .TAG_WIDTH(6)
    ) requantizer_0 (
        .clk(clk), .rst_n(rst_n), .valid_in(requant_inputs_valid),
        .tag_in(requant_index_0),
        .dot_product_int8(dot_buffer[requant_index_0]),
        .query_maximum(query_maximum_buffer[requant_row_0]),
        .key_maximum(requant_key_valid_0
            ? key_maximum_buffer[requant_key_lane_0] : 18'd1),
        .valid_out(requant_valid_0), .tag_out(requant_tag_0),
        .score_q10(requant_score_0)
    );

    attention_dynamic_score_requantizer_q10 #(
        .TAG_WIDTH(6)
    ) requantizer_1 (
        .clk(clk), .rst_n(rst_n), .valid_in(requant_inputs_valid),
        .tag_in(requant_index_1),
        .dot_product_int8(dot_buffer[requant_index_1]),
        .query_maximum(query_maximum_buffer[requant_row_1]),
        .key_maximum(requant_key_valid_1
            ? key_maximum_buffer[requant_key_lane_1] : 18'd1),
        .valid_out(requant_valid_1), .tag_out(requant_tag_1),
        .score_q10(requant_score_1)
    );

    always @* begin
        mac_activations = 0;
        mac_weights = 0;
        for (lane = 0; lane < 8; lane = lane + 1)
            for (channel = 0; channel < 32; channel = channel + 1)
                mac_activations[(lane*32+channel)*8 +: 8] =
                    query_buffer[state == STATE_MAC_SECOND][lane][channel];
        for (lane = 0; lane < N_LANES; lane = lane + 1)
            for (channel = 0; channel < 32; channel = channel + 1)
                if (lane < key_count)
                    mac_weights[(lane*32+channel)*8 +: 8] =
                        key_buffer[state == STATE_MAC_SECOND][lane][channel];
    end

    generate
        if (INTERNAL_MAC) begin : internal_array
            mixed_precision_packed_m8_mac_tile_pipelined #(
                .N_LANES(N_LANES), .TAG_WIDTH(TAG_WIDTH)
            ) array (
                .clk(clk), .rst_n(rst_n), .valid_in(mac_request_valid),
                .narrow_int8_mode(1'b1),
                .clear_accumulators(mac_request_clear),
                .last_k_tile(mac_request_last), .tag_in(mac_request_tag),
                .attention_activations_packed({4*32*18{1'b0}}),
                .attention_weights_packed({N_LANES*32*18{1'b0}}),
                .mlp_activations_packed(mac_activations),
                .mlp_weights_packed(mac_weights),
                .valid_out(internal_mac_valid),
                .narrow_int8_mode_out(internal_mac_mode),
                .tag_out(internal_mac_tag),
                .attention_accumulators_packed(),
                .mlp_accumulators_packed(internal_mac_accumulators)
            );
        end else begin : external_array
            assign internal_mac_valid = 1'b0;
            assign internal_mac_mode = 1'b0;
            assign internal_mac_tag = 0;
            assign internal_mac_accumulators = 0;
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_group_pair <= 0;
            active_key_tile <= 0;
            query_issue_count <= 0;
            query_response_count <= 0;
            key_issue_count <= 0;
            key_response_count <= 0;
            requant_issue_pair <= 0;
            prefetch_active <= 1'b0;
            prefetch_done <= 1'b0;
            done <= 1'b0;
            for (lane = 0; lane < 8; lane = lane + 1)
                query_maximum_buffer[lane] <= 1;
            for (lane = 0; lane < N_LANES; lane = lane + 1) begin
                key_maximum_buffer[lane] <= 1;
                key_maximum_prefetch_buffer[lane] <= 1;
            end
        end else begin
            done <= 1'b0;
            if (scale_load_valid) begin
                query_maximum_table[scale_load_token] <=
                    scale_load_query_maximum;
                key_maximum_table[scale_load_token] <= scale_load_key_maximum;
                query_multiplier_table[scale_load_token] <=
                    scale_load_query_multiplier_q17;
                key_multiplier_table[scale_load_token] <=
                    scale_load_key_multiplier_q17;
            end
            if (prefetch_context && prefetch_active) begin
                if (key_read_valid)
                    key_issue_count <= key_issue_count + 1'b1;
                if (key_data_valid)
                    key_response_count <= key_response_count + 1'b1;
                if (key_data_valid && key_response_count[1:0] == 0)
                    key_maximum_prefetch_buffer[
                        key_response_count >> 2
                    ] <= key_maximum_table[quantizer_token];
                if (quantizer_valid_out) begin
                    for (channel = 0; channel < 16; channel = channel + 1)
                        key_buffer[quantizer_tag_out[1]][
                            quantizer_tag_out >> 2
                        ][quantizer_tag_out[0]*16 + channel] <=
                            quantizer_values_int8[channel*8 +: 8];
                    if (quantizer_tag_out == next_key_read_limit-1) begin
                        prefetch_done <= 1'b1;
                        prefetch_active <= 1'b0;
                    end
                end
            end
            if (start && start_ready) begin
                active_group_pair <= group_pair_in;
                active_key_tile <= 0;
                query_issue_count <= 0;
                query_response_count <= 0;
                prefetch_active <= 1'b0;
                prefetch_done <= 1'b0;
                state <= STATE_QUERY_LOAD;
            end else if (state == STATE_QUERY_LOAD) begin
                if (query_read_valid)
                    query_issue_count <= query_issue_count + 1'b1;
                if (query_data_valid)
                    query_response_count <= query_response_count + 1'b1;
                if (query_data_valid && query_response_count[1:0] == 0)
                    query_maximum_buffer[
                        query_response_count >> 2
                    ] <= query_maximum_table[quantizer_token];
                if (quantizer_valid_out) begin
                    for (channel = 0; channel < 16; channel = channel + 1)
                        query_buffer[quantizer_tag_out[1]][
                            quantizer_tag_out >> 2
                        ][quantizer_tag_out[0]*16 + channel] <=
                            quantizer_values_int8[channel*8 +: 8];
                    if (quantizer_tag_out == 31) begin
                        key_issue_count <= 0;
                        key_response_count <= 0;
                        state <= STATE_KEY_LOAD;
                    end
                end
            end else if (state == STATE_KEY_LOAD) begin
                if (key_read_valid)
                    key_issue_count <= key_issue_count + 1'b1;
                if (key_data_valid)
                    key_response_count <= key_response_count + 1'b1;
                if (key_data_valid && key_response_count[1:0] == 0)
                    key_maximum_buffer[key_response_count >> 2] <=
                        key_maximum_table[quantizer_token];
                if (quantizer_valid_out) begin
                    for (channel = 0; channel < 16; channel = channel + 1)
                        key_buffer[quantizer_tag_out[1]][
                            quantizer_tag_out >> 2
                        ][quantizer_tag_out[0]*16 + channel] <=
                            quantizer_values_int8[channel*8 +: 8];
                    if (quantizer_tag_out == key_read_limit-1)
                        state <= STATE_MAC_FIRST;
                end
            end else if (state == STATE_MAC_FIRST && selected_mac_ready) begin
                state <= STATE_MAC_SECOND;
            end else if (state == STATE_MAC_SECOND && selected_mac_ready) begin
                state <= STATE_MAC_WAIT;
            end else if (state == STATE_MAC_WAIT && selected_mac_valid) begin
                for (output_index = 0; output_index < OUTPUTS;
                     output_index = output_index + 1)
                    dot_buffer[output_index] <= selected_mac_accumulators[
                        output_index*32 +: 21
                    ];
                requant_issue_pair <= 0;
                key_issue_count <= 0;
                key_response_count <= 0;
                prefetch_done <= 1'b0;
                if (active_key_tile != KEY_TILES-1) begin
                    prefetch_active <= 1'b1;
                end else begin
                    prefetch_active <= 1'b0;
                end
                state <= STATE_REQUANTIZE;
            end else if (state == STATE_REQUANTIZE) begin
                if (requant_inputs_valid)
                    requant_issue_pair <= requant_issue_pair + 1'b1;
                if (requant_valid_0)
                    score_buffer[requant_tag_0] <= requant_score_0;
                if (requant_valid_1) begin
                    score_buffer[requant_tag_1] <= requant_score_1;
                    if (requant_tag_1 == OUTPUTS-1)
                        state <= STATE_SCORE_OUTPUT;
                end
            end else if (state == STATE_SCORE_OUTPUT && score_pair_ready) begin
                if (active_key_tile == KEY_TILES-1) begin
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else if (prefetch_done) begin
                    for (lane = 0; lane < N_LANES; lane = lane + 1)
                        key_maximum_buffer[lane] <=
                            key_maximum_prefetch_buffer[lane];
                    active_key_tile <= active_key_tile + 1'b1;
                    state <= STATE_MAC_FIRST;
                end else begin
                    state <= STATE_WAIT_PREFETCH;
                end
            end else if (state == STATE_WAIT_PREFETCH && prefetch_done) begin
                    for (lane = 0; lane < N_LANES; lane = lane + 1)
                        key_maximum_buffer[lane] <=
                            key_maximum_prefetch_buffer[lane];
                    active_key_tile <= active_key_tile + 1'b1;
                    state <= STATE_MAC_FIRST;
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && start && !start_ready)
            $error("packed QK group-pair start arrived while busy");
        if (rst_n && selected_mac_valid && !internal_mac_mode && INTERNAL_MAC)
            $error("packed QK array returned the wrong arithmetic mode");
        if (rst_n && selected_mac_valid
            && selected_mac_tag != active_key_tile[TAG_WIDTH-1:0])
            $error("packed QK array response tag mismatch");
        if (rst_n && requant_valid_0 != requant_valid_1)
            $error("packed QK score requantizer lanes lost alignment");
`endif
    end

endmodule
