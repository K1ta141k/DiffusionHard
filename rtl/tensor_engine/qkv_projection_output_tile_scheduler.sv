`timescale 1ns/1ps

module qkv_projection_output_tile_scheduler #(
    parameter integer INTERNAL_MAC = 1
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output wire start_ready,
    input  wire [8:0] output_tile_in,
    input  wire [6*24-1:0] multipliers_packed,
    input  wire [6*18-1:0] biases_q12_packed,
    input  wire weight_tile_valid,
    output wire weight_tile_ready,
    input  wire [4:0] weight_input_tile,
    input  wire [6*32*16-1:0] weight_int16_packed,
    output wire normalized_read_valid,
    output wire [3:0] normalized_read_group,
    output wire [4:0] normalized_read_input_tile,
    input  wire normalized_read_data_valid,
    input  wire [4*32*18-1:0] normalized_q12_packed,
    output wire qkv_tile_valid,
    input  wire qkv_tile_ready,
    output wire [3:0] qkv_group,
    output wire [8:0] qkv_output_tile,
    output wire [4*6*18-1:0] qkv_q12_packed,
    output wire array_request_valid,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [7:0] array_request_tag,
    output wire [4*32*18-1:0] array_request_activations,
    output wire [6*32*18-1:0] array_request_weights,
    input  wire array_response_valid,
    input  wire [7:0] array_response_tag,
    input  wire [4*6*48-1:0] array_response_accumulators,
    output wire busy,
    output reg  done
);

    localparam [2:0] STATE_IDLE = 3'd0;
    localparam [2:0] STATE_LOAD = 3'd1;
    localparam [2:0] STATE_READ = 3'd2;
    localparam [2:0] STATE_ISSUE = 3'd3;
    localparam [2:0] STATE_WAIT_MAC = 3'd4;
    localparam [2:0] STATE_WAIT_REQUANT = 3'd5;
    localparam [2:0] STATE_OUTPUT = 3'd6;

    reg [2:0] state;
    reg [8:0] active_output_tile;
    reg [4:0] active_input_tile;
    reg [4:0] requests_sent;
    reg [4:0] response_input_tile;
    reg [4:0] loaded_tiles;
    reg [3:0] active_group;
    reg [6*24-1:0] active_multipliers;
    reg [6*18-1:0] active_biases;
    reg [4*6*18-1:0] output_buffer;

    wire [6*32*16-1:0] selected_weight_int16;
    wire [6*32*18-1:0] selected_weight_extended;
    wire [4*6*48-1:0] internal_accumulators;
    wire internal_response_valid;
    wire [7:0] internal_response_tag;
    wire response_valid = INTERNAL_MAC
        ? internal_response_valid : array_response_valid;
    wire [7:0] response_tag = INTERNAL_MAC
        ? internal_response_tag : array_response_tag;
    wire [4*6*48-1:0] response_accumulators = INTERNAL_MAC
        ? internal_accumulators : array_response_accumulators;
    wire [4*6*24-1:0] expanded_multipliers;
    wire [4*6*48-1:0] expanded_biases;
    wire requant_ready;
    wire requant_valid;
    wire [7:0] requant_tag;
    wire [4*6*18-1:0] requant_outputs;
    wire accepted_weight = weight_tile_valid && weight_tile_ready;

    genvar weight_lane;
    genvar token_lane;
    genvar output_lane;

    assign start_ready = (state == STATE_IDLE);
    assign weight_tile_ready = (state == STATE_LOAD)
        && weight_input_tile == loaded_tiles;
    assign normalized_read_valid = (state == STATE_READ) && requests_sent < 24;
    assign normalized_read_group = active_group;
    assign normalized_read_input_tile = requests_sent;
    assign qkv_tile_valid = (state == STATE_OUTPUT);
    assign qkv_group = active_group;
    assign qkv_output_tile = active_output_tile;
    assign qkv_q12_packed = output_buffer;
    assign busy = (state != STATE_IDLE);
    assign array_request_valid = (state == STATE_READ)
        && normalized_read_data_valid;
    assign array_request_clear = (response_input_tile == 0);
    assign array_request_last = (response_input_tile == 23);
    assign array_request_tag = {active_group, response_input_tile[3:0]};
    assign array_request_activations = normalized_q12_packed;
    assign array_request_weights = selected_weight_extended;

    generate
        for (weight_lane = 0; weight_lane < 6*32;
             weight_lane = weight_lane + 1) begin : extend_weights
            assign selected_weight_extended[
                weight_lane*18 +: 18
            ] = {{2{selected_weight_int16[weight_lane*16+15]}},
                 selected_weight_int16[weight_lane*16 +: 16]};
        end
        for (token_lane = 0; token_lane < 4;
             token_lane = token_lane + 1) begin : token_metadata
            for (output_lane = 0; output_lane < 6;
                 output_lane = output_lane + 1) begin : output_metadata
                assign expanded_multipliers[
                    (token_lane*6+output_lane)*24 +: 24
                ] = active_multipliers[output_lane*24 +: 24];
                assign expanded_biases[
                    (token_lane*6+output_lane)*48 +: 48
                ] = {{30{active_biases[output_lane*18+17]}},
                     active_biases[output_lane*18 +: 18]};
            end
        end
    endgenerate

    qkv_weight_tile_buffer weight_buffer (
        .clk(clk), .write_valid(accepted_weight),
        .write_tile(weight_input_tile),
        .write_weights_packed(weight_int16_packed),
        .read_tile(response_input_tile),
        .read_weights_packed(selected_weight_int16)
    );

    generate
        if (INTERNAL_MAC) begin : internal_array
            mixed_precision_mac_tile_pipelined #(
                .M_LANES(4), .N_LANES(6), .STORAGE_WIDTH(18),
                .ACC_WIDTH(48), .TAG_WIDTH(8)
            ) mac (
                .clk(clk), .rst_n(rst_n), .valid_in(array_request_valid),
                .narrow_int8_mode(1'b0),
                .clear_accumulators(array_request_clear),
                .last_k_tile(array_request_last),
                .tag_in(array_request_tag),
                .activations_packed(array_request_activations),
                .weights_packed(selected_weight_extended),
                .valid_out(internal_response_valid),
                .tag_out(internal_response_tag),
                .accumulators_packed(internal_accumulators)
            );
        end else begin : no_internal_array
            assign internal_response_valid = 1'b0;
            assign internal_response_tag = 8'b0;
            assign internal_accumulators = {4*6*48{1'b0}};
        end
    endgenerate

    fixed_requantize_vector_parallel #(
        .LANES(24), .ACC_WIDTH(48), .MULTIPLIER_WIDTH(24),
        .OUTPUT_WIDTH(18), .RIGHT_SHIFT(28), .TAG_WIDTH(8)
    ) requantizer (
        .clk(clk), .rst_n(rst_n), .valid_in(response_valid),
        .ready_in(requant_ready), .tag_in({4'b0, active_group}),
        .accumulators_packed(response_accumulators),
        .multipliers_packed(expanded_multipliers),
        .biases_packed(expanded_biases), .valid_out(requant_valid),
        .tag_out(requant_tag), .outputs_packed(requant_outputs)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_output_tile <= 0;
            active_input_tile <= 0;
            requests_sent <= 0;
            response_input_tile <= 0;
            loaded_tiles <= 0;
            active_group <= 0;
            active_multipliers <= 0;
            active_biases <= 0;
            output_buffer <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && start) begin
                state <= STATE_LOAD;
                active_output_tile <= output_tile_in;
                active_multipliers <= multipliers_packed;
                active_biases <= biases_q12_packed;
                loaded_tiles <= 0;
                active_group <= 0;
            end else if (state == STATE_LOAD && accepted_weight) begin
                if (loaded_tiles == 23) begin
                    state <= STATE_READ;
                    active_input_tile <= 0;
                    requests_sent <= 0;
                end else begin
                    loaded_tiles <= loaded_tiles + 1'b1;
                end
            end else if (state == STATE_READ) begin
                if (normalized_read_valid) begin
                    response_input_tile <= requests_sent;
                    requests_sent <= requests_sent + 1'b1;
                end
                if (normalized_read_data_valid && response_input_tile == 23)
                    state <= STATE_WAIT_MAC;
            end else if (state == STATE_WAIT_MAC && response_valid) begin
                state <= STATE_WAIT_REQUANT;
            end else if (state == STATE_WAIT_REQUANT && requant_valid) begin
                output_buffer <= requant_outputs;
                state <= STATE_OUTPUT;
            end else if (state == STATE_OUTPUT && qkv_tile_ready) begin
                if (active_group == 15) begin
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else begin
                    active_group <= active_group + 1'b1;
                    active_input_tile <= 0;
                    requests_sent <= 0;
                    state <= STATE_READ;
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && state == STATE_WAIT_MAC && response_valid
            && !requant_ready)
            $error("QKV requantizer queue overflow");
`endif
    end

endmodule
