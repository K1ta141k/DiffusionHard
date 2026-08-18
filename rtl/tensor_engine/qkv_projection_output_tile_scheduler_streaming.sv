`timescale 1ns/1ps

module qkv_projection_output_tile_scheduler_streaming #(
    parameter integer INTERNAL_MAC = 1,
    parameter integer ARRAY_BACKPRESSURE = 0
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
    input  wire array_request_ready,
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

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_LOAD = 2'd1;
    localparam [1:0] STATE_STREAM = 2'd2;
    localparam [1:0] STATE_DRAIN = 2'd3;

    reg [1:0] state;
    reg [8:0] active_output_tile;
    reg [4:0] loaded_tiles;
    reg [3:0] issue_group;
    reg [4:0] issue_input_tile;
    reg [3:0] response_group;
    reg [4:0] response_input_tile;
    reg [4:0] outputs_sent;
    reg read_pending;
    reg [6*24-1:0] active_multipliers;
    reg [6*18-1:0] active_biases;

    reg [1:0] fifo_count;
    reg fifo_read_pointer;
    reg fifo_write_pointer;
    reg [3:0] fifo_group [0:1];
    reg [4*6*18-1:0] fifo_data [0:1];

    wire accepted_weight = weight_tile_valid && weight_tile_ready;
    wire accepted_normalized_read = normalized_read_valid;
    wire stream_response = (state == STATE_STREAM || state == STATE_DRAIN)
        && normalized_read_data_valid;
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
    wire selected_array_ready = INTERNAL_MAC || !ARRAY_BACKPRESSURE
        || array_request_ready;
    wire accepted_array_request = array_request_valid
        && selected_array_ready;
    wire [4*6*24-1:0] expanded_multipliers;
    wire [4*6*48-1:0] expanded_biases;
    wire requant_ready;
    wire requant_valid;
    wire [7:0] requant_tag;
    wire [4*6*18-1:0] requant_outputs;
    wire fifo_push = requant_valid;
    wire fifo_pop = qkv_tile_valid && qkv_tile_ready;

    genvar weight_lane;
    genvar token_lane;
    genvar output_lane;

    assign start_ready = state == STATE_IDLE;
    assign busy = state != STATE_IDLE;
    assign weight_tile_ready = state == STATE_LOAD
        && weight_input_tile == loaded_tiles;
    assign normalized_read_valid = state == STATE_STREAM
        && (!ARRAY_BACKPRESSURE || (!read_pending && selected_array_ready));
    assign normalized_read_group = issue_group;
    assign normalized_read_input_tile = issue_input_tile;
    assign array_request_valid = stream_response;
    assign array_request_clear = response_input_tile == 0;
    assign array_request_last = response_input_tile == 23;
    assign array_request_tag = {4'b0, response_group};
    assign array_request_activations = normalized_q12_packed;
    assign array_request_weights = selected_weight_extended;
    assign qkv_tile_valid = fifo_count != 0;
    assign qkv_group = fifo_group[fifo_read_pointer];
    assign qkv_output_tile = active_output_tile;
    assign qkv_q12_packed = fifo_data[fifo_read_pointer];

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
        .OUTPUT_WIDTH(18), .RIGHT_SHIFT(28), .TAG_WIDTH(8),
        .PARALLEL_LANES(4)
    ) requantizer (
        .clk(clk), .rst_n(rst_n), .valid_in(response_valid),
        .ready_in(requant_ready), .tag_in(response_tag),
        .accumulators_packed(response_accumulators),
        .multipliers_packed(expanded_multipliers),
        .biases_packed(expanded_biases), .valid_out(requant_valid),
        .tag_out(requant_tag), .outputs_packed(requant_outputs)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_output_tile <= 0;
            loaded_tiles <= 0;
            issue_group <= 0;
            issue_input_tile <= 0;
            response_group <= 0;
            response_input_tile <= 0;
            outputs_sent <= 0;
            read_pending <= 0;
            active_multipliers <= 0;
            active_biases <= 0;
            fifo_count <= 0;
            fifo_read_pointer <= 0;
            fifo_write_pointer <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && start) begin
                state <= STATE_LOAD;
                active_output_tile <= output_tile_in;
                active_multipliers <= multipliers_packed;
                active_biases <= biases_q12_packed;
                loaded_tiles <= 0;
                issue_group <= 0;
                issue_input_tile <= 0;
                outputs_sent <= 0;
                read_pending <= 0;
                fifo_count <= 0;
                fifo_read_pointer <= 0;
                fifo_write_pointer <= 0;
            end else if (state == STATE_LOAD && accepted_weight) begin
                if (loaded_tiles == 23)
                    state <= STATE_STREAM;
                else
                    loaded_tiles <= loaded_tiles + 1'b1;
            end

            if (accepted_normalized_read) begin
                if (ARRAY_BACKPRESSURE)
                    read_pending <= 1'b1;
                response_group <= issue_group;
                response_input_tile <= issue_input_tile;
                if (issue_input_tile == 23) begin
                    issue_input_tile <= 0;
                    if (issue_group == 15)
                        state <= STATE_DRAIN;
                    else
                        issue_group <= issue_group + 1'b1;
                end else begin
                    issue_input_tile <= issue_input_tile + 1'b1;
                end
            end
            if (ARRAY_BACKPRESSURE && accepted_array_request)
                read_pending <= 1'b0;

            case ({fifo_push, fifo_pop})
                2'b10: begin
                    fifo_data[fifo_write_pointer] <= requant_outputs;
                    fifo_group[fifo_write_pointer] <= requant_tag[3:0];
                    fifo_write_pointer <= ~fifo_write_pointer;
                    fifo_count <= fifo_count + 1'b1;
                end
                2'b01: begin
                    fifo_read_pointer <= ~fifo_read_pointer;
                    fifo_count <= fifo_count - 1'b1;
                end
                2'b11: begin
                    fifo_data[fifo_write_pointer] <= requant_outputs;
                    fifo_group[fifo_write_pointer] <= requant_tag[3:0];
                    fifo_write_pointer <= ~fifo_write_pointer;
                    fifo_read_pointer <= ~fifo_read_pointer;
                end
                default: begin end
            endcase

            if (fifo_pop) begin
                if (outputs_sent == 15) begin
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else begin
                    outputs_sent <= outputs_sent + 1'b1;
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && response_valid && !requant_ready)
            $error("streaming QKV requantizer overflow");
        if (rst_n && fifo_push && fifo_count == 2 && !fifo_pop)
            $error("streaming QKV output FIFO overflow");
`endif
    end

endmodule
