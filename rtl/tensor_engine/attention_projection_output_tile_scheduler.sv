`timescale 1ns/1ps

module attention_projection_output_tile_scheduler #(
    parameter integer DATA_WIDTH = 18,
    parameter integer ACC_WIDTH = 48,
    parameter integer OUTPUT_WIDTH = 24,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer INTERNAL_MAC = 1,
    parameter integer GROUPED_CANVAS = 0,
    parameter integer ARRAY_BACKPRESSURE = 0
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output wire start_ready,
    input  wire [6:0] output_tile_in,
    input  wire [6*MULTIPLIER_WIDTH-1:0] multipliers_packed,
    input  wire weight_tile_valid,
    output wire weight_tile_ready,
    input  wire [4:0] weight_input_tile,
    input  wire [6*32*8-1:0] weight_int8_packed,
    output wire canvas_read_valid,
    output wire [3:0] canvas_read_head,
    output wire [5:0] canvas_read_token,
    input  wire canvas_read_data_valid,
    input  wire [64*DATA_WIDTH-1:0] canvas_read_data_packed,
    output wire canvas_group_read_valid,
    output wire [3:0] canvas_group_read_head,
    output wire [3:0] canvas_group_read_group,
    input  wire canvas_group_read_data_valid,
    input  wire [4*64*DATA_WIDTH-1:0] canvas_group_read_data_packed,
    output wire projection_tile_valid,
    input  wire projection_tile_ready,
    output wire [3:0] projection_group,
    output wire [6:0] projection_output_tile,
    output wire [2:0] projection_valid_channels,
    output wire [4*6*OUTPUT_WIDTH-1:0] projection_q10_packed,
    output wire array_request_valid,
    input  wire array_request_ready,
    output wire array_request_clear,
    output wire array_request_last,
    output wire [7:0] array_request_tag,
    output wire [4*32*DATA_WIDTH-1:0] array_request_activations,
    output wire [6*32*DATA_WIDTH-1:0] array_request_weights,
    input  wire array_response_valid,
    input  wire [7:0] array_response_tag,
    input  wire [4*6*ACC_WIDTH-1:0] array_response_accumulators,
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
    reg [6:0] active_output_tile;
    reg [4:0] active_input_tile;
    reg [4:0] loaded_tiles;
    reg [3:0] active_group;
    reg [4:0] requests_sent;
    reg [2:0] responses_received;
    reg [1:0] response_lane;
    reg [4:0] response_input_tile;
    reg [4*32*DATA_WIDTH-1:0] activation_buffer;
    reg [6*MULTIPLIER_WIDTH-1:0] active_multipliers;
    reg [4*6*OUTPUT_WIDTH-1:0] projection_buffer;
    reg group_read_pending;

    wire [6*32*8-1:0] selected_weight_int8;
    wire [6*32*DATA_WIDTH-1:0] selected_weight_extended;
    wire [4*32*DATA_WIDTH-1:0] grouped_activations;
    wire [4*6*ACC_WIDTH-1:0] internal_mac_accumulators;
    wire internal_mac_output_valid;
    wire [7:0] internal_mac_output_tag;
    wire [4*6*ACC_WIDTH-1:0] mac_accumulators = INTERNAL_MAC
        ? internal_mac_accumulators : array_response_accumulators;
    wire mac_output_valid = INTERNAL_MAC
        ? internal_mac_output_valid : array_response_valid;
    wire [7:0] mac_output_tag = INTERNAL_MAC
        ? internal_mac_output_tag : array_response_tag;
    wire requant_ready;
    wire requant_valid;
    wire [7:0] requant_tag;
    wire [4*6*OUTPUT_WIDTH-1:0] requant_outputs;
    wire [4*6*MULTIPLIER_WIDTH-1:0] expanded_multipliers;
    wire accepted_weight = weight_tile_valid && weight_tile_ready;
    wire selected_array_ready = INTERNAL_MAC || !ARRAY_BACKPRESSURE
        || array_request_ready;
    wire accepted_array_request = array_request_valid
        && selected_array_ready;

    integer channel;
    genvar weight_lane;
    genvar token_lane;
    genvar output_lane;

    assign start_ready = (state == STATE_IDLE);
    assign weight_tile_ready = (state == STATE_LOAD)
        && weight_input_tile == loaded_tiles;
    assign canvas_read_valid = !GROUPED_CANVAS && (state == STATE_READ)
        && requests_sent < 4;
    assign canvas_read_head = active_input_tile[4:1];
    assign canvas_read_token = {active_group, requests_sent[1:0]};
    assign canvas_group_read_valid = GROUPED_CANVAS && state == STATE_READ
        && requests_sent < 24
        && (!ARRAY_BACKPRESSURE
            || (!group_read_pending && selected_array_ready));
    assign canvas_group_read_head = requests_sent[4:1];
    assign canvas_group_read_group = active_group;
    assign projection_tile_valid = (state == STATE_OUTPUT);
    assign projection_group = active_group;
    assign projection_output_tile = active_output_tile;
    assign projection_valid_channels = 3'd6;
    assign projection_q10_packed = projection_buffer;
    assign busy = (state != STATE_IDLE);
    assign array_request_valid = GROUPED_CANVAS
        ? state == STATE_READ && canvas_group_read_data_valid
        : state == STATE_ISSUE;
    assign array_request_clear = GROUPED_CANVAS
        ? response_input_tile == 0 : active_input_tile == 0;
    assign array_request_last = GROUPED_CANVAS
        ? response_input_tile == 23 : active_input_tile == 23;
    assign array_request_tag = {active_group,GROUPED_CANVAS
        ? response_input_tile[3:0]:active_input_tile[3:0]};
    assign array_request_activations = GROUPED_CANVAS
        ? grouped_activations : activation_buffer;
    assign array_request_weights = selected_weight_extended;

    generate
        for (weight_lane = 0; weight_lane < 6*32;
             weight_lane = weight_lane + 1) begin : extend_weights
            assign selected_weight_extended[
                weight_lane*DATA_WIDTH +: DATA_WIDTH
            ] = {
                {(DATA_WIDTH-8){selected_weight_int8[weight_lane*8+7]}},
                selected_weight_int8[weight_lane*8 +: 8]
            };
        end
        for (token_lane = 0; token_lane < 4;
             token_lane = token_lane + 1) begin : token_multipliers
            for (output_lane = 0; output_lane < 6;
                 output_lane = output_lane + 1) begin : output_multipliers
                assign expanded_multipliers[
                    (token_lane*6+output_lane)*MULTIPLIER_WIDTH
                    +: MULTIPLIER_WIDTH
                ] = active_multipliers[
                    output_lane*MULTIPLIER_WIDTH +: MULTIPLIER_WIDTH
                ];
            end
        end
        for (token_lane = 0; token_lane < 4;
             token_lane = token_lane + 1) begin : grouped_token_activations
            genvar grouped_channel;
            for(grouped_channel=0;grouped_channel<32;
                grouped_channel=grouped_channel+1)begin:grouped_channels
                assign grouped_activations[
                    (token_lane*32+grouped_channel)*DATA_WIDTH+:DATA_WIDTH
                ]=canvas_group_read_data_packed[
                    (token_lane*64+grouped_channel
                    +response_input_tile[0]*32)*DATA_WIDTH+:DATA_WIDTH];
            end
        end
    endgenerate

    attention_projection_weight_tile_buffer weight_buffer (
        .clk(clk), .write_valid(accepted_weight),
        .write_tile(weight_input_tile),
        .write_weights_packed(weight_int8_packed),
        .read_tile(GROUPED_CANVAS ? response_input_tile : active_input_tile),
        .read_weights_packed(selected_weight_int8)
    );

    generate
        if (INTERNAL_MAC) begin : internal_array
            mixed_precision_mac_tile_pipelined #(
                .M_LANES(4), .N_LANES(6), .STORAGE_WIDTH(DATA_WIDTH),
                .ACC_WIDTH(ACC_WIDTH), .TAG_WIDTH(8)
            ) projection_mac (
                .clk(clk), .rst_n(rst_n), .valid_in(array_request_valid),
                .narrow_int8_mode(1'b0),
                .clear_accumulators(array_request_clear),
                .last_k_tile(array_request_last),
                .tag_in(array_request_tag),
                .activations_packed(array_request_activations),
                .weights_packed(array_request_weights),
                .valid_out(internal_mac_output_valid),
                .tag_out(internal_mac_output_tag),
                .accumulators_packed(internal_mac_accumulators)
            );
        end else begin : no_internal_array
            assign internal_mac_output_valid = 1'b0;
            assign internal_mac_output_tag = 8'b0;
            assign internal_mac_accumulators = {4*6*ACC_WIDTH{1'b0}};
        end
    endgenerate

    fixed_requantize_vector_parallel #(
        .LANES(24), .ACC_WIDTH(ACC_WIDTH),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH), .RIGHT_SHIFT(24), .TAG_WIDTH(8)
    ) projection_requantizer (
        .clk(clk), .rst_n(rst_n), .valid_in(mac_output_valid),
        .ready_in(requant_ready), .tag_in({4'b0, active_group}),
        .accumulators_packed(mac_accumulators),
        .multipliers_packed(expanded_multipliers),
        .biases_packed({24*ACC_WIDTH{1'b0}}), .valid_out(requant_valid),
        .tag_out(requant_tag), .outputs_packed(requant_outputs)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_output_tile <= 0;
            active_input_tile <= 0;
            loaded_tiles <= 0;
            active_group <= 0;
            requests_sent <= 0;
            responses_received <= 0;
            response_lane <= 0;
            response_input_tile <= 0;
            activation_buffer <= 0;
            active_multipliers <= 0;
            projection_buffer <= 0;
            group_read_pending <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && start) begin
                state <= STATE_LOAD;
                active_output_tile <= output_tile_in;
                active_multipliers <= multipliers_packed;
                loaded_tiles <= 0;
                active_group <= 0;
            end else if (state == STATE_LOAD && accepted_weight) begin
                if (loaded_tiles == 23) begin
                    state <= STATE_READ;
                    active_input_tile <= 0;
                    requests_sent <= 0;
                    responses_received <= 0;
                end else begin
                    loaded_tiles <= loaded_tiles + 1'b1;
                end
            end else if (state == STATE_READ) begin
                if(GROUPED_CANVAS)begin
                    if(canvas_group_read_valid)begin
                        response_input_tile<=requests_sent;
                        requests_sent<=requests_sent+1'b1;
                        if (ARRAY_BACKPRESSURE)
                            group_read_pending<=1'b1;
                    end
                    if(ARRAY_BACKPRESSURE && accepted_array_request)
                        group_read_pending<=1'b0;
                    if(canvas_group_read_data_valid&&selected_array_ready
                        &&response_input_tile==23)
                        state<=STATE_WAIT_MAC;
                end else begin
                    if (canvas_read_valid) begin
                        response_lane <= requests_sent[1:0];
                        requests_sent <= requests_sent + 1'b1;
                    end
                    if (canvas_read_data_valid) begin
                        for (channel = 0; channel < 32; channel = channel + 1)
                            activation_buffer[
                                (response_lane*32+channel)*DATA_WIDTH +: DATA_WIDTH
                            ] <= canvas_read_data_packed[
                                (active_input_tile[0]*32+channel)*DATA_WIDTH
                                +: DATA_WIDTH
                            ];
                        if (responses_received == 3) begin
                            state <= STATE_ISSUE;
                        end else begin
                            responses_received <= responses_received + 1'b1;
                        end
                    end
                end
            end else if (state == STATE_ISSUE && selected_array_ready) begin
                if (active_input_tile == 23) begin
                    state <= STATE_WAIT_MAC;
                end else begin
                    active_input_tile <= active_input_tile + 1'b1;
                    requests_sent <= 0;
                    responses_received <= 0;
                    state <= STATE_READ;
                end
            end else if (state == STATE_WAIT_MAC && mac_output_valid) begin
                state <= STATE_WAIT_REQUANT;
            end else if (state == STATE_WAIT_REQUANT && requant_valid) begin
                projection_buffer <= requant_outputs;
                state <= STATE_OUTPUT;
            end else if (state == STATE_OUTPUT && projection_tile_ready) begin
                if (active_group == 15) begin
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else begin
                    active_group <= active_group + 1'b1;
                    active_input_tile <= 0;
                    requests_sent <= 0;
                    responses_received <= 0;
                    response_input_tile <= 0;
                    state <= STATE_READ;
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && start && !start_ready)
            $error("attention projection output tile start arrived while busy");
        if (rst_n && state == STATE_WAIT_MAC && mac_output_valid
            && !requant_ready)
            $error("attention projection requantizer queue overflow");
`endif
    end

endmodule
