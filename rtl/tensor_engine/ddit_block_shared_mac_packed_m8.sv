`timescale 1ns/1ps

module ddit_block_shared_mac_packed_m8 #(
    parameter integer MLP_TAG_WIDTH = 16,
    parameter integer ATTENTION_PACKED = 0,
    parameter integer PHYSICAL_N_LANES = 6
) (
    input  wire clk,
    input  wire rst_n,
    input  wire mlp_phase,
    input  wire attention_request_valid,
    output wire attention_request_ready,
    input  wire attention_request_narrow_int8_mode,
    input  wire attention_request_clear,
    input  wire attention_request_last,
    input  wire [7:0] attention_request_tag,
    input  wire [4*32*18-1:0] attention_request_activations,
    input  wire [6*32*18-1:0] attention_request_weights,
    input  wire [8*32*8-1:0] attention_request_narrow_activations,
    input  wire [6*32*8-1:0] attention_request_narrow_weights,
    output wire attention_response_valid,
    output wire attention_response_narrow_int8_mode,
    output wire [7:0] attention_response_tag,
    output wire [4*6*48-1:0] attention_response_accumulators,
    output wire [8*6*32-1:0] attention_response_narrow_accumulators,
    input  wire mlp_request_valid,
    output wire mlp_request_ready,
    input  wire mlp_request_clear,
    input  wire mlp_request_last,
    input  wire [MLP_TAG_WIDTH-1:0] mlp_request_tag,
    input  wire [8*32*8-1:0] mlp_request_activations,
    input  wire [6*32*8-1:0] mlp_request_weights,
    output wire mlp_response_valid,
    output wire [MLP_TAG_WIDTH-1:0] mlp_response_tag,
    output wire [8*6*32-1:0] mlp_response_accumulators
);

    localparam integer PHYSICAL_TAG_WIDTH = MLP_TAG_WIDTH + 1;
    wire physical_request_ready;
    wire selected_valid = (mlp_phase
        ? mlp_request_valid : attention_request_valid)
        && physical_request_ready;
    wire selected_narrow = mlp_phase ? 1'b1
        : (ATTENTION_PACKED ? attention_request_narrow_int8_mode : 1'b0);
    wire selected_clear = mlp_phase
        ? mlp_request_clear : attention_request_clear;
    wire selected_last = mlp_phase
        ? mlp_request_last : attention_request_last;
    wire [PHYSICAL_TAG_WIDTH-1:0] selected_tag = mlp_phase
        ? {1'b1, mlp_request_tag}
        : {1'b0, {(MLP_TAG_WIDTH-8){1'b0}}, attention_request_tag};
    wire physical_response_valid;
    wire physical_response_mode;
    wire [PHYSICAL_TAG_WIDTH-1:0] physical_response_tag;
    wire [4*6*48-1:0] physical_attention_accumulators;
    wire [8*6*32-1:0] physical_narrow_accumulators;
    wire response_owner = physical_response_tag[PHYSICAL_TAG_WIDTH-1];
    wire [8*32*8-1:0] selected_narrow_activations = mlp_phase
        ? mlp_request_activations : attention_request_narrow_activations;
    wire [6*32*8-1:0] selected_narrow_weights = mlp_phase
        ? mlp_request_weights : attention_request_narrow_weights;

    assign attention_response_valid = physical_response_valid && !response_owner;
    assign attention_request_ready = !mlp_phase && physical_request_ready;
    assign attention_response_narrow_int8_mode = physical_response_mode;
    assign attention_response_tag = physical_response_tag[7:0];
    assign attention_response_accumulators = physical_attention_accumulators;
    assign attention_response_narrow_accumulators =
        physical_narrow_accumulators;
    assign mlp_response_valid = physical_response_valid && response_owner;
    assign mlp_request_ready = mlp_phase && physical_request_ready;
    assign mlp_response_tag = physical_response_tag[MLP_TAG_WIDTH-1:0];

    generate
        if (PHYSICAL_N_LANES == 6) begin : direct_six_lane_array
            assign physical_request_ready = 1'b1;
            mixed_precision_packed_m8_mac_tile_pipelined #(
                .N_LANES(6), .ATTENTION_ACC_WIDTH(48), .MLP_ACC_WIDTH(32),
                .TAG_WIDTH(PHYSICAL_TAG_WIDTH)
            ) physical_mac (
                .clk(clk), .rst_n(rst_n), .valid_in(selected_valid),
                .narrow_int8_mode(selected_narrow),
                .clear_accumulators(selected_clear),
                .last_k_tile(selected_last), .tag_in(selected_tag),
                .attention_activations_packed(attention_request_activations),
                .attention_weights_packed(attention_request_weights),
                .mlp_activations_packed(selected_narrow_activations),
                .mlp_weights_packed(selected_narrow_weights),
                .valid_out(physical_response_valid),
                .narrow_int8_mode_out(physical_response_mode),
                .tag_out(physical_response_tag),
                .attention_accumulators_packed(
                    physical_attention_accumulators
                ),
                .mlp_accumulators_packed(physical_narrow_accumulators)
            );
        end else begin : folded_two_lane_array
            ddit_block_mac_folded_n2 #(
                .TAG_WIDTH(PHYSICAL_TAG_WIDTH)
            ) physical_mac (
                .clk(clk), .rst_n(rst_n), .valid_in(selected_valid),
                .ready_out(physical_request_ready),
                .narrow_int8_mode(selected_narrow),
                .clear_accumulators(selected_clear),
                .last_k_tile(selected_last), .tag_in(selected_tag),
                .attention_activations_packed(attention_request_activations),
                .attention_weights_packed(attention_request_weights),
                .mlp_activations_packed(selected_narrow_activations),
                .mlp_weights_packed(selected_narrow_weights),
                .valid_out(physical_response_valid),
                .narrow_int8_mode_out(physical_response_mode),
                .tag_out(physical_response_tag),
                .attention_accumulators_packed(
                    physical_attention_accumulators
                ),
                .mlp_accumulators_packed(physical_narrow_accumulators)
            );
        end
    endgenerate

    assign mlp_response_accumulators = physical_narrow_accumulators;

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && attention_request_valid && mlp_request_valid)
            $error("attention and wide MLP requested the DDiT array together");
        if (rst_n && !mlp_phase && mlp_request_valid)
            $error("wide MLP requested the DDiT array outside its phase");
        if (rst_n && mlp_phase && attention_request_valid)
            $error("attention requested the DDiT array during wide MLP phase");
        if (rst_n && physical_response_valid && response_owner
            && !physical_response_mode)
            $error("DDiT MLP response returned in wide mode");
`endif
    end

    initial begin
        if (MLP_TAG_WIDTH < 8)
            $error("DDiT shared MAC requires an MLP tag at least eight bits wide");
        if (PHYSICAL_N_LANES != 6 && PHYSICAL_N_LANES != 2)
            $error("DDiT shared MAC physical N lanes must be six or two");
    end

endmodule

module ddit_block_mac_folded_n2 #(
    parameter integer TAG_WIDTH = 17,
    parameter integer MAX_REQUESTS = 32
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    output wire ready_out,
    input  wire narrow_int8_mode,
    input  wire clear_accumulators,
    input  wire last_k_tile,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [4*32*18-1:0] attention_activations_packed,
    input  wire [6*32*18-1:0] attention_weights_packed,
    input  wire [8*32*8-1:0] mlp_activations_packed,
    input  wire [6*32*8-1:0] mlp_weights_packed,
    output reg  valid_out,
    output reg  narrow_int8_mode_out,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output reg  [4*6*48-1:0] attention_accumulators_packed,
    output reg  [8*6*32-1:0] mlp_accumulators_packed
);

    localparam [1:0] STATE_CAPTURE = 2'd0;
    localparam [1:0] STATE_EXECUTE = 2'd1;
    localparam [1:0] STATE_WAIT = 2'd2;
    localparam integer ADDRESS_WIDTH = $clog2(MAX_REQUESTS);

    reg [1:0] state;
    reg [ADDRESS_WIDTH:0] capture_count;
    reg [ADDRESS_WIDTH:0] burst_length;
    reg [ADDRESS_WIDTH:0] execute_index;
    reg [1:0] execute_slice;
    reg burst_mode;
    reg [TAG_WIDTH-1:0] burst_tag;

    reg [4*32*18-1:0] activation_memory [0:MAX_REQUESTS-1];
    reg [6*32*18-1:0] weight_memory [0:MAX_REQUESTS-1];

    wire physical_valid = state == STATE_EXECUTE;
    assign ready_out = state == STATE_CAPTURE;
    wire physical_clear = execute_index == 0;
    wire physical_last = execute_index + 1'b1 == burst_length;
    wire [4*32*18-1:0] replay_activations =
        activation_memory[execute_index[ADDRESS_WIDTH-1:0]];
    wire [6*32*18-1:0] replay_weights =
        weight_memory[execute_index[ADDRESS_WIDTH-1:0]];
    reg [2*32*18-1:0] physical_attention_weights;
    reg [2*32*8-1:0] physical_mlp_weights;
    wire physical_response_valid;
    wire physical_response_mode;
    wire [TAG_WIDTH-1:0] physical_response_tag;
    wire [4*2*48-1:0] physical_attention_accumulators;
    wire [8*2*32-1:0] physical_mlp_accumulators;

    integer m_lane;
    integer n_lane;

    always @* begin
        physical_attention_weights = replay_weights[2*32*18-1:0];
        physical_mlp_weights = replay_weights[2*32*8-1:0];
        case (execute_slice)
            0: begin
                physical_attention_weights = replay_weights[0 +: 2*32*18];
                physical_mlp_weights = replay_weights[0 +: 2*32*8];
            end
            1: begin
                physical_attention_weights = replay_weights[
                    2*32*18 +: 2*32*18
                ];
                physical_mlp_weights = replay_weights[2*32*8 +: 2*32*8];
            end
            default: begin
                physical_attention_weights = replay_weights[
                    4*32*18 +: 2*32*18
                ];
                physical_mlp_weights = replay_weights[4*32*8 +: 2*32*8];
            end
        endcase
    end

    mixed_precision_packed_m8_mac_tile_pipelined #(
        .N_LANES(2), .ATTENTION_ACC_WIDTH(48), .MLP_ACC_WIDTH(32),
        .TAG_WIDTH(TAG_WIDTH)
    ) physical_mac (
        .clk(clk), .rst_n(rst_n), .valid_in(physical_valid),
        .narrow_int8_mode(burst_mode),
        .clear_accumulators(physical_clear), .last_k_tile(physical_last),
        .tag_in(burst_tag),
        .attention_activations_packed(replay_activations),
        .attention_weights_packed(physical_attention_weights),
        .mlp_activations_packed(replay_activations[8*32*8-1:0]),
        .mlp_weights_packed(physical_mlp_weights),
        .valid_out(physical_response_valid),
        .narrow_int8_mode_out(physical_response_mode),
        .tag_out(physical_response_tag),
        .attention_accumulators_packed(physical_attention_accumulators),
        .mlp_accumulators_packed(physical_mlp_accumulators)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_CAPTURE;
            capture_count <= 0;
            burst_length <= 0;
            execute_index <= 0;
            execute_slice <= 0;
            burst_mode <= 0;
            burst_tag <= 0;
            valid_out <= 0;
            narrow_int8_mode_out <= 0;
            tag_out <= 0;
            attention_accumulators_packed <= 0;
            mlp_accumulators_packed <= 0;
        end else begin
            valid_out <= 1'b0;
            if (state == STATE_CAPTURE && valid_in) begin
                activation_memory[capture_count[ADDRESS_WIDTH-1:0]] <=
                    narrow_int8_mode
                    ? {{(4*32*18-8*32*8){1'b0}}, mlp_activations_packed}
                    : attention_activations_packed;
                weight_memory[capture_count[ADDRESS_WIDTH-1:0]] <=
                    narrow_int8_mode
                    ? {{(6*32*18-6*32*8){1'b0}}, mlp_weights_packed}
                    : attention_weights_packed;
                burst_mode <= narrow_int8_mode;
                burst_tag <= tag_in;
                if (last_k_tile) begin
                    burst_length <= capture_count + 1'b1;
                    capture_count <= 0;
                    execute_index <= 0;
                    execute_slice <= 0;
                    state <= STATE_EXECUTE;
                end else begin
                    capture_count <= capture_count + 1'b1;
                end
            end else if (state == STATE_EXECUTE) begin
                if (physical_last) begin
                    state <= STATE_WAIT;
                end else begin
                    execute_index <= execute_index + 1'b1;
                end
            end else if (state == STATE_WAIT && physical_response_valid) begin
                case (execute_slice)
                    0: begin
                        for (m_lane = 0; m_lane < 4; m_lane = m_lane + 1)
                            attention_accumulators_packed[
                                (m_lane*6)*48 +: 2*48
                            ] <= physical_attention_accumulators[
                                (m_lane*2)*48 +: 2*48
                            ];
                        for (m_lane = 0; m_lane < 8; m_lane = m_lane + 1)
                            mlp_accumulators_packed[
                                (m_lane*6)*32 +: 2*32
                            ] <= physical_mlp_accumulators[
                                (m_lane*2)*32 +: 2*32
                            ];
                    end
                    1: begin
                        for (m_lane = 0; m_lane < 4; m_lane = m_lane + 1)
                            attention_accumulators_packed[
                                (m_lane*6+2)*48 +: 2*48
                            ] <= physical_attention_accumulators[
                                (m_lane*2)*48 +: 2*48
                            ];
                        for (m_lane = 0; m_lane < 8; m_lane = m_lane + 1)
                            mlp_accumulators_packed[
                                (m_lane*6+2)*32 +: 2*32
                            ] <= physical_mlp_accumulators[
                                (m_lane*2)*32 +: 2*32
                            ];
                    end
                    default: begin
                        for (m_lane = 0; m_lane < 4; m_lane = m_lane + 1)
                            attention_accumulators_packed[
                                (m_lane*6+4)*48 +: 2*48
                            ] <= physical_attention_accumulators[
                                (m_lane*2)*48 +: 2*48
                            ];
                        for (m_lane = 0; m_lane < 8; m_lane = m_lane + 1)
                            mlp_accumulators_packed[
                                (m_lane*6+4)*32 +: 2*32
                            ] <= physical_mlp_accumulators[
                                (m_lane*2)*32 +: 2*32
                            ];
                    end
                endcase
                if (execute_slice == 2) begin
                    valid_out <= 1'b1;
                    narrow_int8_mode_out <= physical_response_mode;
                    tag_out <= physical_response_tag;
                    state <= STATE_CAPTURE;
                end else begin
                    execute_slice <= execute_slice + 1'b1;
                    execute_index <= 0;
                    state <= STATE_EXECUTE;
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && state != STATE_CAPTURE && valid_in)
            $error("folded N2 array received a new burst before its response");
        if (rst_n && state == STATE_CAPTURE && valid_in
            && capture_count == MAX_REQUESTS)
            $error("folded N2 array request burst exceeded its buffer");
        if (rst_n && state == STATE_CAPTURE && valid_in
            && capture_count == 0 && !clear_accumulators)
            $error("folded N2 array burst did not begin with clear");
        if (rst_n && state == STATE_CAPTURE && valid_in
            && capture_count != 0 && clear_accumulators)
            $error("folded N2 array clear appeared inside a burst");
`endif
    end

endmodule
