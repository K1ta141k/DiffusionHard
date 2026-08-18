`timescale 1ns/1ps

module ddit_block_shared_mac #(
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer MLP_TAG_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,
    input  wire mlp_phase,

    input  wire attention_request_valid,
    input  wire attention_request_clear,
    input  wire attention_request_last,
    input  wire [7:0] attention_request_tag,
    input  wire [M_LANES*32*18-1:0] attention_request_activations,
    input  wire [N_LANES*32*18-1:0] attention_request_weights,
    output wire attention_response_valid,
    output wire [7:0] attention_response_tag,
    output wire [M_LANES*N_LANES*48-1:0]
        attention_response_accumulators,

    input  wire mlp_request_valid,
    input  wire mlp_request_clear,
    input  wire mlp_request_last,
    input  wire [MLP_TAG_WIDTH-1:0] mlp_request_tag,
    input  wire [M_LANES*32*8-1:0] mlp_request_activations,
    input  wire [N_LANES*32*8-1:0] mlp_request_weights,
    output wire mlp_response_valid,
    output wire [MLP_TAG_WIDTH-1:0] mlp_response_tag,
    output wire [M_LANES*N_LANES*32-1:0] mlp_response_accumulators
);

    localparam integer PHYSICAL_TAG_WIDTH = MLP_TAG_WIDTH + 1;
    localparam integer LANES = M_LANES * N_LANES;

    wire selected_valid = mlp_phase
        ? mlp_request_valid : attention_request_valid;
    wire selected_clear = mlp_phase
        ? mlp_request_clear : attention_request_clear;
    wire selected_last = mlp_phase
        ? mlp_request_last : attention_request_last;
    wire [PHYSICAL_TAG_WIDTH-1:0] selected_tag = mlp_phase
        ? {1'b1, mlp_request_tag}
        : {1'b0, {(MLP_TAG_WIDTH-8){1'b0}}, attention_request_tag};
    wire [M_LANES*32*18-1:0] selected_activations;
    wire [N_LANES*32*18-1:0] selected_weights;
    wire physical_response_valid;
    wire [PHYSICAL_TAG_WIDTH-1:0] physical_response_tag;
    wire [LANES*48-1:0] physical_response_accumulators;
    wire response_owner = physical_response_tag[PHYSICAL_TAG_WIDTH-1];

    genvar activation_index;
    generate
        for (activation_index = 0; activation_index < M_LANES*32;
             activation_index = activation_index + 1) begin : widen_activation
            assign selected_activations[activation_index*18 +: 18] = mlp_phase
                ? {{10{1'b0}}, mlp_request_activations[
                    activation_index*8 +: 8
                ]}
                : attention_request_activations[activation_index*18 +: 18];
        end
    endgenerate

    genvar weight_index;
    generate
        for (weight_index = 0; weight_index < N_LANES*32;
             weight_index = weight_index + 1) begin : widen_weight
            assign selected_weights[weight_index*18 +: 18] = mlp_phase
                ? {{10{1'b0}}, mlp_request_weights[weight_index*8 +: 8]}
                : attention_request_weights[weight_index*18 +: 18];
        end
    endgenerate

    assign attention_response_valid = physical_response_valid
        && !response_owner;
    assign attention_response_tag = physical_response_tag[7:0];
    assign attention_response_accumulators = physical_response_accumulators;
    assign mlp_response_valid = physical_response_valid && response_owner;
    assign mlp_response_tag = physical_response_tag[MLP_TAG_WIDTH-1:0];

    genvar lane_index;
    generate
        for (lane_index = 0; lane_index < LANES;
             lane_index = lane_index + 1) begin : narrow_accumulator
            assign mlp_response_accumulators[lane_index*32 +: 32] =
                physical_response_accumulators[lane_index*48 +: 32];
        end
    endgenerate

    mixed_precision_mac_tile_pipelined #(
        .M_LANES(M_LANES), .N_LANES(N_LANES),
        .TAG_WIDTH(PHYSICAL_TAG_WIDTH)
    ) physical_mac (
        .clk(clk), .rst_n(rst_n), .valid_in(selected_valid),
        .narrow_int8_mode(mlp_phase),
        .clear_accumulators(selected_clear), .last_k_tile(selected_last),
        .tag_in(selected_tag), .activations_packed(selected_activations),
        .weights_packed(selected_weights),
        .valid_out(physical_response_valid), .tag_out(physical_response_tag),
        .accumulators_packed(physical_response_accumulators)
    );

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && attention_request_valid && mlp_request_valid)
            $error("attention and MLP requested the DDiT array together");
        if (rst_n && !mlp_phase && mlp_request_valid)
            $error("MLP requested the DDiT array outside its phase");
        if (rst_n && mlp_phase && attention_request_valid)
            $error("attention requested the DDiT array during MLP phase");
`endif
    end

    initial begin
        if (MLP_TAG_WIDTH < 8)
            $error("DDiT shared MAC requires an MLP tag at least eight bits wide");
    end

endmodule
