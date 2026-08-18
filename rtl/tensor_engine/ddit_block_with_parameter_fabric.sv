`timescale 1ns/1ps

module ddit_block_with_parameter_fabric #(
    parameter integer HEADS=12,
    parameter integer ATTENTION_OUTPUT_TILES=128,
    parameter integer TOKENS=64,
    parameter integer DOWN_INPUT_SIZE=3072,
    parameter integer DOWN_OUTPUT_SIZE=768,
    parameter integer OUTPUT_TILE_TAG_WIDTH=10,
    parameter integer INTERNAL_NORM1=0,
    parameter integer PACKED_ATTENTION=0,
    parameter integer PHYSICAL_N_LANES=6,
    parameter integer MLP_M_LANES=8,
    parameter integer GROUP_WIDTH=((TOKENS/4)<=1)?1:$clog2(TOKENS/4),
    parameter integer DOWN_K_TILE_WIDTH=((DOWN_INPUT_SIZE/32)<=1)
        ? 1 : $clog2(DOWN_INPUT_SIZE/32),
    parameter LUT_FILE="rtl/tensor_engine/exp_neg_q16_lut.hex"
)(
    input wire clk,input wire rst_n,input wire [63:0] block_base_address,
    input wire block_start,
    output wire block_start_ready,output wire busy,output wire done,
    input wire residual_load_valid,input wire [3:0] residual_load_group,
    input wire [6:0] residual_load_output_tile,
    input wire [4*6*24-1:0] residual_load_q10_packed,
    output wire normalized_read_valid,output wire [3:0] normalized_read_group,
    output wire [4:0] normalized_read_input_tile,
    input wire normalized_read_data_valid,input wire [4*32*18-1:0] normalized_q12_packed,
    input wire constant_load_valid,input wire [5:0] constant_load_token,
    input wire [4:0] constant_load_pair,
    input wire signed [15:0] constant_load_cosine_q15,
    input wire signed [15:0] constant_load_sine_q15,
    input wire [17:0] smoothing_reciprocal_q15,
    output wire [9:0] smoothing_reciprocal_channel,
    output wire output_valid,output wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile,
    output wire [GROUP_WIDTH-1:0] output_group,
    output wire [4*6*24-1:0] outputs_packed,
    output wire [63:0] m_axi_araddr,output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,output wire [1:0] m_axi_arburst,
    output wire m_axi_arvalid,input wire m_axi_arready,
    input wire [511:0] m_axi_rdata,input wire [1:0] m_axi_rresp,
    input wire m_axi_rlast,input wire m_axi_rvalid,output wire m_axi_rready,
    output wire parameter_protocol_error,output wire [255:0] client_bytes_read,
    output wire [63:0] read_transactions,
    output wire attention_busy,output wire mlp_busy
);
    wire qkv_meta_valid,qkv_meta_ready,qkv_weight_valid,qkv_weight_ready;
    wire qkv_parameter_request_valid;
    wire [3:0] qkv_meta_head,qkv_meta_channel,qkv_weight_head,qkv_weight_channel;
    wire [1:0] qkv_meta_kind,qkv_weight_kind;wire [4:0] qkv_weight_k;
    wire [143:0] qkv_multipliers;wire [107:0] qkv_biases;
    wire [3071:0] qkv_weights;wire [3:0] requested_qkv_head;
    wire [1:0] requested_qkv_kind;wire [3:0] requested_qkv_channel;
    wire [11:0] requested_qkv_row;
    wire projection_meta_valid,projection_meta_ready;
    wire projection_parameter_request_valid;
    wire projection_weight_valid,projection_weight_ready;
    wire [6:0] projection_meta_tile,projection_weight_tile;
    wire [4:0] projection_weight_k;
    wire [143:0] projection_multipliers;wire [1535:0] projection_weights;
    wire [6:0] requested_projection_tile;
    wire [9:0] requested_up_tile,requested_down_tile;
    wire requested_up_bank,requested_down_bank;
    wire up_meta_valid,up_meta_ready,up_weight_valid,up_weight_ready;
    wire down_meta_valid,down_meta_ready,down_weight_valid,down_weight_ready;
    wire [443:0] up_meta_data;wire [1343:0] down_meta_data;
    wire [1535:0] up_weight_data,down_weight_data;
    wire [4:0] up_weight_k;wire [DOWN_K_TILE_WIDTH-1:0] down_weight_k;
    wire [3:0] loader_busy;
    wire attention_tile_valid;wire [3:0] attention_tile_group;
    wire [6:0] attention_tile_output;wire [575:0] attention_tile_data;

    ddit_block_pipeline #(.HEADS(HEADS),
        .ATTENTION_OUTPUT_TILES(ATTENTION_OUTPUT_TILES),.TOKENS(TOKENS),
        .DOWN_INPUT_SIZE(DOWN_INPUT_SIZE),.DOWN_OUTPUT_SIZE(DOWN_OUTPUT_SIZE),
        .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
        .INTERNAL_NORM1(INTERNAL_NORM1),
        .PACKED_ATTENTION(PACKED_ATTENTION),
        .PHYSICAL_N_LANES(PHYSICAL_N_LANES),.MLP_M_LANES(MLP_M_LANES),
        .GROUP_WIDTH(GROUP_WIDTH),
        .LUT_FILE(LUT_FILE)) compute(
        .clk(clk),.rst_n(rst_n),.block_start(block_start),
        .block_start_ready(block_start_ready),.busy(busy),.done(done),
        .residual_load_valid(residual_load_valid),.residual_load_group(residual_load_group),
        .residual_load_output_tile(residual_load_output_tile),
        .residual_load_q10_packed(residual_load_q10_packed),
        .qkv_metadata_valid(qkv_meta_valid),.qkv_metadata_ready(qkv_meta_ready),
        .qkv_parameter_request_valid(qkv_parameter_request_valid),
        .qkv_metadata_head(qkv_meta_head),.qkv_metadata_kind(qkv_meta_kind),
        .qkv_metadata_channel_tile(qkv_meta_channel),
        .qkv_metadata_multipliers_packed(qkv_multipliers),
        .qkv_metadata_biases_q12_packed(qkv_biases),
        .qkv_weight_tile_valid(qkv_weight_valid),.qkv_weight_tile_ready(qkv_weight_ready),
        .qkv_weight_head(qkv_weight_head),.qkv_weight_kind(qkv_weight_kind),
        .qkv_weight_channel_tile(qkv_weight_channel),
        .qkv_weight_input_tile(qkv_weight_k),.qkv_weight_int16_packed(qkv_weights),
        .requested_qkv_head(requested_qkv_head),.requested_qkv_kind(requested_qkv_kind),
        .requested_qkv_channel_tile(requested_qkv_channel),
        .requested_qkv_global_row(requested_qkv_row),
        .normalized_read_valid(normalized_read_valid),
        .normalized_read_group(normalized_read_group),
        .normalized_read_input_tile(normalized_read_input_tile),
        .normalized_read_data_valid(normalized_read_data_valid),
        .normalized_q12_packed(normalized_q12_packed),
        .constant_load_valid(constant_load_valid),.constant_load_token(constant_load_token),
        .constant_load_pair(constant_load_pair),.constant_load_cosine_q15(constant_load_cosine_q15),
        .constant_load_sine_q15(constant_load_sine_q15),
        .projection_metadata_valid(projection_meta_valid),
        .projection_metadata_ready(projection_meta_ready),
        .projection_parameter_request_valid(projection_parameter_request_valid),
        .projection_metadata_output_tile(projection_meta_tile),
        .projection_metadata_multipliers_packed(projection_multipliers),
        .projection_weight_tile_valid(projection_weight_valid),
        .projection_weight_tile_ready(projection_weight_ready),
        .projection_weight_output_tile(projection_weight_tile),
        .projection_weight_input_tile(projection_weight_k),
        .projection_weight_int8_packed(projection_weights),
        .requested_projection_output_tile(requested_projection_tile),
        .attention_tile_valid(attention_tile_valid),
        .attention_tile_group(attention_tile_group),
        .attention_tile_output_tile(attention_tile_output),
        .attention_tile_q10_packed(attention_tile_data),
        .smoothing_reciprocal_q15(smoothing_reciprocal_q15),
        .smoothing_reciprocal_channel(smoothing_reciprocal_channel),
        .requested_up_output_tile(requested_up_tile),.requested_up_bank(requested_up_bank),
        .up_weight_stream_valid(up_weight_valid),.up_weight_stream_ready(up_weight_ready),
        .up_weight_stream_data(up_weight_data),.up_metadata_stream_valid(up_meta_valid),
        .up_metadata_stream_ready(up_meta_ready),.up_metadata_stream_data(up_meta_data),
        .requested_down_output_tile(requested_down_tile),
        .requested_down_bank(requested_down_bank),
        .down_weight_stream_valid(down_weight_valid),
        .down_weight_stream_ready(down_weight_ready),.down_weight_stream_data(down_weight_data),
        .down_metadata_stream_valid(down_meta_valid),
        .down_metadata_stream_ready(down_meta_ready),.down_metadata_stream_data(down_meta_data),
        .output_valid(output_valid),.output_tile(output_tile),.output_group(output_group),
        .outputs_packed(outputs_packed),.attention_busy(attention_busy),.mlp_busy(mlp_busy));

    mdlm_block_parameter_load_fabric #(.DOWN_INPUT_SIZE(DOWN_INPUT_SIZE)) fabric(
        .clk(clk),.rst_n(rst_n),.block_base_address(block_base_address),
        .qkv_command_valid(qkv_parameter_request_valid),
        .qkv_command_head(requested_qkv_head),.qkv_command_kind(requested_qkv_kind),
        .qkv_command_channel_tile(requested_qkv_channel),
        .qkv_metadata_valid(qkv_meta_valid),.qkv_metadata_ready(qkv_meta_ready),
        .qkv_metadata_head(qkv_meta_head),.qkv_metadata_kind(qkv_meta_kind),
        .qkv_metadata_channel_tile(qkv_meta_channel),
        .qkv_metadata_multipliers(qkv_multipliers),.qkv_metadata_biases(qkv_biases),
        .qkv_weight_valid(qkv_weight_valid),.qkv_weight_ready(qkv_weight_ready),
        .qkv_weight_head(qkv_weight_head),.qkv_weight_kind(qkv_weight_kind),
        .qkv_weight_channel_tile(qkv_weight_channel),
        .qkv_weight_input_tile(qkv_weight_k),.qkv_weight_data(qkv_weights),
        .projection_command_valid(projection_parameter_request_valid),
        .projection_command_output_tile(requested_projection_tile),
        .projection_metadata_valid(projection_meta_valid),
        .projection_metadata_ready(projection_meta_ready),
        .projection_metadata_output_tile(projection_meta_tile),
        .projection_metadata_multipliers(projection_multipliers),
        .projection_weight_valid(projection_weight_valid),
        .projection_weight_ready(projection_weight_ready),
        .projection_weight_output_tile(projection_weight_tile),
        .projection_weight_input_tile(projection_weight_k),
        .projection_weight_data(projection_weights),
        .up_command_valid(mlp_busy && up_meta_ready),.up_command_output_tile(requested_up_tile),
        .up_metadata_valid(up_meta_valid),.up_metadata_ready(up_meta_ready),
        .up_metadata_data(up_meta_data),.up_weight_valid(up_weight_valid),
        .up_weight_ready(up_weight_ready),.up_weight_data(up_weight_data),
        .up_weight_k_tile(up_weight_k),
        .down_command_valid(mlp_busy && down_meta_ready),
        .down_command_output_tile(requested_down_tile),
        .down_metadata_valid(down_meta_valid),.down_metadata_ready(down_meta_ready),
        .down_metadata_data(down_meta_data),.down_weight_valid(down_weight_valid),
        .down_weight_ready(down_weight_ready),.down_weight_data(down_weight_data),
        .down_weight_k_tile(down_weight_k),
        .m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),.m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),.client_busy(loader_busy),
        .protocol_error(parameter_protocol_error),.client_bytes_read(client_bytes_read),
        .read_transactions(read_transactions));
endmodule
