`timescale 1ns/1ps

module ddit_block_with_image_fabric #(
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
    parameter LUT_FILE="rtl/tensor_engine/exp_neg_q16_lut.hex"
)(
    input wire clk,input wire rst_n,input wire [63:0] block_base_address,
    input wire preload_start,output wire preload_start_ready,
    output wire preload_done,output reg constants_loaded,
    input wire block_start,output wire block_start_ready,
    output wire busy,output wire done,
    input wire residual_load_valid,input wire [3:0] residual_load_group,
    input wire [6:0] residual_load_output_tile,
    input wire [4*6*24-1:0] residual_load_q10_packed,
    output wire normalized_read_valid,output wire [3:0] normalized_read_group,
    output wire [4:0] normalized_read_input_tile,
    input wire normalized_read_data_valid,input wire [4*32*18-1:0] normalized_q12_packed,
    output wire output_valid,output wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile,
    output wire [GROUP_WIDTH-1:0] output_group,
    output wire [4*6*24-1:0] outputs_packed,
    output wire [63:0] m_axi_araddr,output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,output wire [1:0] m_axi_arburst,
    output wire m_axi_arvalid,input wire m_axi_arready,
    input wire [511:0] m_axi_rdata,input wire [1:0] m_axi_rresp,
    input wire m_axi_rlast,input wire m_axi_rvalid,output wire m_axi_rready,
    output wire protocol_error,output wire [63:0] read_transactions,
    output wire [63:0] constant_read_transactions,
    output wire [63:0] constant_bytes_read,
    output wire [255:0] dense_client_bytes_read,
    output wire [63:0] dense_read_transactions,
    output wire attention_busy,output wire mlp_busy
);
    wire preloader_ready,preloader_busy,preloader_error;
    wire rotary_valid;wire [5:0] rotary_token;wire [4:0] rotary_pair;
    wire signed [15:0] rotary_cosine,rotary_sine;
    wire [9:0] reciprocal_channel;wire [17:0] reciprocal_q15;
    wire [63:0] constant_araddr,dense_araddr;wire [7:0] constant_arlen,dense_arlen;
    wire [2:0] constant_arsize,dense_arsize;wire [1:0] constant_arburst,dense_arburst;
    wire constant_arvalid,dense_arvalid,constant_arready,dense_arready;
    wire constant_rvalid,dense_rvalid,constant_rready,dense_rready;
    wire dense_error,dense_start_ready,dense_busy;
    wire [255:0] client_araddr;wire [31:0] client_arlen;
    wire [11:0] client_arsize;wire [7:0] client_arburst;
    wire [3:0] client_arvalid,client_arready,client_rvalid,client_rready;
    wire [511:0] client_rdata;wire [1:0] client_rresp;wire client_rlast;
    wire arbiter_error;wire [63:0] arbitration_cycles;
    wire start_preloader=preload_start && preload_start_ready;

    assign preload_start_ready=preloader_ready && !dense_busy;
    assign block_start_ready=constants_loaded && dense_start_ready;
    assign busy=dense_busy || preloader_busy;
    assign protocol_error=preloader_error || dense_error || arbiter_error;
    always @(posedge clk) begin
        if(!rst_n) constants_loaded<=1'b0;
        else begin
            if(start_preloader) constants_loaded<=1'b0;
            if(preload_done) constants_loaded<=1'b1;
        end
    end

    mdlm_block_constant_preloader preloader(
      .clk(clk),.rst_n(rst_n),.start(start_preloader),.start_ready(preloader_ready),
      .block_base_address(block_base_address),.rotary_load_valid(rotary_valid),
      .rotary_load_token(rotary_token),.rotary_load_pair(rotary_pair),
      .rotary_load_cosine_q15(rotary_cosine),.rotary_load_sine_q15(rotary_sine),
      .reciprocal_lookup_channel(reciprocal_channel),
      .reciprocal_lookup_q15(reciprocal_q15),.m_axi_araddr(constant_araddr),
      .m_axi_arlen(constant_arlen),.m_axi_arsize(constant_arsize),
      .m_axi_arburst(constant_arburst),.m_axi_arvalid(constant_arvalid),
      .m_axi_arready(constant_arready),.m_axi_rdata(client_rdata),
      .m_axi_rresp(client_rresp),.m_axi_rlast(client_rlast),
      .m_axi_rvalid(constant_rvalid),.m_axi_rready(constant_rready),
      .busy(preloader_busy),.done(preload_done),.protocol_error(preloader_error),
      .bytes_read(constant_bytes_read),.read_transactions(constant_read_transactions));

    ddit_block_with_parameter_fabric #(.HEADS(HEADS),
      .ATTENTION_OUTPUT_TILES(ATTENTION_OUTPUT_TILES),.TOKENS(TOKENS),
      .DOWN_INPUT_SIZE(DOWN_INPUT_SIZE),.DOWN_OUTPUT_SIZE(DOWN_OUTPUT_SIZE),
      .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),
      .INTERNAL_NORM1(INTERNAL_NORM1),
      .PACKED_ATTENTION(PACKED_ATTENTION),
      .PHYSICAL_N_LANES(PHYSICAL_N_LANES),.MLP_M_LANES(MLP_M_LANES),
      .GROUP_WIDTH(GROUP_WIDTH),
      .LUT_FILE(LUT_FILE)) dense(
      .clk(clk),.rst_n(rst_n),.block_base_address(block_base_address),
      .block_start(block_start && block_start_ready),
      .block_start_ready(dense_start_ready),.busy(dense_busy),.done(done),
      .residual_load_valid(residual_load_valid),.residual_load_group(residual_load_group),
      .residual_load_output_tile(residual_load_output_tile),
      .residual_load_q10_packed(residual_load_q10_packed),
      .normalized_read_valid(normalized_read_valid),
      .normalized_read_group(normalized_read_group),
      .normalized_read_input_tile(normalized_read_input_tile),
      .normalized_read_data_valid(normalized_read_data_valid),
      .normalized_q12_packed(normalized_q12_packed),
      .constant_load_valid(rotary_valid),.constant_load_token(rotary_token),
      .constant_load_pair(rotary_pair),.constant_load_cosine_q15(rotary_cosine),
      .constant_load_sine_q15(rotary_sine),
      .smoothing_reciprocal_q15(reciprocal_q15),
      .smoothing_reciprocal_channel(reciprocal_channel),
      .output_valid(output_valid),.output_tile(output_tile),
      .output_group(output_group),.outputs_packed(outputs_packed),
      .m_axi_araddr(dense_araddr),.m_axi_arlen(dense_arlen),
      .m_axi_arsize(dense_arsize),.m_axi_arburst(dense_arburst),
      .m_axi_arvalid(dense_arvalid),.m_axi_arready(dense_arready),
      .m_axi_rdata(client_rdata),.m_axi_rresp(client_rresp),
      .m_axi_rlast(client_rlast),.m_axi_rvalid(dense_rvalid),
      .m_axi_rready(dense_rready),.parameter_protocol_error(dense_error),
      .client_bytes_read(dense_client_bytes_read),
      .read_transactions(dense_read_transactions),
      .attention_busy(attention_busy),.mlp_busy(mlp_busy));

    assign client_araddr[0+:64]=constant_araddr;
    assign client_araddr[64+:64]=dense_araddr;
    assign client_araddr[128+:128]=0;
    assign client_arlen[0+:8]=constant_arlen;assign client_arlen[8+:8]=dense_arlen;
    assign client_arlen[16+:16]=0;
    assign client_arsize[0+:3]=constant_arsize;
    assign client_arsize[3+:3]=dense_arsize;assign client_arsize[6+:6]=0;
    assign client_arburst[0+:2]=constant_arburst;
    assign client_arburst[2+:2]=dense_arburst;assign client_arburst[4+:4]=0;
    assign client_arvalid={2'b00,dense_arvalid,constant_arvalid};
    assign constant_arready=client_arready[0];assign dense_arready=client_arready[1];
    assign constant_rvalid=client_rvalid[0];assign dense_rvalid=client_rvalid[1];
    assign client_rready={2'b00,dense_rready,constant_rready};
    axi512_read_arbiter_4 outer_arbiter(.clk(clk),.rst_n(rst_n),
      .s_axi_araddr(client_araddr),.s_axi_arlen(client_arlen),
      .s_axi_arsize(client_arsize),.s_axi_arburst(client_arburst),
      .s_axi_arvalid(client_arvalid),.s_axi_arready(client_arready),
      .s_axi_rdata(client_rdata),.s_axi_rresp(client_rresp),
      .s_axi_rlast(client_rlast),.s_axi_rvalid(client_rvalid),
      .s_axi_rready(client_rready),.m_axi_araddr(m_axi_araddr),
      .m_axi_arlen(m_axi_arlen),.m_axi_arsize(m_axi_arsize),
      .m_axi_arburst(m_axi_arburst),.m_axi_arvalid(m_axi_arvalid),
      .m_axi_arready(m_axi_arready),.m_axi_rdata(m_axi_rdata),
      .m_axi_rresp(m_axi_rresp),.m_axi_rlast(m_axi_rlast),
      .m_axi_rvalid(m_axi_rvalid),.m_axi_rready(m_axi_rready),
      .busy(),.protocol_error(arbiter_error),.read_transactions(read_transactions),
      .arbitration_cycles(arbitration_cycles));
endmodule
