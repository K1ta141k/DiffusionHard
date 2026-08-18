`timescale 1ns/1ps

module ddit_block_kv260_kernel_core #(
    parameter integer HEADS=12,
    parameter integer ATTENTION_OUTPUT_TILES=128,
    parameter integer TOKENS=64,
    parameter integer DOWN_INPUT_SIZE=3072,
    parameter integer DOWN_OUTPUT_SIZE=768,
    parameter integer OUTPUT_TILE_TAG_WIDTH=10,
    parameter integer PACKED_ATTENTION=0,
    parameter integer PHYSICAL_N_LANES=6,
    parameter integer MLP_M_LANES=4,
    parameter integer GROUP_WIDTH=((TOKENS/4)<=1)?1:$clog2(TOKENS/4),
    parameter LUT_FILE="rtl/tensor_engine/exp_neg_q16_lut.hex"
)(
    input wire clk,input wire rst_n,input wire start,output wire start_ready,
    input wire [63:0] block_image_base_address,
    input wire [63:0] residual_input_base_address,
    input wire [63:0] output_base_address,
    output wire busy,output reg done,output wire protocol_error,
    output reg [63:0] kernel_cycles,
    output wire [63:0] m_axi_araddr,output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,output wire [1:0] m_axi_arburst,
    output wire m_axi_arvalid,input wire m_axi_arready,
    input wire [511:0] m_axi_rdata,input wire [1:0] m_axi_rresp,
    input wire m_axi_rlast,input wire m_axi_rvalid,output wire m_axi_rready,
    output wire [63:0] m_axi_awaddr,output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,output wire [1:0] m_axi_awburst,
    output wire m_axi_awvalid,input wire m_axi_awready,
    output wire [511:0] m_axi_wdata,output wire [63:0] m_axi_wstrb,
    output wire m_axi_wlast,output wire m_axi_wvalid,input wire m_axi_wready,
    input wire [1:0] m_axi_bresp,input wire m_axi_bvalid,output wire m_axi_bready,
    output wire [63:0] read_transactions,
    output wire [63:0] image_read_transactions,
    output wire [63:0] image_constant_bytes,
    output wire [255:0] image_dense_bytes,
    output wire [63:0] residual_read_bytes,
    output wire [63:0] output_write_bytes,
    output wire [63:0] output_write_transactions
);
    localparam [2:0] IDLE=0,INPUTS_START=1,INPUTS_WAIT=2,
      BLOCK_START=3,BLOCK_WAIT=4,DRAIN=5;
    localparam integer TOKEN_GROUPS=TOKENS/4;
    localparam integer DOWN_OUTPUT_TILES=DOWN_OUTPUT_SIZE/6;
    reg [2:0] state;reg block_finished,last_output_seen;
    reg residual_start_pending,preload_start_pending;
    reg residual_finished,preload_finished;
    wire residual_start_ready,residual_busy,residual_done,residual_error;
    wire residual_load_valid;wire [3:0] residual_load_group;
    wire [6:0] residual_load_tile;wire [575:0] residual_load_data;
    wire [63:0] residual_araddr,residual_transactions;wire [7:0] residual_arlen;
    wire [2:0] residual_arsize;wire [1:0] residual_arburst;
    wire residual_arvalid,residual_arready,residual_rvalid,residual_rready;
    wire image_preload_ready,image_preload_done,image_constants_loaded;
    wire image_block_ready,image_busy,image_done,image_error;
    wire image_output_valid;wire [OUTPUT_TILE_TAG_WIDTH-1:0] image_output_tile;
    wire [GROUP_WIDTH-1:0] image_output_group;wire [575:0] image_output_data;
    wire [63:0] image_araddr,image_total_transactions,image_dense_transactions;
    wire [7:0] image_arlen;wire [2:0] image_arsize;wire [1:0] image_arburst;
    wire image_arvalid,image_arready,image_rvalid,image_rready;
    wire [63:0] constant_transactions;
    wire output_ready,output_busy,output_done,output_error;
    wire arbiter_error;wire [63:0] arbitration_cycles;
    wire [255:0] client_araddr;wire [31:0] client_arlen;
    wire [11:0] client_arsize;wire [7:0] client_arburst;
    wire [3:0] client_arvalid,client_arready,client_rvalid,client_rready;
    wire [511:0] client_rdata;wire [1:0] client_rresp;wire client_rlast;
    wire launch_residual=state==INPUTS_START && residual_start_pending
      && residual_start_ready;
    wire launch_preload=state==INPUTS_START && preload_start_pending
      && image_preload_ready;
    wire launch_block=state==BLOCK_START && image_block_ready;
    wire final_output=image_output_valid
      && image_output_tile==DOWN_OUTPUT_TILES-1
      && image_output_group==TOKEN_GROUPS-1;
    assign start_ready=state==IDLE;assign busy=state!=IDLE;
    assign image_read_transactions=image_total_transactions;
    assign protocol_error=residual_error||image_error||output_error||arbiter_error
      ||(image_output_valid&&!output_ready);

    axi512_residual_canvas_reader residual_reader(.clk(clk),.rst_n(rst_n),
      .start(launch_residual),.start_ready(residual_start_ready),
      .input_base_address(residual_input_base_address),
      .residual_load_valid(residual_load_valid),
      .residual_load_group(residual_load_group),
      .residual_load_output_tile(residual_load_tile),
      .residual_load_q10_packed(residual_load_data),
      .m_axi_araddr(residual_araddr),.m_axi_arlen(residual_arlen),
      .m_axi_arsize(residual_arsize),.m_axi_arburst(residual_arburst),
      .m_axi_arvalid(residual_arvalid),.m_axi_arready(residual_arready),
      .m_axi_rdata(client_rdata),.m_axi_rresp(client_rresp),
      .m_axi_rlast(client_rlast),.m_axi_rvalid(residual_rvalid),
      .m_axi_rready(residual_rready),.busy(residual_busy),.done(residual_done),
      .protocol_error(residual_error),.bytes_read(residual_read_bytes),
      .read_transactions(residual_transactions));

    ddit_block_with_image_fabric #(.HEADS(HEADS),
      .ATTENTION_OUTPUT_TILES(ATTENTION_OUTPUT_TILES),.TOKENS(TOKENS),
      .DOWN_INPUT_SIZE(DOWN_INPUT_SIZE),.DOWN_OUTPUT_SIZE(DOWN_OUTPUT_SIZE),
      .OUTPUT_TILE_TAG_WIDTH(OUTPUT_TILE_TAG_WIDTH),.INTERNAL_NORM1(1),
      .PACKED_ATTENTION(PACKED_ATTENTION),
      .PHYSICAL_N_LANES(PHYSICAL_N_LANES),
      .MLP_M_LANES(MLP_M_LANES),
      .GROUP_WIDTH(GROUP_WIDTH),.LUT_FILE(LUT_FILE)) image_block(
      .clk(clk),.rst_n(rst_n),.block_base_address(block_image_base_address),
      .preload_start(launch_preload),.preload_start_ready(image_preload_ready),
      .preload_done(image_preload_done),.constants_loaded(image_constants_loaded),
      .block_start(launch_block),.block_start_ready(image_block_ready),
      .busy(image_busy),.done(image_done),
      .residual_load_valid(residual_load_valid),
      .residual_load_group(residual_load_group),
      .residual_load_output_tile(residual_load_tile),
      .residual_load_q10_packed(residual_load_data),
      .normalized_read_valid(),.normalized_read_group(),
      .normalized_read_input_tile(),.normalized_read_data_valid(1'b0),
      .normalized_q12_packed(2304'b0),.output_valid(image_output_valid),
      .output_tile(image_output_tile),.output_group(image_output_group),
      .outputs_packed(image_output_data),.m_axi_araddr(image_araddr),
      .m_axi_arlen(image_arlen),.m_axi_arsize(image_arsize),
      .m_axi_arburst(image_arburst),.m_axi_arvalid(image_arvalid),
      .m_axi_arready(image_arready),.m_axi_rdata(client_rdata),
      .m_axi_rresp(client_rresp),.m_axi_rlast(client_rlast),
      .m_axi_rvalid(image_rvalid),.m_axi_rready(image_rready),
      .protocol_error(image_error),.read_transactions(image_total_transactions),
      .constant_read_transactions(constant_transactions),
      .constant_bytes_read(image_constant_bytes),
      .dense_client_bytes_read(image_dense_bytes),
      .dense_read_transactions(image_dense_transactions),
      .attention_busy(),.mlp_busy());

    axi512_output_canvas_writer output_writer(.clk(clk),.rst_n(rst_n),
      .output_base_address(output_base_address),.input_valid(image_output_valid),
      .input_ready(output_ready),.input_tile(image_output_tile),
      .input_group(image_output_group),.input_q10_packed(image_output_data),
      .m_axi_awaddr(m_axi_awaddr),.m_axi_awlen(m_axi_awlen),
      .m_axi_awsize(m_axi_awsize),.m_axi_awburst(m_axi_awburst),
      .m_axi_awvalid(m_axi_awvalid),.m_axi_awready(m_axi_awready),
      .m_axi_wdata(m_axi_wdata),.m_axi_wstrb(m_axi_wstrb),
      .m_axi_wlast(m_axi_wlast),.m_axi_wvalid(m_axi_wvalid),
      .m_axi_wready(m_axi_wready),.m_axi_bresp(m_axi_bresp),
      .m_axi_bvalid(m_axi_bvalid),.m_axi_bready(m_axi_bready),
      .busy(output_busy),.done(output_done),.protocol_error(output_error),
      .bytes_written(output_write_bytes),
      .write_transactions(output_write_transactions));

    assign client_araddr[0+:64]=residual_araddr;
    assign client_araddr[64+:64]=image_araddr;assign client_araddr[128+:128]=0;
    assign client_arlen[0+:8]=residual_arlen;assign client_arlen[8+:8]=image_arlen;
    assign client_arlen[16+:16]=0;assign client_arsize[0+:3]=residual_arsize;
    assign client_arsize[3+:3]=image_arsize;assign client_arsize[6+:6]=0;
    assign client_arburst[0+:2]=residual_arburst;
    assign client_arburst[2+:2]=image_arburst;assign client_arburst[4+:4]=0;
    assign client_arvalid={2'b00,image_arvalid,residual_arvalid};
    assign residual_arready=client_arready[0];assign image_arready=client_arready[1];
    assign residual_rvalid=client_rvalid[0];assign image_rvalid=client_rvalid[1];
    assign client_rready={2'b00,image_rready,residual_rready};
    axi512_read_arbiter_4 read_arbiter(.clk(clk),.rst_n(rst_n),
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

    always @(posedge clk) begin
      if(!rst_n) begin state<=IDLE;done<=0;kernel_cycles<=0;
        block_finished<=0;last_output_seen<=0;
        residual_start_pending<=0;preload_start_pending<=0;
        residual_finished<=0;preload_finished<=0;end
      else begin
        done<=0;
        if(state==IDLE && start) begin state<=INPUTS_START;kernel_cycles<=0;
          block_finished<=0;last_output_seen<=0;
          residual_start_pending<=1;preload_start_pending<=1;
          residual_finished<=0;preload_finished<=0;end
        else if(state!=IDLE) kernel_cycles<=kernel_cycles+1'b1;
        if(launch_residual)residual_start_pending<=0;
        if(launch_preload)preload_start_pending<=0;
        if(state==INPUTS_START
          &&(!residual_start_pending||launch_residual)
          &&(!preload_start_pending||launch_preload))state<=INPUTS_WAIT;
        if(state==INPUTS_WAIT) begin
          if(residual_done)residual_finished<=1;
          if(image_preload_done)preload_finished<=1;
          if((residual_finished||residual_done)
            &&(preload_finished||image_preload_done))state<=BLOCK_START;
        end
        if(launch_block)state<=BLOCK_WAIT;
        if(state==BLOCK_WAIT) begin
          if(image_done)block_finished<=1;
          if(final_output)last_output_seen<=1;
          if((block_finished||image_done)&&(last_output_seen||final_output))state<=DRAIN;
        end
        if(state==DRAIN && !output_busy) begin state<=IDLE;done<=1;end
      end
    end
endmodule
