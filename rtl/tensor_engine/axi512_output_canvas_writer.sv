`timescale 1ns/1ps

module axi512_output_canvas_writer(
    input wire clk,input wire rst_n,input wire [63:0] output_base_address,
    input wire input_valid,output wire input_ready,
    input wire [9:0] input_tile,input wire [3:0] input_group,
    input wire [4*6*24-1:0] input_q10_packed,
    output wire [63:0] m_axi_awaddr,output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,output wire [1:0] m_axi_awburst,
    output wire m_axi_awvalid,input wire m_axi_awready,
    output wire [511:0] m_axi_wdata,output wire [63:0] m_axi_wstrb,
    output wire m_axi_wlast,output wire m_axi_wvalid,input wire m_axi_wready,
    input wire [1:0] m_axi_bresp,input wire m_axi_bvalid,output wire m_axi_bready,
    output wire busy,output reg done,output reg protocol_error,
    output reg [63:0] bytes_written,output reg [63:0] write_transactions
);
    localparam [2:0] IDLE=0,ADDRESS=1,DATA0=2,DATA1=3,RESPONSE=4;
    reg [2:0] state;reg [9:0] active_tile;reg [3:0] active_group;
    reg [4*6*24-1:0] active_data;
    wire [63:0] tile_offset={{43{1'b0}},active_tile,11'b0};
    wire [63:0] group_offset={{53{1'b0}},active_group,7'b0};
    assign input_ready=state==IDLE;assign busy=state!=IDLE;
    assign m_axi_awaddr=output_base_address+tile_offset+group_offset;
    assign m_axi_awlen=8'd1;assign m_axi_awsize=3'd6;
    assign m_axi_awburst=2'b01;assign m_axi_awvalid=state==ADDRESS;
    assign m_axi_wdata=state==DATA0?active_data[511:0]:
      {{448{1'b0}},active_data[575:512]};
    assign m_axi_wstrb=state==DATA0?64'hffffffffffffffff:64'h00000000000000ff;
    assign m_axi_wlast=state==DATA1;
    assign m_axi_wvalid=state==DATA0 || state==DATA1;
    assign m_axi_bready=state==RESPONSE;
    always @(posedge clk) begin
      if(!rst_n) begin state<=IDLE;active_tile<=0;active_group<=0;
        active_data<=0;done<=0;protocol_error<=0;bytes_written<=0;
        write_transactions<=0;end
      else begin
        done<=0;
        if(state==IDLE && input_valid) begin active_tile<=input_tile;
          active_group<=input_group;active_data<=input_q10_packed;
          protocol_error<=0;state<=ADDRESS;end
        else if(state==ADDRESS && m_axi_awready) begin
          write_transactions<=write_transactions+1'b1;state<=DATA0;end
        else if(state==DATA0 && m_axi_wready) begin
          bytes_written<=bytes_written+64;state<=DATA1;end
        else if(state==DATA1 && m_axi_wready) begin
          bytes_written<=bytes_written+8;state<=RESPONSE;end
        else if(state==RESPONSE && m_axi_bvalid) begin
          if(m_axi_bresp!=0)protocol_error<=1;state<=IDLE;done<=1;end
      end
    end
    always @(posedge clk) begin
`ifndef SYNTHESIS
      if(rst_n && input_valid && !input_ready)
        $error("output canvas writer overrun");
`endif
    end
endmodule
