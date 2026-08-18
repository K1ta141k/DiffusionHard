`timescale 1ns/1ps

module axi512_residual_canvas_reader #(
    parameter integer RECORDS=2048
)(
    input wire clk,input wire rst_n,input wire start,output wire start_ready,
    input wire [63:0] input_base_address,
    output reg residual_load_valid,output reg [3:0] residual_load_group,
    output reg [6:0] residual_load_output_tile,
    output reg [4*6*24-1:0] residual_load_q10_packed,
    output wire [63:0] m_axi_araddr,output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,output wire [1:0] m_axi_arburst,
    output wire m_axi_arvalid,input wire m_axi_arready,
    input wire [511:0] m_axi_rdata,input wire [1:0] m_axi_rresp,
    input wire m_axi_rlast,input wire m_axi_rvalid,output wire m_axi_rready,
    output wire busy,output reg done,output wire protocol_error,
    output wire [63:0] bytes_read,output reg [63:0] read_transactions
);
    localparam [1:0] IDLE=0,REQUEST=1,DATA=2;
    localparam integer RECORD_WIDTH=(RECORDS<=1)?1:$clog2(RECORDS);
    reg [1:0] state;reg [RECORD_WIDTH-1:0] record_index;
    reg first_beat;reg [511:0] first_data;
    wire command_ready,stream_valid,stream_last,master_busy,master_done;
    wire [511:0] stream_data;wire [RECORD_WIDTH-1:0] stream_tag;
    wire master_error;
    wire [63:0] address_stalls,data_stalls;
    wire accept_stream=stream_valid;
    wire [63:0] record_offset={{(64-RECORD_WIDTH-7){1'b0}},record_index,7'b0};
    assign start_ready=state==IDLE;assign busy=state!=IDLE || master_busy;
    assign protocol_error=master_error;

    axi512_read_burst_master #(.TAG_WIDTH(RECORD_WIDTH)) master(
      .clk(clk),.rst_n(rst_n),.command_valid(state==REQUEST),
      .command_ready(command_ready),.command_address(input_base_address+record_offset),
      .command_beats(7'd2),.command_tag(record_index),
      .m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),
      .m_axi_arsize(m_axi_arsize),.m_axi_arburst(m_axi_arburst),
      .m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
      .m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),
      .m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),
      .m_axi_rready(m_axi_rready),.stream_valid(stream_valid),
      .stream_ready(1'b1),.stream_data(stream_data),.stream_last(stream_last),
      .stream_tag(stream_tag),.busy(master_busy),.done(master_done),
      .protocol_error(master_error),.bytes_read(bytes_read),
      .address_stall_cycles(address_stalls),.data_stall_cycles(data_stalls));

    always @(posedge clk) begin
      if(!rst_n) begin state<=IDLE;record_index<=0;first_beat<=1;
        first_data<=0;residual_load_valid<=0;residual_load_group<=0;
        residual_load_output_tile<=0;residual_load_q10_packed<=0;
        done<=0;read_transactions<=0;end
      else begin
        residual_load_valid<=0;done<=0;
        if(state==IDLE && start) begin record_index<=0;first_beat<=1;
          read_transactions<=0;state<=REQUEST;end
        else if(state==REQUEST && command_ready) begin
          read_transactions<=read_transactions+1'b1;state<=DATA;
        end
        if(state==DATA && accept_stream) begin
          if(first_beat) begin first_data<=stream_data;first_beat<=0;end
          else begin
            residual_load_q10_packed<={stream_data[63:0],first_data};
            residual_load_group<=record_index>>7;
            residual_load_output_tile<=record_index;
            residual_load_valid<=1;first_beat<=1;
            if(record_index==RECORDS-1) begin state<=IDLE;done<=1;end
            else begin record_index<=record_index+1'b1;state<=REQUEST;end
          end
        end
      end
    end
    always @(posedge clk) begin
`ifndef SYNTHESIS
      if(rst_n && stream_valid && first_beat && stream_last)
        $error("residual canvas record ended after one beat");
      if(rst_n && stream_valid && !first_beat && !stream_last)
        $error("residual canvas record exceeded two beats");
      if(rst_n && stream_valid && stream_tag!=record_index)
        $error("residual canvas read tag changed");
`endif
    end
endmodule
