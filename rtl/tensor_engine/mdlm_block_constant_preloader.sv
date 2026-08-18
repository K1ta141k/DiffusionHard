`timescale 1ns/1ps

module mdlm_block_constant_preloader(
    input wire clk,input wire rst_n,input wire start,output wire start_ready,
    input wire [63:0] block_base_address,
    output wire rotary_load_valid,output wire [5:0] rotary_load_token,
    output wire [4:0] rotary_load_pair,
    output wire signed [15:0] rotary_load_cosine_q15,
    output wire signed [15:0] rotary_load_sine_q15,
    input wire [9:0] reciprocal_lookup_channel,
    output wire [17:0] reciprocal_lookup_q15,
    output wire [63:0] m_axi_araddr,output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,output wire [1:0] m_axi_arburst,
    output wire m_axi_arvalid,input wire m_axi_arready,
    input wire [511:0] m_axi_rdata,input wire [1:0] m_axi_rresp,
    input wire m_axi_rlast,input wire m_axi_rvalid,output wire m_axi_rready,
    output wire busy,output reg done,output wire protocol_error,
    output wire [63:0] bytes_read,output reg [63:0] read_transactions);
    localparam [2:0] IDLE=0,ROTARY_REQUEST=1,ROTARY_WAIT=2,
        ROTARY_EMIT=3,RECIP_REQUEST=4,RECIP_WAIT=5,RECIP_EMIT=6;
    reg [2:0] state;reg [6:0] line_index;reg [3:0] slot_index;
    reg [511:0] line_data;reg [17:0] reciprocal_mem[0:767];
    wire dma_request_ready,dma_stream_valid,dma_stream_last,dma_done,dma_invalid;
    wire dma_error;wire [511:0] dma_stream_data;wire [15:0] dma_stream_tag;
    wire [5:0] dma_offset;wire [9:0] dma_payload;wire [10:0] rotary_index;
    wire [31:0] selected_word=line_data[slot_index*32 +: 32];
    wire stream_phase=state==ROTARY_WAIT || state==RECIP_WAIT;
    assign start_ready=state==IDLE;assign busy=state!=IDLE;
    assign rotary_load_valid=state==ROTARY_EMIT;
    assign rotary_index={line_index,slot_index};
    assign rotary_load_token=rotary_index[10:5];
    assign rotary_load_pair=rotary_index[4:0];
    assign rotary_load_cosine_q15=selected_word[15:0];
    assign rotary_load_sine_q15=selected_word[31:16];
    assign reciprocal_lookup_q15=reciprocal_mem[reciprocal_lookup_channel];
    mdlm_block_parameter_dma dma(.clk(clk),.rst_n(rst_n),
      .block_base_address(block_base_address),
      .request_valid(state==ROTARY_REQUEST || state==RECIP_REQUEST),
      .request_ready(dma_request_ready),
      .request_section_id(state==ROTARY_REQUEST?4'd2:4'd5),
      .request_record_index({3'd0,line_index,4'd0}),.request_tag({9'd0,line_index}),
      .m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),
      .m_axi_arsize(m_axi_arsize),.m_axi_arburst(m_axi_arburst),
      .m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
      .m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),
      .m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),
      .m_axi_rready(m_axi_rready),.stream_valid(dma_stream_valid),
      .stream_ready(stream_phase),.stream_data(dma_stream_data),
      .stream_last(dma_stream_last),.stream_tag(dma_stream_tag),
      .stream_record_byte_offset(dma_offset),
      .stream_record_payload_bytes(dma_payload),.busy(),.done(dma_done),
      .invalid_request(dma_invalid),.protocol_error(dma_error),
      .bytes_read(bytes_read),.address_stall_cycles(),.data_stall_cycles());
    assign protocol_error=dma_error|dma_invalid;
    always @(posedge clk) begin
      if(!rst_n) begin state<=IDLE;line_index<=0;slot_index<=0;line_data<=0;
        done<=0;read_transactions<=0;end
      else begin
        done<=0;
        if(state==IDLE && start) begin state<=ROTARY_REQUEST;line_index<=0;
          slot_index<=0;read_transactions<=0;end
        else if((state==ROTARY_REQUEST || state==RECIP_REQUEST)
          && dma_request_ready) begin
          read_transactions<=read_transactions+1'b1;
          state<=state==ROTARY_REQUEST?ROTARY_WAIT:RECIP_WAIT;
        end else if(state==ROTARY_WAIT && dma_stream_valid) begin
          line_data<=dma_stream_data;slot_index<=0;state<=ROTARY_EMIT;
        end else if(state==ROTARY_EMIT) begin
          if(slot_index==15) begin
            slot_index<=0;
            if(line_index==127) begin line_index<=0;state<=RECIP_REQUEST;end
            else begin line_index<=line_index+1'b1;state<=ROTARY_REQUEST;end
          end else slot_index<=slot_index+1'b1;
        end else if(state==RECIP_WAIT && dma_stream_valid) begin
          line_data<=dma_stream_data;slot_index<=0;state<=RECIP_EMIT;
        end else if(state==RECIP_EMIT) begin
          reciprocal_mem[{line_index[5:0],slot_index}]<=selected_word[17:0];
          if(slot_index==15) begin
            slot_index<=0;
            if(line_index==47) begin state<=IDLE;done<=1;end
            else begin line_index<=line_index+1'b1;state<=RECIP_REQUEST;end
          end else slot_index<=slot_index+1'b1;
        end
      end
    end
endmodule
