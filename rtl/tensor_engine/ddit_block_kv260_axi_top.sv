`timescale 1ns/1ps

module ddit_block_kv260_axi_top #(
    parameter integer PACKED_ATTENTION=1,
    parameter integer PHYSICAL_N_LANES=6,
    parameter integer MLP_M_LANES=4
)(
    input wire ap_clk,input wire ap_rst_n,
    input wire [7:0] s_axi_control_awaddr,input wire s_axi_control_awvalid,
    output wire s_axi_control_awready,input wire [31:0] s_axi_control_wdata,
    input wire [3:0] s_axi_control_wstrb,input wire s_axi_control_wvalid,
    output wire s_axi_control_wready,output reg [1:0] s_axi_control_bresp,
    output reg s_axi_control_bvalid,input wire s_axi_control_bready,
    input wire [7:0] s_axi_control_araddr,input wire s_axi_control_arvalid,
    output wire s_axi_control_arready,output reg [31:0] s_axi_control_rdata,
    output reg [1:0] s_axi_control_rresp,output reg s_axi_control_rvalid,
    input wire s_axi_control_rready,
    output wire [63:0] m_axi_gmem_araddr,output wire [7:0] m_axi_gmem_arlen,
    output wire [2:0] m_axi_gmem_arsize,output wire [1:0] m_axi_gmem_arburst,
    output wire m_axi_gmem_arvalid,input wire m_axi_gmem_arready,
    input wire [511:0] m_axi_gmem_rdata,input wire [1:0] m_axi_gmem_rresp,
    input wire m_axi_gmem_rlast,input wire m_axi_gmem_rvalid,
    output wire m_axi_gmem_rready,
    output wire [63:0] m_axi_gmem_awaddr,output wire [7:0] m_axi_gmem_awlen,
    output wire [2:0] m_axi_gmem_awsize,output wire [1:0] m_axi_gmem_awburst,
    output wire m_axi_gmem_awvalid,input wire m_axi_gmem_awready,
    output wire [511:0] m_axi_gmem_wdata,output wire [63:0] m_axi_gmem_wstrb,
    output wire m_axi_gmem_wlast,output wire m_axi_gmem_wvalid,
    input wire m_axi_gmem_wready,input wire [1:0] m_axi_gmem_bresp,
    input wire m_axi_gmem_bvalid,output wire m_axi_gmem_bready
);
    reg aw_pending,w_pending;reg [7:0] awaddr_reg;reg [31:0] wdata_reg;
    reg [3:0] wstrb_reg;reg start_pulse,done_sticky,error_sticky;
    reg [63:0] block_base,residual_base,output_base;
    wire core_start_ready,core_busy,core_done,core_error;
    wire [63:0] kernel_cycles,read_transactions,image_read_transactions;
    wire [63:0] image_constant_bytes,residual_read_bytes,output_write_bytes;
    wire [63:0] output_write_transactions;wire [255:0] image_dense_bytes;
    wire aw_handshake=s_axi_control_awvalid&&s_axi_control_awready;
    wire w_handshake=s_axi_control_wvalid&&s_axi_control_wready;
    wire write_fire=(aw_pending||aw_handshake)&&(w_pending||w_handshake)
      &&!s_axi_control_bvalid;
    wire [7:0] write_address=aw_pending?awaddr_reg:s_axi_control_awaddr;
    wire [31:0] write_data=w_pending?wdata_reg:s_axi_control_wdata;
    wire [3:0] write_strobes=w_pending?wstrb_reg:s_axi_control_wstrb;
    reg [31:0] selected_read_data;
    function automatic [31:0] merge_strobes;
      input [31:0] prior;
      input [31:0] value;
      input [3:0] strobes;
      integer byte_index;begin merge_strobes=prior;
        for(byte_index=0;byte_index<4;byte_index=byte_index+1)
          if(strobes[byte_index])merge_strobes[byte_index*8+:8]=
            value[byte_index*8+:8];end
    endfunction
    assign s_axi_control_awready=!aw_pending&&!s_axi_control_bvalid;
    assign s_axi_control_wready=!w_pending&&!s_axi_control_bvalid;
    assign s_axi_control_arready=!s_axi_control_rvalid;
    always @* begin
      case(s_axi_control_araddr[7:2])
        0:selected_read_data={28'd0,error_sticky,done_sticky,core_busy,
          core_start_ready};
        4:selected_read_data=block_base[31:0];
        5:selected_read_data=block_base[63:32];
        6:selected_read_data=residual_base[31:0];
        7:selected_read_data=residual_base[63:32];
        8:selected_read_data=output_base[31:0];
        9:selected_read_data=output_base[63:32];
        10:selected_read_data=kernel_cycles[31:0];
        11:selected_read_data=kernel_cycles[63:32];
        12:selected_read_data=read_transactions[31:0];
        13:selected_read_data=residual_read_bytes[31:0];
        14:selected_read_data=image_constant_bytes[31:0];
        15:selected_read_data=image_dense_bytes[31:0]+image_dense_bytes[95:64]
          +image_dense_bytes[159:128]+image_dense_bytes[223:192];
        16:selected_read_data=output_write_bytes[31:0];
        17:selected_read_data=output_write_transactions[31:0];
        default:selected_read_data=0;
      endcase
    end
    always @(posedge ap_clk) begin
      if(!ap_rst_n) begin aw_pending<=0;w_pending<=0;awaddr_reg<=0;
        wdata_reg<=0;wstrb_reg<=0;s_axi_control_bresp<=0;
        s_axi_control_bvalid<=0;s_axi_control_rdata<=0;
        s_axi_control_rresp<=0;s_axi_control_rvalid<=0;start_pulse<=0;
        done_sticky<=0;error_sticky<=0;block_base<=0;residual_base<=0;
        output_base<=0;end
      else begin
        start_pulse<=0;
        if(aw_handshake)begin aw_pending<=1;awaddr_reg<=s_axi_control_awaddr;end
        if(w_handshake)begin w_pending<=1;wdata_reg<=s_axi_control_wdata;
          wstrb_reg<=s_axi_control_wstrb;end
        if(write_fire) begin
          aw_pending<=0;w_pending<=0;s_axi_control_bvalid<=1;
          s_axi_control_bresp<=0;
          case(write_address[7:2])
            0:begin
              if(write_strobes[0]&&write_data[0])begin
                if(core_start_ready)start_pulse<=1;else error_sticky<=1;end
              if(write_strobes[0]&&write_data[1])done_sticky<=0;
              if(write_strobes[0]&&write_data[2])error_sticky<=0;
            end
            4:block_base[31:0]<=merge_strobes(block_base[31:0],write_data,write_strobes);
            5:block_base[63:32]<=merge_strobes(block_base[63:32],write_data,write_strobes);
            6:residual_base[31:0]<=merge_strobes(residual_base[31:0],write_data,write_strobes);
            7:residual_base[63:32]<=merge_strobes(residual_base[63:32],write_data,write_strobes);
            8:output_base[31:0]<=merge_strobes(output_base[31:0],write_data,write_strobes);
            9:output_base[63:32]<=merge_strobes(output_base[63:32],write_data,write_strobes);
            default:begin s_axi_control_bresp<=2'b10;error_sticky<=1;end
          endcase
        end else if(s_axi_control_bvalid&&s_axi_control_bready)
          s_axi_control_bvalid<=0;
        if(s_axi_control_arvalid&&s_axi_control_arready)begin
          s_axi_control_rdata<=selected_read_data;s_axi_control_rresp<=0;
          s_axi_control_rvalid<=1;end
        else if(s_axi_control_rvalid&&s_axi_control_rready)
          s_axi_control_rvalid<=0;
        if(core_done)done_sticky<=1;
        if(core_error)error_sticky<=1;
      end
    end

    ddit_block_kv260_kernel_core #(
      .PACKED_ATTENTION(PACKED_ATTENTION),
      .PHYSICAL_N_LANES(PHYSICAL_N_LANES),.MLP_M_LANES(MLP_M_LANES)
    ) core(.clk(ap_clk),.rst_n(ap_rst_n),
      .start(start_pulse),.start_ready(core_start_ready),
      .block_image_base_address(block_base),
      .residual_input_base_address(residual_base),.output_base_address(output_base),
      .busy(core_busy),.done(core_done),.protocol_error(core_error),
      .kernel_cycles(kernel_cycles),.m_axi_araddr(m_axi_gmem_araddr),
      .m_axi_arlen(m_axi_gmem_arlen),.m_axi_arsize(m_axi_gmem_arsize),
      .m_axi_arburst(m_axi_gmem_arburst),.m_axi_arvalid(m_axi_gmem_arvalid),
      .m_axi_arready(m_axi_gmem_arready),.m_axi_rdata(m_axi_gmem_rdata),
      .m_axi_rresp(m_axi_gmem_rresp),.m_axi_rlast(m_axi_gmem_rlast),
      .m_axi_rvalid(m_axi_gmem_rvalid),.m_axi_rready(m_axi_gmem_rready),
      .m_axi_awaddr(m_axi_gmem_awaddr),.m_axi_awlen(m_axi_gmem_awlen),
      .m_axi_awsize(m_axi_gmem_awsize),.m_axi_awburst(m_axi_gmem_awburst),
      .m_axi_awvalid(m_axi_gmem_awvalid),.m_axi_awready(m_axi_gmem_awready),
      .m_axi_wdata(m_axi_gmem_wdata),.m_axi_wstrb(m_axi_gmem_wstrb),
      .m_axi_wlast(m_axi_gmem_wlast),.m_axi_wvalid(m_axi_gmem_wvalid),
      .m_axi_wready(m_axi_gmem_wready),.m_axi_bresp(m_axi_gmem_bresp),
      .m_axi_bvalid(m_axi_gmem_bvalid),.m_axi_bready(m_axi_gmem_bready),
      .read_transactions(read_transactions),
      .image_read_transactions(image_read_transactions),
      .image_constant_bytes(image_constant_bytes),.image_dense_bytes(image_dense_bytes),
      .residual_read_bytes(residual_read_bytes),
      .output_write_bytes(output_write_bytes),
      .output_write_transactions(output_write_transactions));
endmodule
