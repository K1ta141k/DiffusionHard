`timescale 1ns/1ps
module tb_axi512_read_arbiter_4;
  reg clk=0,rst_n=0;reg [255:0] s_addr=0;reg [31:0] s_len=0;
  reg [11:0] s_size={4{3'd6}};reg [7:0] s_burst={4{2'd1}};
  reg [3:0] s_valid=0,s_rready=0;wire [3:0] s_ready,s_rvalid;
  wire [511:0] s_rdata;wire [1:0] s_rresp;wire s_rlast;
  wire [63:0] m_addr;wire [7:0] m_len;wire [2:0] m_size;
  wire [1:0] m_burst;wire m_valid,m_rready,busy,error;
  reg m_ready=0,m_rvalid=0,m_rlast=0;reg [511:0] m_rdata=0;
  wire [63:0] transactions,arb_cycles;
  integer cycle=0,source_beat=0,source_beats=0,order_count=0,total_beats=0;
  integer expected_order[0:3];integer client_beats[0:3];integer client;
  reg inject_late_last=0;
  axi512_read_arbiter_4 dut(
    .clk(clk),.rst_n(rst_n),.s_axi_araddr(s_addr),.s_axi_arlen(s_len),
    .s_axi_arsize(s_size),.s_axi_arburst(s_burst),.s_axi_arvalid(s_valid),
    .s_axi_arready(s_ready),.s_axi_rdata(s_rdata),.s_axi_rresp(s_rresp),
    .s_axi_rlast(s_rlast),.s_axi_rvalid(s_rvalid),.s_axi_rready(s_rready),
    .m_axi_araddr(m_addr),.m_axi_arlen(m_len),.m_axi_arsize(m_size),
    .m_axi_arburst(m_burst),.m_axi_arvalid(m_valid),.m_axi_arready(m_ready),
    .m_axi_rdata(m_rdata),.m_axi_rresp(0),.m_axi_rlast(m_rlast),
    .m_axi_rvalid(m_rvalid),.m_axi_rready(m_rready),.busy(busy),
    .protocol_error(error),.read_transactions(transactions),
    .arbitration_cycles(arb_cycles));
  always #2 clk=~clk;
  always @(posedge clk) begin
    cycle=cycle+1;m_ready<=m_valid && cycle[0];
    s_rready<={cycle[0]||cycle[2],cycle[1],cycle[0],cycle[2]};
    if(m_valid && m_ready) begin
      client=m_addr[9:8];
      if(client!==expected_order[order_count]) $fatal(1,"grant order mismatch");
      order_count=order_count+1;source_beat=0;source_beats=m_len+1;
      m_rdata<=m_addr;m_rlast<=source_beats==1 && !inject_late_last;m_rvalid<=1;
    end else if(m_rvalid && m_rready) begin
      source_beat=source_beat+1;
      if(source_beat==source_beats) begin m_rvalid<=0;m_rlast<=0;end
      else begin m_rdata<=m_addr+source_beat;
        m_rlast<=source_beat==source_beats-1 && !inject_late_last;end
    end
    for(client=0;client<4;client=client+1) begin
      if(s_valid[client] && s_ready[client]) s_valid[client]<=0;
      if(s_rvalid[client] && s_rready[client]) begin
        if(s_rdata[63:0]!==s_addr[client*64 +: 64]+client_beats[client])
          $fatal(1,"routed data mismatch client=%0d",client);
        client_beats[client]=client_beats[client]+1;total_beats=total_beats+1;
      end
    end
  end
  initial begin
    expected_order[0]=0;expected_order[1]=2;expected_order[2]=3;expected_order[3]=1;
    for(client=0;client<4;client=client+1) begin
      s_addr[client*64 +: 64]=client<<8;client_beats[client]=0;
      s_len[client*8 +: 8]=client==0 ? 1 : 0;
    end
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;s_valid=4'b0101;
    wait(transactions==2 && !busy);@(negedge clk);s_valid=4'b1010;
    wait(transactions==3 && !busy);inject_late_last=1;
    wait(transactions==4 && !busy);repeat(2) @(posedge clk);#1;
    if(!error || order_count!=4 || transactions!=4 || total_beats!=5 ||
       client_beats[0]!=2 || client_beats[1]!=1 || client_beats[2]!=1 ||
       client_beats[3]!=1 || arb_cycles!=4)
      $fatal(1,"arbiter completion mismatch");
    $display("tb_axi512_read_arbiter_4: PASS transactions=%0d beats=%0d",
      transactions,total_beats);$finish;
  end
  initial begin repeat(300) @(posedge clk);$fatal(1,"timeout");end
endmodule
