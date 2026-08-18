`timescale 1ns/1ps

module tb_axi512_read_burst_master;
  reg clk=0,rst_n=0,command_valid=0;
  reg [63:0] command_address=0;
  reg [6:0] command_beats=0;
  reg [15:0] command_tag=0;
  reg arready=0,rvalid=0,rlast=0,stream_ready=0;
  reg [511:0] rdata=0;
  reg [1:0] rresp=0;
  wire command_ready;
  wire [63:0] araddr;
  wire [7:0] arlen;
  wire [2:0] arsize;
  wire [1:0] arburst;
  wire arvalid,rready,stream_valid,stream_last,busy,done,protocol_error;
  wire [511:0] stream_data;
  wire [15:0] stream_tag;
  wire [63:0] bytes_read,address_stalls,data_stalls;
  integer cycle=0,source_beat=0,sink_beats=0,done_count=0;
  reg inject_error=0;

  axi512_read_burst_master dut(
    .clk(clk),.rst_n(rst_n),.command_valid(command_valid),
    .command_ready(command_ready),.command_address(command_address),
    .command_beats(command_beats),.command_tag(command_tag),
    .m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),
    .m_axi_arburst(arburst),.m_axi_arvalid(arvalid),.m_axi_arready(arready),
    .m_axi_rdata(rdata),.m_axi_rresp(rresp),.m_axi_rlast(rlast),
    .m_axi_rvalid(rvalid),.m_axi_rready(rready),
    .stream_valid(stream_valid),.stream_ready(stream_ready),
    .stream_data(stream_data),.stream_last(stream_last),.stream_tag(stream_tag),
    .busy(busy),.done(done),.protocol_error(protocol_error),
    .bytes_read(bytes_read),.address_stall_cycles(address_stalls),
    .data_stall_cycles(data_stalls));

  always #2 clk=~clk;
  always @(posedge clk) begin
    cycle=cycle+1;
    arready<=arvalid && cycle[0];
    stream_ready<=cycle[0] || cycle[2];
    if(arvalid && arready) begin
      if(araddr!==command_address || arlen!==command_beats-1 ||
         arsize!==6 || arburst!==1)
        $fatal(1,"AXI address channel mismatch");
      source_beat=0;rvalid<=1;
      rdata<=512'h1000;rresp<=0;rlast<=inject_error;
    end else if(rvalid && rready) begin
      source_beat=source_beat+1;
      if(source_beat==command_beats) begin
        rvalid<=0;rlast<=0;
      end else begin
        rdata<=512'h1000+source_beat;
        rlast<=inject_error ? 0 : source_beat==command_beats-1;
        if(inject_error && source_beat==1) rresp<=2;
      end
    end
    #1;
    if(stream_valid && stream_ready) begin
      if(stream_tag!==command_tag || stream_data!==512'h1000+sink_beats)
        $fatal(1,"AXI stream mismatch");
      sink_beats=sink_beats+1;
    end
    if(done) done_count=done_count+1;
  end

  task launch;
    input [63:0] address;
    input [6:0] beats;
    input [15:0] tag;
    begin
      wait(command_ready);@(negedge clk);command_address=address;
      command_beats=beats;command_tag=tag;command_valid=1;
      @(posedge clk);@(negedge clk);command_valid=0;
    end
  endtask

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    launch(64'h2000,3,16'h1234);wait(done);repeat(2) @(posedge clk);
    if(protocol_error || sink_beats!=3 || bytes_read!=192)
      $fatal(1,"valid AXI burst failed");
    sink_beats=0;inject_error=1;
    launch(64'h4000,3,16'h5678);wait(done);repeat(2) @(posedge clk);
    if(!protocol_error || sink_beats!=1 || bytes_read!=256)
      $fatal(1,"early-last AXI error was not detected");
    if(done_count!=2 || address_stalls==0 || data_stalls==0 || busy)
      $fatal(1,"AXI counters or completion mismatch");
    $display("tb_axi512_read_burst_master: PASS bytes=%0d ar_stalls=%0d data_stalls=%0d",
      bytes_read,address_stalls,data_stalls);
    $finish;
  end
  initial begin repeat(200) @(posedge clk);
    $display("timeout state=%0d arvalid=%0d arready=%0d rvalid=%0d rready=%0d source=%0d sink=%0d done_count=%0d",
      dut.state,arvalid,arready,rvalid,rready,source_beat,sink_beats,done_count);
    $fatal(1,"timeout");end
endmodule
