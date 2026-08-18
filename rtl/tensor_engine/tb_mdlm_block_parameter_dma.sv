`timescale 1ns/1ps

module tb_mdlm_block_parameter_dma;
  reg clk=0,rst_n=0,request_valid=0;
  reg [63:0] block_base=64'h0000_0000_2000_0000;
  reg [3:0] request_section=0;
  reg [13:0] request_record=0;
  reg [15:0] request_tag=0;
  reg arready=0,rvalid=0,rlast=0,stream_ready=0;
  reg [511:0] rdata=0;
  reg [1:0] rresp=0;
  wire request_ready;
  wire [63:0] araddr;
  wire [7:0] arlen;
  wire [2:0] arsize;
  wire [1:0] arburst;
  wire arvalid,rready,stream_valid,stream_last,busy,done;
  wire invalid_request,protocol_error;
  wire [511:0] stream_data;
  wire [15:0] stream_tag;
  wire [5:0] stream_byte_offset;
  wire [9:0] stream_payload_bytes;
  wire [63:0] bytes_read,address_stalls,data_stalls;
  integer cycle=0,source_beat=0,sink_beat=0,expected_beats=0;
  integer done_count=0,invalid_count=0;
  reg [63:0] expected_address=0;
  reg [15:0] expected_tag=0;
  reg [5:0] expected_offset=0;
  reg [9:0] expected_payload=0;

  mdlm_block_parameter_dma dut(
    .clk(clk),.rst_n(rst_n),.block_base_address(block_base),
    .request_valid(request_valid),.request_ready(request_ready),
    .request_section_id(request_section),.request_record_index(request_record),
    .request_tag(request_tag),.m_axi_araddr(araddr),.m_axi_arlen(arlen),
    .m_axi_arsize(arsize),.m_axi_arburst(arburst),.m_axi_arvalid(arvalid),
    .m_axi_arready(arready),.m_axi_rdata(rdata),.m_axi_rresp(rresp),
    .m_axi_rlast(rlast),.m_axi_rvalid(rvalid),.m_axi_rready(rready),
    .stream_valid(stream_valid),.stream_ready(stream_ready),
    .stream_data(stream_data),.stream_last(stream_last),.stream_tag(stream_tag),
    .stream_record_byte_offset(stream_byte_offset),
    .stream_record_payload_bytes(stream_payload_bytes),.busy(busy),.done(done),
    .invalid_request(invalid_request),.protocol_error(protocol_error),
    .bytes_read(bytes_read),.address_stall_cycles(address_stalls),
    .data_stall_cycles(data_stalls));

  always #2 clk=~clk;

  always @(posedge clk) begin
    cycle=cycle+1;
    arready<=arvalid && cycle[0];
    stream_ready<=cycle[0] || cycle[2];
    if(arvalid && arready) begin
      if(araddr!==expected_address || arlen!==expected_beats-1 ||
         arsize!==6 || arburst!==1)
        $fatal(1,"DMA address mismatch");
      source_beat=0;rvalid<=1;rdata<=512'h5000;rlast<=expected_beats==1;
    end else if(rvalid && rready) begin
      source_beat=source_beat+1;
      if(source_beat==expected_beats) begin
        rvalid<=0;rlast<=0;
      end else begin
        rdata<=512'h5000+source_beat;
        rlast<=source_beat==expected_beats-1;
      end
    end
    #1;
    if(stream_valid && stream_ready) begin
      if(stream_tag!==expected_tag || stream_data!==512'h5000+sink_beat ||
         stream_byte_offset!==expected_offset ||
         stream_payload_bytes!==expected_payload)
        $fatal(1,"DMA stream or record metadata mismatch");
      if(stream_last!==(sink_beat==expected_beats-1))
        $fatal(1,"DMA stream last mismatch");
      sink_beat=sink_beat+1;
    end
    if(done) done_count=done_count+1;
    if(invalid_request) invalid_count=invalid_count+1;
  end

  task launch_valid;
    input [3:0] section;
    input [13:0] record_index;
    input [15:0] tag;
    input [63:0] address;
    input integer beats;
    input [5:0] byte_offset;
    input [9:0] payload;
    begin
      wait(request_ready);@(negedge clk);
      request_section=section;request_record=record_index;request_tag=tag;
      expected_address=address;expected_beats=beats;expected_tag=tag;
      expected_offset=byte_offset;expected_payload=payload;sink_beat=0;
      request_valid=1;@(posedge clk);@(negedge clk);request_valid=0;
    end
  endtask

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    launch_valid(1,7,16'h1111,block_base+28672+7*384,6,0,384);
    wait(done);repeat(2) @(posedge clk);
    if(protocol_error || sink_beat!=6 || bytes_read!=384)
      $fatal(1,"QKV weight DMA failed");
    launch_valid(5,17,16'h2222,block_base+4284416+64,1,4,3);
    wait(done);repeat(2) @(posedge clk);
    if(protocol_error || sink_beat!=1 || bytes_read!=448)
      $fatal(1,"compact reciprocal DMA failed");

    wait(request_ready);@(negedge clk);
    request_section=1;request_record=9504;request_tag=16'h3333;
    request_valid=1;@(posedge clk);@(negedge clk);request_valid=0;
    repeat(2) @(posedge clk);
    if(invalid_count!=1 || done_count!=3 || busy || bytes_read!=448)
      $fatal(1,"invalid DMA request handling failed");
    if(address_stalls==0 || data_stalls==0)
      $fatal(1,"DMA stall counters were not exercised");
    $display("tb_mdlm_block_parameter_dma: PASS bytes=%0d done=%0d invalid=%0d",
      bytes_read,done_count,invalid_count);
    $finish;
  end

  initial begin
    repeat(300) @(posedge clk);
    $fatal(1,"timeout");
  end
endmodule
