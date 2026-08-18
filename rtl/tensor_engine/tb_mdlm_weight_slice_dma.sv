`timescale 1ns/1ps

module tb_mdlm_weight_slice_dma;
  localparam [63:0] BASE=64'h0000_0000_3000_0000;
  reg clk=0,rst_n=0,command_valid=0,command_bank=1;
  reg [8:0] output_tile=7;
  reg arready=0,rvalid=0,rlast=0,weight_ready=0;
  reg [511:0] rdata=0;
  reg [1:0] rresp=0;
  wire command_ready,weight_valid,weight_bank;
  wire [4:0] weight_k;
  wire [3071:0] weight_data;
  wire [63:0] araddr,bytes_read,address_stalls,data_stalls;
  wire [7:0] arlen;
  wire [2:0] arsize;
  wire [1:0] arburst;
  wire arvalid,rready,busy,done,protocol_error;
  integer cycle=0,current_record=0,source_beat=0,accepted_tiles=0;
  integer expected_record;
  integer beat;

  mdlm_weight_slice_dma #(.SECTION_ID(1),.DATA_WIDTH(16)) dut(
    .clk(clk),.rst_n(rst_n),.block_base_address(BASE),
    .command_valid(command_valid),.command_ready(command_ready),
    .command_bank(command_bank),.command_output_tile(output_tile),
    .weight_load_valid(weight_valid),.weight_load_ready(weight_ready),
    .weight_load_bank(weight_bank),.weight_load_k_tile(weight_k),
    .weight_load_data(weight_data),.m_axi_araddr(araddr),.m_axi_arlen(arlen),
    .m_axi_arsize(arsize),.m_axi_arburst(arburst),.m_axi_arvalid(arvalid),
    .m_axi_arready(arready),.m_axi_rdata(rdata),.m_axi_rresp(rresp),
    .m_axi_rlast(rlast),.m_axi_rvalid(rvalid),.m_axi_rready(rready),
    .busy(busy),.done(done),.protocol_error(protocol_error),
    .bytes_read(bytes_read),.address_stall_cycles(address_stalls),
    .data_stall_cycles(data_stalls));

  always #2 clk=~clk;
  always @(posedge clk) begin
    cycle=cycle+1;
    arready<=arvalid && (cycle[1:0]!=0);
    weight_ready<=cycle[1] || cycle[3];
    if(arvalid && arready) begin
      expected_record=output_tile*24+accepted_tiles;
      if(araddr!==BASE+28672+expected_record*384 || arlen!==5 ||
         arsize!==6 || arburst!==1)
        $fatal(1,"weight slice DMA address mismatch tile=%0d",accepted_tiles);
      current_record=expected_record;source_beat=0;rvalid<=1;
      rdata<=(expected_record<<8);rlast<=0;
    end else if(rvalid && rready) begin
      source_beat=source_beat+1;
      if(source_beat==6) begin rvalid<=0;rlast<=0;end
      else begin
        rdata<=(current_record<<8)+source_beat;
        rlast<=source_beat==5;
      end
    end
    #1;
    if(weight_valid && weight_ready) begin
      if(!weight_bank || weight_k!==accepted_tiles[4:0])
        $fatal(1,"weight slice tag mismatch");
      for(beat=0;beat<6;beat=beat+1)
        if(weight_data[beat*512 +: 512]!==
           ((output_tile*24+accepted_tiles)<<8)+beat)
          $fatal(1,"weight slice data mismatch tile=%0d beat=%0d",
            accepted_tiles,beat);
      accepted_tiles=accepted_tiles+1;
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    command_valid=1;@(posedge clk);@(negedge clk);command_valid=0;
    wait(done);repeat(2) @(posedge clk);#1;
    if(protocol_error || accepted_tiles!=24 || bytes_read!=24*384 || busy)
      $fatal(1,"weight slice completion mismatch");
    if(address_stalls==0)
      $fatal(1,"weight slice did not exercise AXI address backpressure");
    $display("tb_mdlm_weight_slice_dma: PASS tiles=%0d bytes=%0d cycles=%0d",
      accepted_tiles,bytes_read,cycle);
    $finish;
  end
  initial begin repeat(2000) @(posedge clk);$fatal(1,"timeout");end
endmodule
