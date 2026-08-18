`timescale 1ns/1ps

module tb_ddit_block_kv260_kernel_core;
  reg clk=0,rst_n=0,start=0,arready=0,rvalid=0,rlast=0;
  reg [511:0] rdata=0;reg awready=0,wready=0,bvalid=0;
  wire start_ready,busy,done,error;wire [63:0] cycles;
  wire [63:0] araddr,total_tx,image_tx,constant_bytes,residual_bytes;
  wire [255:0] dense_bytes;wire [7:0] arlen;wire [2:0] arsize;
  wire [1:0] arburst;wire arvalid,rready;wire [63:0] awaddr;
  wire [7:0] awlen;wire [2:0] awsize;wire [1:0] awburst;
  wire awvalid;wire [511:0] wdata;wire [63:0] wstrb;
  wire wlast,wvalid,bready;wire [63:0] output_bytes,output_tx;
  integer wall_cycles=0,source_beat=0,source_beats=0,write_beats=0;
  reg saw_residual=0,saw_image=0,saw_input_overlap=0;
  localparam [63:0] RESIDUAL_BASE=64'h10000000;
  localparam [63:0] OUTPUT_BASE=64'h20000000;
  ddit_block_kv260_kernel_core #(.HEADS(1),.ATTENTION_OUTPUT_TILES(2),
    .TOKENS(4),.DOWN_INPUT_SIZE(768),.DOWN_OUTPUT_SIZE(6)) dut(
    .clk(clk),.rst_n(rst_n),.start(start),.start_ready(start_ready),
    .block_image_base_address(0),.residual_input_base_address(RESIDUAL_BASE),
    .output_base_address(OUTPUT_BASE),.busy(busy),.done(done),
    .protocol_error(error),.kernel_cycles(cycles),.m_axi_araddr(araddr),
    .m_axi_arlen(arlen),.m_axi_arsize(arsize),.m_axi_arburst(arburst),
    .m_axi_arvalid(arvalid),.m_axi_arready(arready),.m_axi_rdata(rdata),
    .m_axi_rresp(0),.m_axi_rlast(rlast),.m_axi_rvalid(rvalid),
    .m_axi_rready(rready),.m_axi_awaddr(awaddr),.m_axi_awlen(awlen),
    .m_axi_awsize(awsize),.m_axi_awburst(awburst),.m_axi_awvalid(awvalid),
    .m_axi_awready(awready),.m_axi_wdata(wdata),.m_axi_wstrb(wstrb),
    .m_axi_wlast(wlast),.m_axi_wvalid(wvalid),.m_axi_wready(wready),
    .m_axi_bresp(0),.m_axi_bvalid(bvalid),.m_axi_bready(bready),
    .read_transactions(total_tx),.image_read_transactions(image_tx),
    .image_constant_bytes(constant_bytes),.image_dense_bytes(dense_bytes),
    .residual_read_bytes(residual_bytes),.output_write_bytes(output_bytes),
    .output_write_transactions(output_tx));
  genvar canvas_bank;
  generate for(canvas_bank=0;canvas_bank<64;canvas_bank=canvas_bank+1) begin: pad_heads
    integer canvas_address;
    initial for(canvas_address=16;canvas_address<192;canvas_address=canvas_address+1)
      dut.image_block.dense.compute.attention.producer.canvas
        .canvas_banks[canvas_bank].memory[canvas_address]=0;
  end endgenerate
  always #2 clk=~clk;
  always @(posedge clk) begin
    wall_cycles=wall_cycles+1;arready<=arvalid&&wall_cycles[0];
    awready<=awvalid&&wall_cycles[0];wready<=wvalid;
    if(dut.residual_busy&&dut.image_block.preloader_busy)saw_input_overlap=1;
    if(arvalid&&arready)begin
      if(arsize!==6||arburst!==1)$fatal(1,"kernel read attributes mismatch");
      source_beat=0;source_beats=arlen+1;rvalid<=1;rlast<=source_beats==1;
      if(araddr>=RESIDUAL_BASE)begin rdata<=0;saw_residual=1;end
      else if(araddr>=64'd3678208&&araddr<64'd3686400)
        rdata<={16{32'h00007fff}};
      else if(araddr>=64'd4284416&&araddr<64'd4287488)
        rdata<={16{32'h00008000}};
      else begin rdata<=0;saw_image=1;end
    end else if(rvalid&&rready)begin
      source_beat=source_beat+1;
      if(source_beat==source_beats)begin rvalid<=0;rlast<=0;end
      else rlast<=source_beat==source_beats-1;
    end
    if(awvalid&&awready)begin
      if(awaddr!==OUTPUT_BASE||awlen!==1||awsize!==6||awburst!==1)
        $fatal(1,"kernel output address mismatch");write_beats=0;
    end
    if(wvalid&&wready)begin
      if(wdata!==0)$fatal(1,"kernel output data mismatch");
      if(write_beats==0)begin
        if(wstrb!==64'hffffffffffffffff||wlast)
          $fatal(1,"kernel output first beat mismatch");write_beats=1;
      end else begin
        if(wstrb!==64'hff||!wlast)$fatal(1,"kernel output last beat mismatch");
        write_beats=2;bvalid<=1;
      end
    end
    if(bvalid&&bready)bvalid<=0;
  end
  initial begin repeat(3)@(posedge clk);@(negedge clk);rst_n=1;start=1;
    wait(start_ready);@(posedge clk);@(negedge clk);start=0;wait(done);
    repeat(4)@(posedge clk);#1;
    if(error||busy||!saw_residual||!saw_image||!saw_input_overlap||
      total_tx!=6324||image_tx!=4276||
      residual_bytes!=262144||constant_bytes!=11264||
      dense_bytes[63:0]!=306240||dense_bytes[127:64]!=9344||
      dense_bytes[191:128]!=598016||dense_bytes[255:192]!=4800||
      output_bytes!=72||output_tx!=1||write_beats!=2)
      $fatal(1,"kernel completion mismatch total=%0d image=%0d residual=%0d output=%0d/%0d",
        total_tx,image_tx,residual_bytes,output_bytes,output_tx);
    $display("tb_ddit_block_kv260_kernel_core: PASS cycles=%0d reads=%0d read_bytes=%0d output_bytes=%0d",
      cycles,total_tx,residual_bytes+constant_bytes+dense_bytes[63:0]+
      dense_bytes[127:64]+dense_bytes[191:128]+dense_bytes[255:192],output_bytes);
    $finish;
  end
  initial begin repeat(220000)@(posedge clk);$fatal(1,"timeout");end
endmodule
