`timescale 1ns/1ps

module tb_ddit_block_with_image_fabric;
  reg clk=0,rst_n=0,preload_start=0,block_start=0,residual_load=0;
  reg [3:0] residual_group=0;reg [6:0] residual_tile=0;
  reg normalized_data_valid=0,arready=0,rvalid=0,rlast=0;
  reg [511:0] rdata=0;
  wire preload_start_ready,preload_done,constants_loaded,block_start_ready;
  wire busy,done,normalized_read_valid;
  wire [3:0] normalized_group;wire [4:0] normalized_tile;
  wire [9:0] output_tile;wire [0:0] output_group;
  wire output_valid;wire [575:0] outputs;
  wire [63:0] araddr,transactions,constant_transactions,constant_bytes;
  wire [63:0] dense_transactions;wire [7:0] arlen;
  wire [2:0] arsize;wire [1:0] arburst;wire arvalid,rready;
  wire protocol_error,attention_busy,mlp_busy;wire [255:0] client_bytes;
  integer index,cycles=0,attention_tiles=0,outputs_seen=0;
  integer source_beat=0,source_beats=0;
  reg saw_preload=0,saw_attention=0,saw_mlp=0;

  ddit_block_with_image_fabric #(.HEADS(1),
    .ATTENTION_OUTPUT_TILES(2),.TOKENS(4),.DOWN_INPUT_SIZE(768),
    .DOWN_OUTPUT_SIZE(6),.INTERNAL_NORM1(1)) dut(
    .clk(clk),.rst_n(rst_n),.block_base_address(0),
    .preload_start(preload_start),.preload_start_ready(preload_start_ready),
    .preload_done(preload_done),.constants_loaded(constants_loaded),
    .block_start(block_start),.block_start_ready(block_start_ready),
    .busy(busy),.done(done),.residual_load_valid(residual_load),
    .residual_load_group(residual_group),
    .residual_load_output_tile(residual_tile),.residual_load_q10_packed(0),
    .normalized_read_valid(normalized_read_valid),
    .normalized_read_group(normalized_group),
    .normalized_read_input_tile(normalized_tile),
    .normalized_read_data_valid(normalized_data_valid),.normalized_q12_packed(0),
    .output_valid(output_valid),.output_tile(output_tile),
    .output_group(output_group),.outputs_packed(outputs),
    .m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),
    .m_axi_arburst(arburst),.m_axi_arvalid(arvalid),.m_axi_arready(arready),
    .m_axi_rdata(rdata),.m_axi_rresp(0),.m_axi_rlast(rlast),
    .m_axi_rvalid(rvalid),.m_axi_rready(rready),
    .protocol_error(protocol_error),.read_transactions(transactions),
    .constant_read_transactions(constant_transactions),
    .constant_bytes_read(constant_bytes),
    .dense_client_bytes_read(client_bytes),
    .dense_read_transactions(dense_transactions),
    .attention_busy(attention_busy),.mlp_busy(mlp_busy));

  genvar canvas_bank;
  generate for(canvas_bank=0;canvas_bank<64;canvas_bank=canvas_bank+1) begin: pad_heads
    integer canvas_address;
    initial for(canvas_address=16;canvas_address<192;canvas_address=canvas_address+1)
      dut.dense.compute.attention.producer.canvas.canvas_banks[canvas_bank]
        .memory[canvas_address]=0;
  end endgenerate

  always #2 clk=~clk;
  always @(posedge clk) begin
    cycles=cycles+1;normalized_data_valid<=0;
    arready<=arvalid && cycles[0];
    if(arvalid && arready) begin
      if(arsize!==6 || arburst!==1) $fatal(1,"image AXI attributes mismatch");
      source_beat=0;source_beats=arlen+1;rvalid<=1;rlast<=source_beats==1;
      if(araddr>=64'd3678208 && araddr<64'd3686400)
        rdata<={16{32'h00007fff}};
      else if(araddr>=64'd4284416 && araddr<64'd4287488)
        rdata<={16{32'h00008000}};
      else rdata<=0;
    end else if(rvalid && rready) begin
      source_beat=source_beat+1;
      if(source_beat==source_beats) begin rvalid<=0;rlast<=0;end
      else rlast<=source_beat==source_beats-1;
    end
    if(dut.preloader_busy) saw_preload=1;
    if(attention_busy) saw_attention=1;if(mlp_busy) saw_mlp=1;
    if(dut.preloader_busy && dut.dense_busy)
      $fatal(1,"constant preload and dense compute overlapped");
    if(attention_busy && mlp_busy) $fatal(1,"DDiT phases overlapped");
    #1;
    if(dut.dense.attention_tile_valid) begin
      if(dut.dense.attention_tile_group!==(attention_tiles%16) ||
         dut.dense.attention_tile_output!==(attention_tiles/16) ||
         dut.dense.attention_tile_data!==0) $fatal(1,"image attention mismatch");
      attention_tiles=attention_tiles+1;
    end
    if(output_valid) begin
      if(output_tile!==0 || output_group!==0 || outputs!==0)
        $fatal(1,"image final output mismatch");
      outputs_seen=outputs_seen+1;
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    if(block_start_ready || constants_loaded)
      $fatal(1,"block became ready before constant preload");
    preload_start=1;wait(preload_start_ready);@(posedge clk);
    @(negedge clk);preload_start=0;
    wait(preload_done);@(posedge clk);#1;
    if(!constants_loaded || !block_start_ready)
      $fatal(1,"block did not become ready after preload");
    for(index=0;index<2048;index=index+1) begin
      @(negedge clk);residual_load=1;residual_group=index/128;
      residual_tile=index%128;
    end
    @(negedge clk);residual_load=0;block_start=1;
    wait(block_start_ready);@(posedge clk);@(negedge clk);block_start=0;
    wait(done);repeat(4) @(posedge clk);#1;
    if(protocol_error || !saw_preload || !saw_attention || !saw_mlp ||
       attention_tiles!=32 || outputs_seen!=1 || busy ||
       constant_transactions!=176 || constant_bytes!=11264 ||
       dense_transactions!=4100 || transactions!=4276 ||
       client_bytes[63:0]!=306240 || client_bytes[127:64]!=9344 ||
       client_bytes[191:128]!=598016 || client_bytes[255:192]!=4800)
      $fatal(1,"image completion mismatch total_tx=%0d constant=%0d/%0d dense=%0d bytes=%0d/%0d/%0d/%0d",
        transactions,constant_transactions,constant_bytes,dense_transactions,
        client_bytes[63:0],client_bytes[127:64],
        client_bytes[191:128],client_bytes[255:192]);
    $display("tb_ddit_block_with_image_fabric: PASS cycles=%0d transactions=%0d bytes=%0d constants=%0d dense=%0d",
      cycles,transactions,constant_bytes+client_bytes[63:0]+
      client_bytes[127:64]+client_bytes[191:128]+client_bytes[255:192],
      constant_bytes,client_bytes[63:0]+client_bytes[127:64]+
      client_bytes[191:128]+client_bytes[255:192]);
    $finish;
  end
  initial begin repeat(180000) @(posedge clk);
    $display("timeout cycles=%0d constants=%b preloader=%0d dense=%0d tx=%0d constant_tx=%0d dense_tx=%0d",
      cycles,constants_loaded,dut.preloader.state,dut.dense.compute.state,
      transactions,constant_transactions,dense_transactions);
    $fatal(1,"timeout");
  end
endmodule
