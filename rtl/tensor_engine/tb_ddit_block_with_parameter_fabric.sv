`timescale 1ns/1ps

module tb_ddit_block_with_parameter_fabric;
  reg clk=0,rst_n=0,block_start=0,residual_load=0,constant_load=0;
  reg [3:0] residual_group=0;reg [6:0] residual_tile=0;
  reg [5:0] constant_token=0;reg [4:0] constant_pair=0;
  reg normalized_data_valid=0,arready=0,rvalid=0,rlast=0;
  wire block_start_ready,busy,done,normalized_read_valid;
  wire [3:0] normalized_group;wire [4:0] normalized_tile;
  wire [9:0] reciprocal_channel,output_tile;wire [0:0] output_group;
  wire output_valid;wire [575:0] outputs;
  wire [63:0] araddr,transactions;wire [7:0] arlen;
  wire [2:0] arsize;wire [1:0] arburst;wire arvalid,rready;
  wire parameter_error,attention_busy,mlp_busy;wire [255:0] client_bytes;
  integer index,cycles=0,attention_tiles=0,outputs_seen=0;
  integer source_beat=0,source_beats=0;
  reg saw_attention=0,saw_mlp=0;

  ddit_block_with_parameter_fabric #(.HEADS(1),
    .ATTENTION_OUTPUT_TILES(2),.TOKENS(4),.DOWN_INPUT_SIZE(768),
    .DOWN_OUTPUT_SIZE(6)) dut(
    .clk(clk),.rst_n(rst_n),.block_base_address(0),.block_start(block_start),
    .block_start_ready(block_start_ready),.busy(busy),.done(done),
    .residual_load_valid(residual_load),.residual_load_group(residual_group),
    .residual_load_output_tile(residual_tile),.residual_load_q10_packed(0),
    .normalized_read_valid(normalized_read_valid),
    .normalized_read_group(normalized_group),
    .normalized_read_input_tile(normalized_tile),
    .normalized_read_data_valid(normalized_data_valid),.normalized_q12_packed(0),
    .constant_load_valid(constant_load),.constant_load_token(constant_token),
    .constant_load_pair(constant_pair),.constant_load_cosine_q15(16'sd32767),
    .constant_load_sine_q15(0),.smoothing_reciprocal_q15(18'd32768),
    .smoothing_reciprocal_channel(reciprocal_channel),
    .output_valid(output_valid),.output_tile(output_tile),
    .output_group(output_group),.outputs_packed(outputs),
    .m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),
    .m_axi_arburst(arburst),.m_axi_arvalid(arvalid),.m_axi_arready(arready),
    .m_axi_rdata(0),.m_axi_rresp(0),.m_axi_rlast(rlast),
    .m_axi_rvalid(rvalid),.m_axi_rready(rready),
    .parameter_protocol_error(parameter_error),.client_bytes_read(client_bytes),
    .read_transactions(transactions),.attention_busy(attention_busy),
    .mlp_busy(mlp_busy));

  genvar canvas_bank;
  generate for(canvas_bank=0;canvas_bank<64;canvas_bank=canvas_bank+1) begin: pad_heads
    integer canvas_address;
    initial for(canvas_address=16;canvas_address<192;canvas_address=canvas_address+1)
      dut.compute.attention.producer.canvas.canvas_banks[canvas_bank]
        .memory[canvas_address]=0;
  end endgenerate

  always #2 clk=~clk;
  always @(posedge clk) begin
    cycles=cycles+1;normalized_data_valid<=normalized_read_valid;
    arready<=arvalid && cycles[0];
    if(arvalid && arready) begin
      if(arsize!==6 || arburst!==1) $fatal(1,"fabric AXI attributes mismatch");
      source_beat=0;source_beats=arlen+1;rvalid<=1;rlast<=source_beats==1;
    end else if(rvalid && rready) begin
      source_beat=source_beat+1;
      if(source_beat==source_beats) begin rvalid<=0;rlast<=0;end
      else rlast<=source_beat==source_beats-1;
    end
    if(attention_busy) saw_attention=1;if(mlp_busy) saw_mlp=1;
    if(attention_busy && mlp_busy) $fatal(1,"DDiT phases overlapped");
    #1;
    if(dut.attention_tile_valid) begin
      if(dut.attention_tile_group!==(attention_tiles%16) ||
         dut.attention_tile_output!==(attention_tiles/16) ||
         dut.attention_tile_data!==0) $fatal(1,"fabric attention mismatch");
      attention_tiles=attention_tiles+1;
    end
    if(output_valid) begin
      if(output_tile!==0 || output_group!==0 || outputs!==0)
        $fatal(1,"fabric final output mismatch");
      outputs_seen=outputs_seen+1;
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(index=0;index<128;index=index+1) begin
      @(negedge clk);residual_load=1;residual_group=0;residual_tile=index;
    end
    for(index=0;index<30;index=index+1) begin
      @(negedge clk);residual_load=1;residual_group=1+index%15;
      residual_tile=index/15;
    end
    @(negedge clk);residual_load=0;
    for(index=0;index<2048;index=index+1) begin
      @(negedge clk);constant_load=1;constant_token=index/32;constant_pair=index%32;
    end
    @(negedge clk);constant_load=0;block_start=1;
    wait(block_start_ready);@(posedge clk);@(negedge clk);block_start=0;
    wait(done);repeat(4) @(posedge clk);#1;
    if(parameter_error || !saw_attention || !saw_mlp || attention_tiles!=32 ||
       outputs_seen!=1 || busy || transactions!=4100 ||
       client_bytes[63:0]!=306240 || client_bytes[127:64]!=9344 ||
       client_bytes[191:128]!=598016 || client_bytes[255:192]!=4800)
      $fatal(1,"fabric DDiT completion mismatch tx=%0d bytes=%0d/%0d/%0d/%0d",
        transactions,client_bytes[63:0],client_bytes[127:64],
        client_bytes[191:128],client_bytes[255:192]);
    $display("tb_ddit_block_with_parameter_fabric: PASS cycles=%0d transactions=%0d bytes=%0d",
      cycles,transactions,client_bytes[63:0]+client_bytes[127:64]+
      client_bytes[191:128]+client_bytes[255:192]);
    $finish;
  end
  initial begin repeat(120000) @(posedge clk);
    $display("timeout cycles=%0d block_state=%0d attention_state=%0d mlp_state=%0d tx=%0d client_busy=%b bytes=%0d/%0d/%0d/%0d",
      cycles,dut.compute.state,dut.compute.attention.state,dut.compute.mlp.controller.state,
      transactions,dut.loader_busy,client_bytes[63:0],client_bytes[127:64],
      client_bytes[191:128],client_bytes[255:192]);
    $display("qkv req=%0d/%0d/%0d meta=%b/%b weight=%b/%b loader_state=%0d projection req=%0d meta=%b/%b loader_state=%0d",
      dut.requested_qkv_head,dut.requested_qkv_kind,dut.requested_qkv_channel,
      dut.qkv_meta_valid,dut.qkv_meta_ready,dut.qkv_weight_valid,dut.qkv_weight_ready,
      dut.fabric.qkv.state,dut.requested_projection_tile,
      dut.projection_meta_valid,dut.projection_meta_ready,dut.fabric.projection.state);
    $display("up req=%0d meta=%b/%b weight=%b/%b loader_state=%0d down req=%0d meta=%b/%b weight=%b/%b loader_state=%0d axi=%b/%b r=%b/%b",
      dut.requested_up_tile,dut.up_meta_valid,dut.up_meta_ready,
      dut.up_weight_valid,dut.up_weight_ready,dut.fabric.up.state,
      dut.requested_down_tile,dut.down_meta_valid,dut.down_meta_ready,
      dut.down_weight_valid,dut.down_weight_ready,dut.fabric.down.state,
      arvalid,arready,rvalid,rready);
    $fatal(1,"timeout");
  end
endmodule
