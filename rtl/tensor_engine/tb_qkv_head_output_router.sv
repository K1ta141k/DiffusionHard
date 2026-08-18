`timescale 1ns/1ps

module tb_qkv_head_output_router;
  reg clk=0,rst_n=0,tile_valid=0;
  reg [3:0] tile_head=0,tile_group=0,tile_channel_tile=0;
  reg [1:0] tile_kind=0;
  reg [2:0] tile_valid_channels=0;
  reg [431:0] tile_data=0;
  wire tile_ready,q_load,k_load,v_load,tile_done,busy;
  wire [3:0] done_head,done_group,done_channel_tile;
  wire [1:0] done_kind;
  wire [3:0] load_head;
  wire [5:0] load_token,load_channel;
  wire signed [17:0] load_data;
  integer kind,token,lane,write_count=0;

  qkv_head_output_router dut(
    .clk(clk),.rst_n(rst_n),.tile_valid(tile_valid),.tile_ready(tile_ready),
    .tile_head(tile_head),.tile_kind(tile_kind),.tile_group(tile_group),
    .tile_channel_tile(tile_channel_tile),
    .tile_valid_channels(tile_valid_channels),.tile_q12_packed(tile_data),
    .query_load_valid(q_load),.key_load_valid(k_load),.value_load_valid(v_load),
    .load_head(load_head),.load_token(load_token),.load_channel(load_channel),
    .load_q12(load_data),.tile_done(tile_done),.done_head(done_head),
    .done_kind(done_kind),.done_group(done_group),
    .done_channel_tile(done_channel_tile),.busy(busy));
  always #2 clk=~clk;
  always @(posedge clk) begin
    #1;
    if(q_load || k_load || v_load) begin
      if((q_load+k_load+v_load)!==1) $fatal(1,"router asserted multiple kinds");
      if(load_head!==7 || load_token!=={4'd9,token[1:0]} ||
         load_channel!==60+lane || $signed(load_data)!==(kind*1000+token*10+lane))
        $fatal(1,"router output mismatch");
      write_count=write_count+1;
      lane=lane+1;
      if(lane==4) begin lane=0;token=token+1;end
    end
  end
  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(kind=0;kind<3;kind=kind+1) begin
      token=0;lane=0;tile_data=0;
      for(token=0;token<4;token=token+1)
        for(lane=0;lane<4;lane=lane+1)
          tile_data[(token*6+lane)*18 +: 18]=kind*1000+token*10+lane;
      token=0;lane=0;wait(tile_ready);@(negedge clk);tile_valid=1;
      tile_head=7;tile_kind=kind;tile_group=9;tile_channel_tile=10;
      tile_valid_channels=4;@(negedge clk);tile_valid=0;wait(tile_done);
      if(done_head!==7 || done_kind!==kind || done_group!==9 ||
         done_channel_tile!==10) $fatal(1,"router completion tag mismatch");
    end
    repeat(2) @(posedge clk);
    if(write_count!=48) $fatal(1,"router write count mismatch");
    if(busy) $fatal(1,"router remained busy");
    $display("tb_qkv_head_output_router: PASS");$finish;
  end
  initial begin repeat(500) @(posedge clk);$fatal(1,"timeout");end
endmodule
