`timescale 1ns/1ps

module tb_qkv_head_tile_controller;
  reg clk=0,rst_n=0,start=0,metadata_fire=0,tile_start_ready=1,tile_done=0;
  wire start_ready,metadata_enable,tile_start,busy,done;
  wire [3:0] head,channel_tile;
  wire [1:0] kind;
  wire [2:0] valid_channels;
  wire [11:0] global_row;
  integer expected_kind,expected_tile,start_count=0,done_count=0;

  qkv_head_tile_controller dut(
    .clk(clk),.rst_n(rst_n),.start(start),.start_ready(start_ready),.head_in(5),
    .metadata_enable(metadata_enable),.metadata_fire(metadata_fire),
    .tile_start_ready(tile_start_ready),.tile_start(tile_start),.tile_done(tile_done),
    .active_head(head),.active_kind(kind),.active_channel_tile(channel_tile),
    .active_valid_channels(valid_channels),.active_global_row(global_row),
    .busy(busy),.done(done));
  always #2 clk=~clk;
  always @(posedge clk) begin
    if(tile_start) start_count=start_count+1;
    if(done) done_count=done_count+1;
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;start=1;
    @(negedge clk);start=0;
    for(expected_kind=0;expected_kind<3;expected_kind=expected_kind+1)
      for(expected_tile=0;expected_tile<11;expected_tile=expected_tile+1) begin
        wait(metadata_enable);#1;
        if(head!==5 || kind!==expected_kind || channel_tile!==expected_tile)
          $fatal(1,"QKV controller tag mismatch");
        if(global_row!==(expected_kind*768+5*64+expected_tile*6))
          $fatal(1,"QKV controller global row mismatch");
        if(valid_channels!==((expected_tile==10)?4:6))
          $fatal(1,"QKV controller valid channel mismatch");
        @(negedge clk);metadata_fire=1;@(negedge clk);metadata_fire=0;
        wait(tile_start);@(negedge clk);tile_done=1;
        @(negedge clk);tile_done=0;
      end
    wait(done);@(posedge clk);#1;
    if(start_count!=33 || done_count!=1)
      $fatal(1,"QKV controller sequence count mismatch");
    if(busy || !start_ready) $fatal(1,"QKV controller did not return idle");
    $display("tb_qkv_head_tile_controller: PASS");$finish;
  end
  initial begin repeat(1000) @(posedge clk);$fatal(1,"timeout");end
endmodule
