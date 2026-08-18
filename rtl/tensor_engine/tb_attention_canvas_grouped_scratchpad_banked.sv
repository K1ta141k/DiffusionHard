`timescale 1ns/1ps
module tb_attention_canvas_grouped_scratchpad_banked;
  reg clk=0,rst_n=0,tile_valid=0,read_valid=0,group_read_valid=0;
  reg [3:0] tile_head=0,tile_group=0,tile_channel_tile=0;
  reg [2:0] tile_valid_channels=0;reg [1943:0] tile_data=0;
  reg [3:0] read_head=0,group_read_head=0,group_read_group=0;
  reg [5:0] read_token=0;wire tile_ready,tile_done,read_data_valid;
  wire group_read_data_valid;wire [1151:0] read_data;
  wire [4607:0] group_read_data;integer token,lane,channel;
  attention_canvas_grouped_scratchpad_banked dut(.clk(clk),.rst_n(rst_n),
    .tile_valid(tile_valid),.tile_ready(tile_ready),.tile_head(tile_head),
    .tile_group(tile_group),.tile_channel_tile(tile_channel_tile),
    .tile_valid_channels(tile_valid_channels),.tile_data_packed(tile_data),
    .tile_done(tile_done),.read_valid(read_valid),.read_head(read_head),
    .read_token(read_token),.read_data_valid(read_data_valid),
    .read_data_packed(read_data),.group_read_valid(group_read_valid),
    .group_read_head(group_read_head),.group_read_group(group_read_group),
    .group_read_data_valid(group_read_data_valid),
    .group_read_data_packed(group_read_data));
  always #2 clk=~clk;
  task load_tile(input [3:0] tile);
    begin
      @(negedge clk);tile_valid=1;tile_head=2;tile_group=3;
      tile_channel_tile=tile;tile_valid_channels=6;
      for(token=0;token<4;token=token+1)
        for(lane=0;lane<6;lane=lane+1)
          tile_data[(token*6+lane)*18+:18]=token*100+tile*6+lane;
      @(negedge clk);tile_valid=0;
    end
  endtask
  initial begin
    repeat(3)@(posedge clk);@(negedge clk);rst_n=1;
    load_tile(0);load_tile(1);
    @(negedge clk);read_valid=1;read_head=2;read_token={4'd3,2'd2};
    @(negedge clk);read_valid=0;wait(read_data_valid);#1;
    for(channel=0;channel<12;channel=channel+1)
      if(read_data[channel*18+:18]!==200+channel)
        $fatal(1,"grouped canvas token mismatch channel=%0d",channel);
    @(negedge clk);group_read_valid=1;group_read_head=2;group_read_group=3;
    @(negedge clk);group_read_valid=0;wait(group_read_data_valid);#1;
    for(token=0;token<4;token=token+1)
      for(channel=0;channel<12;channel=channel+1)
        if(group_read_data[(token*64+channel)*18+:18]!==token*100+channel)
          $fatal(1,"grouped canvas mismatch token=%0d channel=%0d",token,channel);
    $display("tb_attention_canvas_grouped_scratchpad_banked: PASS");$finish;
  end
  initial begin repeat(200)@(posedge clk);$fatal(1,"timeout");end
endmodule
