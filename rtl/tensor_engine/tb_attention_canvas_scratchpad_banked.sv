`timescale 1ns/1ps

module tb_attention_canvas_scratchpad_banked;
  reg clk=0,rst_n=0,tile_valid=0,read_valid=0;
  wire tile_ready,tile_done,read_data_valid;
  reg [3:0] tile_head=0,tile_group=0,tile_channel_tile=0,read_head=0;
  reg [2:0] tile_valid_channels=0;
  reg [431:0] tile_data_packed=0;
  reg [5:0] read_token=0;
  wire [1151:0] read_data_packed;
  integer head,group,tile,token,lane,channel;
  attention_canvas_scratchpad_banked dut(
    .clk(clk),.rst_n(rst_n),.tile_valid(tile_valid),.tile_ready(tile_ready),
    .tile_head(tile_head),.tile_group(tile_group),
    .tile_channel_tile(tile_channel_tile),
    .tile_valid_channels(tile_valid_channels),.tile_data_packed(tile_data_packed),
    .tile_done(tile_done),.read_valid(read_valid),.read_head(read_head),
    .read_token(read_token),.read_data_valid(read_data_valid),
    .read_data_packed(read_data_packed));
  always #2 clk=~clk;
  function integer expected(input integer h,input integer t,input integer c);
    expected=h*7000+t*70+c-40000;
  endfunction
  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(head=0;head<12;head=head+1)
      for(group=0;group<16;group=group+1)
        for(tile=0;tile<11;tile=tile+1) begin
          wait(tile_ready);@(negedge clk);tile_valid=1;tile_head=head;
          tile_group=group;tile_channel_tile=tile;
          tile_valid_channels=(tile==10)?4:6;tile_data_packed=0;
          for(token=0;token<4;token=token+1)
            for(lane=0;lane<tile_valid_channels;lane=lane+1) begin
              channel=tile*6+lane;
              tile_data_packed[(token*6+lane)*18 +: 18]=
                expected(head,group*4+token,channel);
            end
          @(negedge clk);tile_valid=0;
        end
    wait(tile_ready);@(negedge clk);read_valid=1;read_head=7;read_token=37;
    @(posedge clk);#1;
    if(!read_data_valid) $fatal(1,"canvas read valid missing");
    for(channel=0;channel<64;channel=channel+1)
      if($signed(read_data_packed[channel*18 +: 18])!==expected(7,37,channel))
        $fatal(1,"canvas mismatch channel %0d",channel);
    @(negedge clk);read_valid=0;
    $display("tb_attention_canvas_scratchpad_banked: PASS");$finish;
  end
  initial begin repeat(20000) @(posedge clk);$fatal(1,"timeout");end
endmodule
