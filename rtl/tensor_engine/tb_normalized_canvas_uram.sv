`timescale 1ns/1ps

module tb_normalized_canvas_uram;
  reg clk=0,load_valid=0,read_valid=0;
  reg [3:0] load_group=0,read_group=0;
  reg [9:0] load_channel=0;
  reg [71:0] load_data=0;
  reg [4:0] read_tile=0;
  wire read_data_valid;
  wire [2303:0] read_data;
  integer group,channel,lane,bank;

  function integer expected(input integer g,input integer t,input integer c);
    expected=g*3000+t*700+c-20000;
  endfunction

  normalized_canvas_uram dut(
    .clk(clk),.load_valid(load_valid),.load_group(load_group),
    .load_channel(load_channel),.load_q12_packed(load_data),
    .read_valid(read_valid),.read_group(read_group),
    .read_input_tile(read_tile),.read_data_valid(read_data_valid),
    .read_q12_packed(read_data));
  always #2 clk=~clk;

  initial begin
    for(group=0;group<16;group=group+1)
      for(channel=0;channel<768;channel=channel+1) begin
        @(negedge clk);load_valid=1;load_group=group;load_channel=channel;
        for(lane=0;lane<4;lane=lane+1)
          load_data[lane*18 +: 18]=expected(group,lane,channel);
      end
    @(negedge clk);load_valid=0;read_valid=1;read_group=9;read_tile=17;
    @(posedge clk);#1;
    if(!read_data_valid) $fatal(1,"normalized canvas read valid missing");
    for(lane=0;lane<4;lane=lane+1)
      for(bank=0;bank<32;bank=bank+1)
        if($signed(read_data[(lane*32+bank)*18 +: 18])!==
           expected(9,lane,17*32+bank))
          $fatal(1,"normalized canvas mismatch lane %0d bank %0d",lane,bank);
    $display("tb_normalized_canvas_uram: PASS");$finish;
  end
  initial begin repeat(20000) @(posedge clk);$fatal(1,"timeout");end
endmodule
