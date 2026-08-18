`timescale 1ns/1ps

module tb_fixed_weight_record_adapter;
  reg clk=0,rst_n=0;
  reg q_valid=0,q_last=0,q_ready=0;
  reg [511:0] q_data=0;
  reg [15:0] q_tag=0;
  wire q_stream_ready,q_record_valid,q_error;
  wire [3071:0] q_record;
  wire [15:0] q_record_tag;
  reg i_valid=0,i_last=0,i_ready=0;
  reg [511:0] i_data=0;
  reg [15:0] i_tag=0;
  wire i_stream_ready,i_record_valid,i_error;
  wire [1535:0] i_record;
  wire [15:0] i_record_tag;
  integer beat;

  fixed_weight_record_adapter #(.DATA_WIDTH(16)) qkv(
    .clk(clk),.rst_n(rst_n),.stream_valid(q_valid),.stream_ready(q_stream_ready),
    .stream_data(q_data),.stream_last(q_last),.stream_tag(q_tag),
    .record_valid(q_record_valid),.record_ready(q_ready),.record_data(q_record),
    .record_tag(q_record_tag),.protocol_error(q_error));
  fixed_weight_record_adapter #(.DATA_WIDTH(8)) int8_weights(
    .clk(clk),.rst_n(rst_n),.stream_valid(i_valid),.stream_ready(i_stream_ready),
    .stream_data(i_data),.stream_last(i_last),.stream_tag(i_tag),
    .record_valid(i_record_valid),.record_ready(i_ready),.record_data(i_record),
    .record_tag(i_record_tag),.protocol_error(i_error));

  always #2 clk=~clk;

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    q_tag=16'h1234;q_valid=1;
    for(beat=0;beat<6;beat=beat+1) begin
      q_data=512'h1000+beat;q_last=beat==5;
      wait(q_stream_ready);@(posedge clk);@(negedge clk);
    end
    q_valid=0;q_last=0;repeat(2) @(posedge clk);#1;
    if(!q_record_valid || q_record_tag!==16'h1234 || q_error)
      $fatal(1,"QKV record output missing");
    for(beat=0;beat<6;beat=beat+1)
      if(q_record[beat*512 +: 512]!==512'h1000+beat)
        $fatal(1,"QKV beat order mismatch");
    q_ready=1;@(posedge clk);@(negedge clk);q_ready=0;

    i_tag=16'h5678;i_valid=1;i_ready=0;
    for(beat=0;beat<3;beat=beat+1) begin
      i_data=512'h2000+beat;i_last=beat==2;
      wait(i_stream_ready);@(posedge clk);@(negedge clk);
    end
    i_valid=0;i_last=0;repeat(2) @(posedge clk);#1;
    if(!i_record_valid || i_record_tag!==16'h5678 || i_error)
      $fatal(1,"INT8 record output missing");
    for(beat=0;beat<3;beat=beat+1)
      if(i_record[beat*512 +: 512]!==512'h2000+beat)
        $fatal(1,"INT8 beat order mismatch");
    i_ready=1;@(posedge clk);@(negedge clk);i_ready=0;

    q_tag=16'h9999;q_data=512'hdead;q_last=1;q_valid=1;
    wait(q_stream_ready);@(posedge clk);@(negedge clk);q_valid=0;q_last=0;
    repeat(2) @(posedge clk);#1;
    if(!q_error || q_record_valid)
      $fatal(1,"early-last record was not discarded");
    $display("tb_fixed_weight_record_adapter: PASS");
    $finish;
  end
endmodule
