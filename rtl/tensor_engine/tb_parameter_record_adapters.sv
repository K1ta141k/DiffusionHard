`timescale 1ns/1ps

module tb_parameter_record_adapters;
  reg clk=0,rst_n=0;
  reg a_valid=0,a_last=0,a_ready=0;
  reg [511:0] a_data=0;
  reg [15:0] a_tag=0;
  reg [5:0] a_offset=0;
  reg [9:0] a_payload=168;
  wire a_stream_ready,a_record_valid,a_error;
  wire [1343:0] a_record;
  wire [15:0] a_record_tag;
  reg c_valid=0,c_last=0,c_ready=0;
  reg [511:0] c_data=0;
  reg [15:0] c_tag=0;
  reg [5:0] c_offset=60;
  reg [9:0] c_payload=3;
  wire c_stream_ready,c_record_valid,c_error;
  wire [17:0] c_record;
  wire [15:0] c_record_tag;
  integer beat;

  fixed_aligned_record_adapter #(.RECORD_WIDTH(1344),
    .EXPECTED_PAYLOAD_BYTES(168)) aligned(
    .clk(clk),.rst_n(rst_n),.stream_valid(a_valid),.stream_ready(a_stream_ready),
    .stream_data(a_data),.stream_last(a_last),.stream_tag(a_tag),
    .stream_record_byte_offset(a_offset),.stream_record_payload_bytes(a_payload),
    .record_valid(a_record_valid),.record_ready(a_ready),.record_data(a_record),
    .record_tag(a_record_tag),.protocol_error(a_error));
  compact_table_record_adapter compact(
    .clk(clk),.rst_n(rst_n),.stream_valid(c_valid),.stream_ready(c_stream_ready),
    .stream_data(c_data),.stream_last(c_last),.stream_tag(c_tag),
    .stream_record_byte_offset(c_offset),.stream_record_payload_bytes(c_payload),
    .record_valid(c_record_valid),.record_ready(c_ready),.record_data(c_record),
    .record_tag(c_record_tag),.protocol_error(c_error));

  always #2 clk=~clk;
  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    a_tag=16'h1234;a_valid=1;
    for(beat=0;beat<3;beat=beat+1) begin
      a_data=512'h1000+beat;a_last=beat==2;
      wait(a_stream_ready);@(posedge clk);@(negedge clk);
    end
    a_valid=0;a_last=0;repeat(2) @(posedge clk);#1;
    if(!a_record_valid || a_record_tag!==16'h1234 || a_error)
      $fatal(1,"aligned record missing");
    if(a_record[511:0]!==512'h1000 || a_record[1023:512]!==512'h1001 ||
       a_record[1343:1024]!==320'h1002)
      $fatal(1,"aligned record packing mismatch");
    a_ready=1;@(posedge clk);@(negedge clk);a_ready=0;

    c_tag=16'h5678;c_data=0;c_data[511:480]=32'hfffc1234;
    c_last=1;c_valid=1;wait(c_stream_ready);@(posedge clk);@(negedge clk);
    c_valid=0;c_last=0;repeat(2) @(posedge clk);#1;
    if(!c_record_valid || c_record_tag!==16'h5678 ||
       c_record!==18'h01234 || c_error)
      $fatal(1,"compact record extraction mismatch");
    c_ready=1;@(posedge clk);@(negedge clk);c_ready=0;

    a_data=512'hdead;a_last=1;a_payload=167;a_valid=1;
    wait(a_stream_ready);@(posedge clk);@(negedge clk);a_valid=0;a_last=0;
    repeat(2) @(posedge clk);#1;
    if(!a_error || a_record_valid)
      $fatal(1,"bad aligned geometry was not rejected");
    $display("tb_parameter_record_adapters: PASS");
    $finish;
  end
endmodule
