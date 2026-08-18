`timescale 1ns/1ps

module tb_mdlm_block_parameter_address_generator;
  reg [63:0] block_base=64'h0000_0000_1000_0000;
  reg [3:0] section=0;
  reg [13:0] record=0;
  wire valid;
  wire [63:0] address;
  wire [6:0] beats;
  wire [5:0] byte_offset;
  wire [9:0] payload_bytes;

  mdlm_block_parameter_address_generator dut(
    .block_base_address(block_base),.section_id(section),
    .record_index(record),.record_valid(valid),.axi_address(address),
    .burst_beats(beats),.record_byte_offset(byte_offset),
    .record_payload_bytes(payload_bytes));

  task check;
    input [3:0] selected_section;
    input [13:0] selected_record;
    input [63:0] expected_address;
    input [6:0] expected_beats;
    input [5:0] expected_offset;
    input [9:0] expected_payload;
    begin
      section=selected_section;record=selected_record;#1;
      if(!valid || address!==expected_address || beats!==expected_beats ||
         byte_offset!==expected_offset || payload_bytes!==expected_payload)
        $fatal(1,"address mismatch section=%0d record=%0d",section,record);
    end
  endtask

  initial begin
    check(0,395,block_base+395*64,1,0,32);
    check(1,9503,block_base+28672+9503*384,6,0,384);
    check(2,17,block_base+3678208+64,1,4,4);
    check(3,127,block_base+3686400+127*64,1,0,18);
    check(4,3071,block_base+3694592+3071*192,3,0,192);
    check(5,767,block_base+4284416+47*64,1,60,3);
    check(6,511,block_base+4288512+511*64,1,0,56);
    check(7,12287,block_base+4321280+12287*192,3,0,192);
    check(8,127,block_base+6680576+127*192,3,0,168);
    check(9,12287,block_base+6705152+12287*192,3,0,192);
    section=1;record=9504;#1;
    if(valid || beats!==0 || payload_bytes!==0)
      $fatal(1,"out-of-range record was accepted");
    section=15;record=0;#1;
    if(valid || beats!==0) $fatal(1,"unknown section was accepted");
    $display("tb_mdlm_block_parameter_address_generator: PASS");
    $finish;
  end
endmodule
