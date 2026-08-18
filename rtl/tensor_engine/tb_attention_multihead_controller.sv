`timescale 1ns/1ps

module tb_attention_multihead_controller;
  reg clk=0,rst_n=0,block_start=0,load_fire=0;
  reg head_start_ready=1,head_done=0,canvas_idle=1;
  wire block_start_ready,load_enable,head_start,busy,done;
  wire [3:0] expected_head;
  integer head,load_index,start_count=0,done_count=0;

  attention_multihead_controller #(
    .HEADS(12),.LOADS_PER_HEAD(4)
  ) dut(
    .clk(clk),.rst_n(rst_n),.block_start(block_start),
    .block_start_ready(block_start_ready),.load_fire(load_fire),
    .load_enable(load_enable),.expected_head(expected_head),
    .head_start_ready(head_start_ready),.head_start(head_start),
    .head_done(head_done),.canvas_idle(canvas_idle),.busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    if(head_start) start_count=start_count+1;
    if(done) done_count=done_count+1;
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    block_start=1;@(negedge clk);block_start=0;
    for(head=0;head<12;head=head+1) begin
      wait(load_enable);
      if(expected_head!==head) $fatal(1,"expected head mismatch");
      for(load_index=0;load_index<4;load_index=load_index+1) begin
        @(negedge clk);load_fire=1;
      end
      @(negedge clk);load_fire=0;
      wait(head_start);@(negedge clk);
      head_done=1;@(negedge clk);head_done=0;
      if(head==11) wait(done); else wait(load_enable);
    end
    @(posedge clk);#1;
    if(start_count!=12) $fatal(1,"missing head starts");
    if(done_count!=1) $fatal(1,"missing block done pulse");
    if(busy || !block_start_ready) $fatal(1,"controller did not return idle");
    $display("tb_attention_multihead_controller: PASS");$finish;
  end
  initial begin repeat(500) @(posedge clk);$fatal(1,"timeout");end
endmodule
