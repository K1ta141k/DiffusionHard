`timescale 1ns/1ps

module tb_ddit_block_shared_mac;
  reg clk=0,rst_n=0,mlp_phase=0;
  reg attention_valid=0,attention_clear=0,attention_last=0;
  reg [7:0] attention_tag=0;
  reg [2303:0] attention_activations=0;
  reg [3455:0] attention_weights=0;
  reg mlp_valid=0,mlp_clear=0,mlp_last=0;
  reg [15:0] mlp_tag=0;
  reg [1023:0] mlp_activations=0;
  reg [1535:0] mlp_weights=0;
  wire attention_response_valid,mlp_response_valid;
  wire [7:0] attention_response_tag;
  wire [15:0] mlp_response_tag;
  wire [1151:0] attention_accumulators;
  wire [767:0] mlp_accumulators;
  integer index,attention_seen=0,mlp_seen=0;

  ddit_block_shared_mac dut(
    .clk(clk),.rst_n(rst_n),.mlp_phase(mlp_phase),
    .attention_request_valid(attention_valid),
    .attention_request_clear(attention_clear),
    .attention_request_last(attention_last),
    .attention_request_tag(attention_tag),
    .attention_request_activations(attention_activations),
    .attention_request_weights(attention_weights),
    .attention_response_valid(attention_response_valid),
    .attention_response_tag(attention_response_tag),
    .attention_response_accumulators(attention_accumulators),
    .mlp_request_valid(mlp_valid),.mlp_request_clear(mlp_clear),
    .mlp_request_last(mlp_last),.mlp_request_tag(mlp_tag),
    .mlp_request_activations(mlp_activations),
    .mlp_request_weights(mlp_weights),.mlp_response_valid(mlp_response_valid),
    .mlp_response_tag(mlp_response_tag),
    .mlp_response_accumulators(mlp_accumulators));

  always #2 clk=~clk;
  always @(posedge clk) begin
    #1;
    if(attention_response_valid) begin
      if(attention_response_tag!==8'h5a) $fatal(1,"attention tag mismatch");
      for(index=0;index<24;index=index+1)
        if($signed(attention_accumulators[index*48 +: 48])!==32)
          $fatal(1,"attention accumulator mismatch");
      attention_seen=attention_seen+1;
    end
    if(mlp_response_valid) begin
      if(mlp_response_tag!==16'hc123) $fatal(1,"MLP tag mismatch");
      for(index=0;index<24;index=index+1)
        if($signed(mlp_accumulators[index*32 +: 32])!==-192)
          $fatal(1,"MLP accumulator mismatch");
      mlp_seen=mlp_seen+1;
    end
  end

  initial begin
    for(index=0;index<128;index=index+1)
      attention_activations[index*18 +: 18]=18'sd1;
    for(index=0;index<192;index=index+1)
      attention_weights[index*18 +: 18]=18'sd1;
    for(index=0;index<128;index=index+1)
      mlp_activations[index*8 +: 8]=-8'sd2;
    for(index=0;index<192;index=index+1)
      mlp_weights[index*8 +: 8]=8'sd3;
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    attention_valid=1;attention_clear=1;attention_last=1;attention_tag=8'h5a;
    @(negedge clk);attention_valid=0;attention_clear=0;attention_last=0;
    mlp_phase=1;mlp_valid=1;mlp_clear=1;mlp_last=1;mlp_tag=16'hc123;
    @(negedge clk);mlp_valid=0;mlp_clear=0;mlp_last=0;
    wait(attention_seen==1 && mlp_seen==1);repeat(3) @(posedge clk);
    $display("tb_ddit_block_shared_mac: PASS");$finish;
  end
  initial begin repeat(100) @(posedge clk);$fatal(1,"timeout");end
endmodule
