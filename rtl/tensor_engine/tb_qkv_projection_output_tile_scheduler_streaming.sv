`timescale 1ns/1ps

module tb_qkv_projection_output_tile_scheduler_streaming;
  reg clk=0,rst_n=0,start=0,weight_valid=0,norm_data_valid=0;
  reg [4:0] weight_tile=0;reg [5:0] drain_count=0;
  wire start_ready,weight_ready,norm_read_valid,qkv_valid,busy,done;
  wire [3:0] norm_group,qkv_group;wire [4:0] norm_tile;
  wire [8:0] qkv_output_tile;wire [431:0] qkv_data;
  wire array_valid,array_clear,array_last;wire [7:0] array_tag;
  wire [2303:0] array_activations;wire [3455:0] array_weights;
  wire qkv_ready=drain_count==0;
  integer cycles=0,reads=0,requests=0,outputs=0,lane;

  qkv_projection_output_tile_scheduler_streaming dut(
    .clk(clk),.rst_n(rst_n),.start(start),.start_ready(start_ready),
    .output_tile_in(9'd7),.multipliers_packed(0),.biases_q12_packed(0),
    .weight_tile_valid(weight_valid),.weight_tile_ready(weight_ready),
    .weight_input_tile(weight_tile),.weight_int16_packed(0),
    .normalized_read_valid(norm_read_valid),
    .normalized_read_group(norm_group),
    .normalized_read_input_tile(norm_tile),
    .normalized_read_data_valid(norm_data_valid),.normalized_q12_packed(0),
    .qkv_tile_valid(qkv_valid),.qkv_tile_ready(qkv_ready),
    .qkv_group(qkv_group),.qkv_output_tile(qkv_output_tile),
    .qkv_q12_packed(qkv_data),.array_request_valid(array_valid),
    .array_request_clear(array_clear),.array_request_last(array_last),
    .array_request_tag(array_tag),.array_request_activations(array_activations),
    .array_request_weights(array_weights),.array_response_valid(0),
    .array_response_tag(0),.array_response_accumulators(0),
    .busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    norm_data_valid<=norm_read_valid;
    if(busy)cycles=cycles+1;
    if(drain_count!=0)drain_count<=drain_count-1'b1;
    if(norm_read_valid)begin
      if(norm_group!==reads/24 || norm_tile!==reads%24)
        $fatal(1,"streaming normalized read order mismatch");
      reads=reads+1;
    end
    if(array_valid)begin
      if(array_clear!==(requests%24==0) || array_last!==(requests%24==23))
        $fatal(1,"streaming MAC boundary mismatch");
      requests=requests+1;
    end
    if(qkv_valid&&qkv_ready)begin
      if(qkv_group!==outputs || qkv_output_tile!==7)
        $fatal(1,"streaming output tag mismatch");
      for(lane=0;lane<24;lane=lane+1)
        if($signed(qkv_data[lane*18 +: 18])!==0)
          $fatal(1,"streaming zero fixture emitted nonzero output");
      outputs=outputs+1;drain_count<=23;
    end
  end

  initial begin
    repeat(3)@(posedge clk);@(negedge clk);rst_n=1;start=1;
    @(negedge clk);start=0;
    for(weight_tile=0;weight_tile<24;weight_tile=weight_tile+1)begin
      wait(weight_ready);@(negedge clk);weight_valid=1;
      @(negedge clk);weight_valid=0;
    end
    wait(done);repeat(3)@(posedge clk);
    if(reads!=384||requests!=384||outputs!=16||busy)
      $fatal(1,"streaming completion mismatch reads=%0d requests=%0d outputs=%0d",
        reads,requests,outputs);
    $display("tb_qkv_projection_output_tile_scheduler_streaming: PASS cycles=%0d",cycles);
    $finish;
  end
  initial begin repeat(1000)@(posedge clk);$fatal(1,"timeout");end
endmodule
