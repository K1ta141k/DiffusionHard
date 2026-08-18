`timescale 1ns/1ps
module tb_attention_projection_grouped_scheduler;
  reg clk=0,rst_n=0,start=0,weight_valid=0,group_data_valid=0;
  reg [4:0] weight_tile=0;reg [1535:0] weight_data=0;
  wire start_ready,weight_ready,canvas_read_valid,group_read_valid;
  wire [3:0] group_read_head,group_read_group;wire [5:0] canvas_token;
  wire projection_valid,busy,done,array_valid;wire [3:0] projection_group;
  wire [6:0] projection_tile;wire [2:0] projection_channels;
  wire [575:0] projection_data;wire [7:0] array_tag;
  wire [2303:0] array_activations;wire [3455:0] array_weights;
  integer cycles=0,reads=0,requests=0,outputs=0,tile;
  attention_projection_output_tile_scheduler #(.GROUPED_CANVAS(1)) dut(
    .clk(clk),.rst_n(rst_n),.start(start),.start_ready(start_ready),
    .output_tile_in(7'd9),.multipliers_packed({144{1'b0}}),
    .weight_tile_valid(weight_valid),.weight_tile_ready(weight_ready),
    .weight_input_tile(weight_tile),.weight_int8_packed(weight_data),
    .canvas_read_valid(canvas_read_valid),.canvas_read_head(),
    .canvas_read_token(canvas_token),.canvas_read_data_valid(1'b0),
    .canvas_read_data_packed(0),.canvas_group_read_valid(group_read_valid),
    .canvas_group_read_head(group_read_head),
    .canvas_group_read_group(group_read_group),
    .canvas_group_read_data_valid(group_data_valid),
    .canvas_group_read_data_packed(0),.projection_tile_valid(projection_valid),
    .projection_tile_ready(1'b1),.projection_group(projection_group),
    .projection_output_tile(projection_tile),
    .projection_valid_channels(projection_channels),
    .projection_q10_packed(projection_data),.array_request_valid(array_valid),
    .array_request_clear(),.array_request_last(),.array_request_tag(array_tag),
    .array_request_activations(array_activations),
    .array_request_weights(array_weights),.array_response_valid(1'b0),
    .array_response_tag(0),.array_response_accumulators(0),
    .busy(busy),.done(done));
  always #2 clk=~clk;
  always @(posedge clk)begin
    group_data_valid<=group_read_valid;
    if(busy)cycles=cycles+1;
    if(group_read_valid)reads=reads+1;
    if(array_valid)requests=requests+1;
    if(projection_valid)begin
      if(projection_group!==outputs||projection_tile!==9||
        projection_channels!==6||projection_data!==0)
        $fatal(1,"grouped projection output mismatch index=%0d",outputs);
      outputs=outputs+1;
    end
  end
  initial begin
    repeat(3)@(posedge clk);@(negedge clk);rst_n=1;start=1;
    wait(start_ready);@(posedge clk);@(negedge clk);start=0;
    for(tile=0;tile<24;tile=tile+1)begin
      @(negedge clk);weight_tile=tile;wait(weight_ready);weight_valid=1;
      @(negedge clk);weight_valid=0;
    end
    wait(done);@(posedge clk);#1;
    if(busy||reads!=384||requests!=384||outputs!=16||cycles>=1500)
      $fatal(1,"grouped projection counts mismatch cycles=%0d reads=%0d requests=%0d outputs=%0d",
        cycles,reads,requests,outputs);
    $display("tb_attention_projection_grouped_scheduler: PASS cycles=%0d reads=%0d requests=%0d outputs=%0d",
      cycles,reads,requests,outputs);$finish;
  end
  initial begin repeat(5000)@(posedge clk);
    $display("timeout state=%0d loaded=%0d sent=%0d response=%0d cycles=%0d reads=%0d requests=%0d outputs=%0d mac_valid=%0d requant=%0d",
      dut.state,dut.loaded_tiles,dut.requests_sent,dut.response_input_tile,
      cycles,reads,requests,outputs,dut.mac_output_valid,dut.requant_valid);
    $fatal(1,"timeout");end
endmodule
