`timescale 1ns/1ps

module tb_qkv_attention_multihead_canvas_pipeline;
  reg clk=0,rst_n=0,block_start=0,constant_load=0,canvas_read=0;
  reg [5:0] constant_token=0,canvas_read_token=0;
  reg [4:0] constant_pair=0;
  reg [3:0] canvas_read_head=0;
  reg normalized_data_valid=0,array_response_valid=0;
  reg [7:0] array_response_tag_reg=0;
  reg [4:0] weight_input_tile=0;
  wire block_start_ready,metadata_ready,weight_ready,normalized_read_valid;
  wire [3:0] normalized_group,requested_head,requested_channel_tile;
  wire [4:0] normalized_input_tile;
  wire [1:0] requested_kind;
  wire [2:0] requested_valid_channels;
  wire [11:0] requested_global_row;
  wire canvas_data_valid;
  wire [1151:0] canvas_data;
  wire array_request_valid,array_request_clear,array_request_last;
  wire [7:0] array_request_tag;
  wire [2303:0] array_activations;
  wire [3455:0] array_weights;
  wire [3:0] active_head;
  wire busy,done;
  integer index,channel,cycles=0;
  reg saw_head_one=0;

  qkv_attention_multihead_canvas_pipeline #(.HEADS(2)) dut(
    .clk(clk),.rst_n(rst_n),.block_start(block_start),
    .block_start_ready(block_start_ready),.metadata_valid(busy),
    .metadata_ready(metadata_ready),.metadata_head(requested_head),
    .metadata_kind(requested_kind),
    .metadata_channel_tile(requested_channel_tile),
    .metadata_multipliers_packed(144'b0),.metadata_biases_q12_packed(108'b0),
    .weight_tile_valid(busy),.weight_tile_ready(weight_ready),
    .weight_head(requested_head),.weight_kind(requested_kind),
    .weight_channel_tile(requested_channel_tile),
    .weight_input_tile(weight_input_tile),.weight_int16_packed(3072'b0),
    .requested_head(requested_head),.requested_kind(requested_kind),
    .requested_channel_tile(requested_channel_tile),
    .requested_valid_channels(requested_valid_channels),
    .requested_global_row(requested_global_row),
    .normalized_read_valid(normalized_read_valid),
    .normalized_read_group(normalized_group),
    .normalized_read_input_tile(normalized_input_tile),
    .normalized_read_data_valid(normalized_data_valid),
    .normalized_q12_packed(2304'b0),.constant_load_valid(constant_load),
    .constant_load_token(constant_token),.constant_load_pair(constant_pair),
    .constant_load_cosine_q15(16'sd32767),.constant_load_sine_q15(16'sd0),
    .canvas_read_valid(canvas_read),.canvas_read_head(canvas_read_head),
    .canvas_read_token(canvas_read_token),.canvas_read_data_valid(canvas_data_valid),
    .canvas_read_data_packed(canvas_data),
    .array_request_valid(array_request_valid),
    .array_request_clear(array_request_clear),
    .array_request_last(array_request_last),.array_request_tag(array_request_tag),
    .array_request_activations(array_activations),
    .array_request_weights(array_weights),
    .array_response_valid(array_response_valid),
    .array_response_tag(array_response_tag_reg),
    .array_response_accumulators(1152'b0),.active_head(active_head),
    .busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    normalized_data_valid<=normalized_read_valid;
    array_response_valid<=array_request_valid && array_request_last;
    array_response_tag_reg<=array_request_tag;
    if(weight_ready) begin
      if(weight_input_tile==23) weight_input_tile<=0;
      else weight_input_tile<=weight_input_tile+1'b1;
    end
    if(busy) cycles=cycles+1;
    if(active_head==1 && busy) saw_head_one=1;
  end

  task check_canvas_zero;
    input [3:0] read_head;
    input [5:0] read_token;
    begin
      @(negedge clk);canvas_read=1;canvas_read_head=read_head;
      canvas_read_token=read_token;
      @(posedge clk);#1;
      if(!canvas_data_valid) $fatal(1,"canvas read valid missing");
      for(channel=0;channel<64;channel=channel+1)
        if($signed(canvas_data[channel*18 +: 18])!==0)
          $fatal(1,"canvas mismatch head %0d token %0d channel %0d",
                 read_head,read_token,channel);
      @(negedge clk);canvas_read=0;
    end
  endtask

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(index=0;index<2048;index=index+1) begin
      @(negedge clk);constant_load=1;
      constant_token=index/32;constant_pair=index%32;
    end
    @(negedge clk);constant_load=0;block_start=1;
    @(negedge clk);block_start=0;
    wait(done);repeat(3) @(posedge clk);
    if(!saw_head_one) $fatal(1,"second head never started");
    if(busy) $fatal(1,"multihead producer remained busy");
    check_canvas_zero(0,17);
    check_canvas_zero(1,63);
    $display("tb_qkv_attention_multihead_canvas_pipeline: PASS cycles=%0d",cycles);
    $finish;
  end
  initial begin repeat(120000) @(posedge clk);$fatal(1,"timeout");end
endmodule
