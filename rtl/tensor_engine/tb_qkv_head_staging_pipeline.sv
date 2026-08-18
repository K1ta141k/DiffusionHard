`timescale 1ns/1ps

module tb_qkv_head_staging_pipeline;
  reg clk=0,rst_n=0,start=0,constant_load=0;
  reg [5:0] constant_token=0;
  reg [4:0] constant_pair=0;
  reg normalized_data_valid=0,array_response_valid=0;
  reg [7:0] array_response_tag=0;
  reg [4:0] weight_input_tile=0;
  wire start_ready,metadata_ready,weight_ready,normalized_read_valid;
  wire [3:0] normalized_group,requested_head,requested_channel_tile;
  wire [4:0] normalized_input_tile;
  wire [1:0] requested_kind;
  wire [2:0] requested_valid_channels;
  wire [11:0] requested_global_row;
  wire query_write,key_write,value_write;
  wire [5:0] qk_token,qk_channel,value_token,value_channel;
  wire signed [17:0] query_data,key_data,value_data;
  wire array_request_valid,array_request_clear,array_request_last;
  wire [7:0] array_request_tag;
  wire [2303:0] array_activations;
  wire [3455:0] array_weights;
  wire projection_busy,rotary_busy,busy,done;
  integer index,q_writes=0,k_writes=0,v_writes=0,cycles=0;
  reg overlap_seen=0;

  qkv_head_staging_pipeline #(
    .CHANNEL_TILES(1),.LAST_TILE_VALID_CHANNELS(4),.INTERNAL_MAC(0)
  ) dut(
    .clk(clk),.rst_n(rst_n),.start(start),.start_ready(start_ready),.head_in(3),
    .metadata_valid(busy),.metadata_ready(metadata_ready),
    .metadata_head(requested_head),.metadata_kind(requested_kind),
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
    .query_write_valid(query_write),.key_write_valid(key_write),
    .value_write_valid(value_write),.query_key_write_token(qk_token),
    .query_key_write_channel(qk_channel),.query_write_q12(query_data),
    .key_write_q12(key_data),.value_write_token(value_token),
    .value_write_channel(value_channel),.value_write_q12(value_data),
    .array_request_valid(array_request_valid),
    .array_request_clear(array_request_clear),
    .array_request_last(array_request_last),.array_request_tag(array_request_tag),
    .array_request_activations(array_activations),
    .array_request_weights(array_weights),
    .array_response_valid(array_response_valid),
    .array_response_tag(array_response_tag),
    .array_response_accumulators(1152'b0),.projection_busy(projection_busy),
    .rotary_busy(rotary_busy),.busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    normalized_data_valid<=normalized_read_valid;
    array_response_valid<=array_request_valid && array_request_last;
    array_response_tag<=array_request_tag;
    if(weight_ready) begin
      if(weight_input_tile==23) weight_input_tile<=0;
      else weight_input_tile<=weight_input_tile+1'b1;
    end
    if(busy) cycles=cycles+1;
    if(rotary_busy && projection_busy && requested_kind==2) overlap_seen=1;
    #1;
    if(query_write) q_writes=q_writes+1;
    if(key_write) k_writes=k_writes+1;
    if(value_write) begin
      if(value_channel>3) $fatal(1,"tail channel escaped router");
      v_writes=v_writes+1;
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(index=0;index<2048;index=index+1) begin
      @(negedge clk);constant_load=1;
      constant_token=index/32;constant_pair=index%32;
    end
    @(negedge clk);constant_load=0;start=1;
    @(negedge clk);start=0;
    wait(done);repeat(3) @(posedge clk);
    if(q_writes!=4096 || k_writes!=4096)
      $fatal(1,"rotary write count mismatch Q=%0d K=%0d",q_writes,k_writes);
    if(v_writes!=256) $fatal(1,"value write count mismatch %0d",v_writes);
    if(!overlap_seen) $fatal(1,"rotary did not overlap V projection");
    if(busy || projection_busy || rotary_busy)
      $fatal(1,"staging pipeline remained busy");
    $display("tb_qkv_head_staging_pipeline: PASS cycles=%0d",cycles);
    $finish;
  end
  initial begin repeat(20000) @(posedge clk);$fatal(1,"timeout");end
endmodule
