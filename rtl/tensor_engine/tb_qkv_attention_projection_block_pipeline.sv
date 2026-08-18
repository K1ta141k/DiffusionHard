`timescale 1ns/1ps

module tb_qkv_attention_projection_block_pipeline;
  reg clk=0,rst_n=0,block_start=0,constant_load=0,residual_load=0;
  reg [5:0] constant_token=0;
  reg [4:0] constant_pair=0,qkv_weight_input_tile=0;
  reg [4:0] projection_weight_input_tile=0;
  reg [3:0] residual_group=0;
  reg [6:0] residual_output_tile=0;
  reg normalized_data_valid=0,array_response_valid=0;
  reg [7:0] array_response_tag_reg=0;
`ifdef PACKED_M8
  reg array_response_narrow_reg=0;
`endif
  wire block_start_ready,qkv_metadata_ready,qkv_weight_ready;
  wire projection_metadata_ready,projection_weight_ready;
  wire [3:0] requested_qkv_head,requested_qkv_channel_tile;
  wire [1:0] requested_qkv_kind;
  wire [11:0] requested_qkv_global_row;
  wire normalized_read_valid;
  wire [3:0] normalized_group;
  wire [4:0] normalized_input_tile;
  wire [6:0] requested_projection_tile;
  wire block_tile_valid;
  wire [3:0] block_group;
  wire [6:0] block_output_tile;
  wire [575:0] block_data;
  wire array_request_valid,array_request_clear,array_request_last;
  wire [7:0] array_request_tag;
  wire [2303:0] array_activations;
  wire [3455:0] array_weights;
`ifdef PACKED_M8
  wire array_request_narrow;
  wire [2047:0] array_narrow_activations;
  wire [1535:0] array_narrow_weights;
`endif
  wire producer_busy,projection_busy,busy,done;
  integer index,lane,tile_count=0,cycles=0;

`ifdef PACKED_M8
  qkv_attention_projection_block_pipeline_packed_m8 #(
`else
  qkv_attention_projection_block_pipeline #(
`endif
    .HEADS(2),.OUTPUT_TILES(2)
  ) dut(
    .clk(clk),.rst_n(rst_n),.block_start(block_start),
    .block_start_ready(block_start_ready),.residual_load_valid(residual_load),
    .residual_load_group(residual_group),
    .residual_load_output_tile(residual_output_tile),
    .residual_load_q10_packed(576'b0),.qkv_metadata_valid(busy),
    .qkv_metadata_ready(qkv_metadata_ready),
    .qkv_metadata_head(requested_qkv_head),
    .qkv_metadata_kind(requested_qkv_kind),
    .qkv_metadata_channel_tile(requested_qkv_channel_tile),
    .qkv_metadata_multipliers_packed(144'b0),
    .qkv_metadata_biases_q12_packed(108'b0),.qkv_weight_tile_valid(busy),
    .qkv_weight_tile_ready(qkv_weight_ready),
    .qkv_weight_head(requested_qkv_head),
    .qkv_weight_kind(requested_qkv_kind),
    .qkv_weight_channel_tile(requested_qkv_channel_tile),
    .qkv_weight_input_tile(qkv_weight_input_tile),
    .qkv_weight_int16_packed(3072'b0),
    .requested_qkv_head(requested_qkv_head),
    .requested_qkv_kind(requested_qkv_kind),
    .requested_qkv_channel_tile(requested_qkv_channel_tile),
    .requested_qkv_global_row(requested_qkv_global_row),
    .normalized_read_valid(normalized_read_valid),
    .normalized_read_group(normalized_group),
    .normalized_read_input_tile(normalized_input_tile),
    .normalized_read_data_valid(normalized_data_valid),
    .normalized_q12_packed(2304'b0),.constant_load_valid(constant_load),
    .constant_load_token(constant_token),.constant_load_pair(constant_pair),
    .constant_load_cosine_q15(16'sd32767),.constant_load_sine_q15(16'sd0),
    .projection_metadata_valid(busy),
    .projection_metadata_ready(projection_metadata_ready),
    .projection_metadata_output_tile(requested_projection_tile),
    .projection_metadata_multipliers_packed(144'b0),
    .projection_weight_tile_valid(busy),
    .projection_weight_tile_ready(projection_weight_ready),
    .projection_weight_output_tile(requested_projection_tile),
    .projection_weight_input_tile(projection_weight_input_tile),
    .projection_weight_int8_packed(1536'b0),
    .requested_projection_output_tile(requested_projection_tile),
    .block_tile_valid(block_tile_valid),.block_tile_ready(1'b1),
    .block_group(block_group),.block_output_tile(block_output_tile),
    .block_q10_packed(block_data),.array_request_valid(array_request_valid),
    .array_request_clear(array_request_clear),
    .array_request_last(array_request_last),.array_request_tag(array_request_tag),
    .array_request_activations(array_activations),
    .array_request_weights(array_weights),
`ifdef PACKED_M8
    .array_request_narrow_int8_mode(array_request_narrow),
    .array_request_narrow_activations(array_narrow_activations),
    .array_request_narrow_weights(array_narrow_weights),
    .array_response_narrow_int8_mode(array_response_narrow_reg),
    .array_response_narrow_accumulators(1536'b0),
`endif
    .array_response_valid(array_response_valid),
    .array_response_tag(array_response_tag_reg),
    .array_response_accumulators(1152'b0),.producer_busy(producer_busy),
    .projection_busy(projection_busy),.busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    normalized_data_valid<=normalized_read_valid;
    array_response_valid<=array_request_valid && array_request_last;
    array_response_tag_reg<=array_request_tag;
`ifdef PACKED_M8
    array_response_narrow_reg<=array_request_narrow;
`endif
    if(qkv_weight_ready) begin
      if(qkv_weight_input_tile==23) qkv_weight_input_tile<=0;
      else qkv_weight_input_tile<=qkv_weight_input_tile+1'b1;
    end
    if(projection_weight_ready) begin
      if(projection_weight_input_tile==23) projection_weight_input_tile<=0;
      else projection_weight_input_tile<=projection_weight_input_tile+1'b1;
    end
    if(busy) cycles=cycles+1;
    #1;
    if(block_tile_valid) begin
      if(block_group!==(tile_count%16) || block_output_tile!==(tile_count/16))
        $fatal(1,"block output tag mismatch");
      for(lane=0;lane<24;lane=lane+1)
        if($signed(block_data[lane*24 +: 24])!==0)
          $fatal(1,"zero block emitted nonzero residual output");
      tile_count=tile_count+1;
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(index=0;index<32;index=index+1) begin
      @(negedge clk);residual_load=1;residual_output_tile=index/16;
      residual_group=index%16;
    end
    @(negedge clk);residual_load=0;
    for(index=0;index<2048;index=index+1) begin
      @(negedge clk);constant_load=1;
      constant_token=index/32;constant_pair=index%32;
    end
    @(negedge clk);constant_load=0;block_start=1;
    @(negedge clk);block_start=0;
    wait(done);repeat(3) @(posedge clk);
    if(tile_count!=32) $fatal(1,"missing block output tiles %0d",tile_count);
    if(busy || producer_busy || projection_busy)
      $fatal(1,"connected attention block remained busy");
    $display("tb_qkv_attention_projection_block_pipeline: PASS cycles=%0d",cycles);
    $finish;
  end
  initial begin repeat(120000) @(posedge clk);$fatal(1,"timeout");end
endmodule
