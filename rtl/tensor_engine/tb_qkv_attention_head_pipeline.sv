`timescale 1ns/1ps

module tb_qkv_attention_head_pipeline;
  reg clk=0,rst_n=0,start=0,constant_load=0;
  reg [5:0] constant_token=0;
  reg [4:0] constant_pair=0;
  reg normalized_data_valid=0;
  reg [4:0] weight_input_tile=0;
  wire start_ready,metadata_ready,weight_ready,normalized_read_valid;
  wire [3:0] normalized_group,requested_head,requested_channel_tile;
  wire [4:0] normalized_input_tile;
  wire [1:0] requested_kind;
  wire [2:0] requested_valid_channels;
  wire [11:0] requested_global_row;
  wire attention_valid;
  wire [3:0] attention_group,attention_output_tile;
  wire [2:0] attention_valid_channels;
  wire [431:0] attention_data;
  wire array_request_valid,array_request_clear,array_request_last;
  wire [7:0] array_request_tag;
  wire [2303:0] array_activations;
  wire [3455:0] array_weights;
  wire array_response_valid;
  wire [7:0] array_response_tag;
  wire [1151:0] array_response_accumulators;
  wire staging_busy,attention_busy,busy,done;
  integer index,tiles=0,lane,cycles=0;
  integer staging_cycles=0,head_cycles=0,qkv_projection_cycles=0;
  integer rotary_cycles=0,qk_cycles=0,softmax_cycles=0,pv_cycles=0;

  qkv_attention_head_pipeline #(.INTERNAL_MAC(0)) dut(
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
    .attention_tile_valid(attention_valid),.attention_tile_ready(1'b1),
    .attention_group(attention_group),
    .attention_output_tile(attention_output_tile),
    .attention_valid_channels(attention_valid_channels),
    .attention_q12_packed(attention_data),
    .array_request_valid(array_request_valid),
    .array_request_clear(array_request_clear),
    .array_request_last(array_request_last),.array_request_tag(array_request_tag),
    .array_request_activations(array_activations),
    .array_request_weights(array_weights),
    .array_response_valid(array_response_valid),
    .array_response_tag(array_response_tag),
    .array_response_accumulators(array_response_accumulators),
    .staging_busy(staging_busy),
    .attention_busy(attention_busy),.busy(busy),.done(done));

  mixed_precision_mac_tile_pipelined #(
    .M_LANES(4),.N_LANES(6),.STORAGE_WIDTH(18),.ACC_WIDTH(48),.TAG_WIDTH(8)
  ) shared_mac(
    .clk(clk),.rst_n(rst_n),.valid_in(array_request_valid),
    .narrow_int8_mode(1'b0),.clear_accumulators(array_request_clear),
    .last_k_tile(array_request_last),.tag_in(array_request_tag),
    .activations_packed(array_activations),.weights_packed(array_weights),
    .valid_out(array_response_valid),.tag_out(array_response_tag),
    .accumulators_packed(array_response_accumulators));

  always #2 clk=~clk;
  always @(posedge clk) begin
    normalized_data_valid<=normalized_read_valid;
    if(weight_ready) begin
      if(weight_input_tile==23) weight_input_tile<=0;
      else weight_input_tile<=weight_input_tile+1'b1;
    end
    if(busy) cycles=cycles+1;
    if(staging_busy) staging_cycles=staging_cycles+1;
    if(attention_busy) head_cycles=head_cycles+1;
    if(dut.staging_projection_busy)
      qkv_projection_cycles=qkv_projection_cycles+1;
    if(dut.staging_rotary_busy) rotary_cycles=rotary_cycles+1;
    if(dut.head.group_pipeline.qk_busy) qk_cycles=qk_cycles+1;
    if(dut.head.group_pipeline.softmax_busy)
      softmax_cycles=softmax_cycles+1;
    if(dut.head.group_pipeline.pv_busy) pv_cycles=pv_cycles+1;
    #1;
    if(attention_valid) begin
      if(attention_group!==(tiles/11) || attention_output_tile!==(tiles%11))
        $fatal(1,"attention output tag mismatch");
      for(lane=0;lane<24;lane=lane+1)
        if($signed(attention_data[lane*18 +: 18])!==0)
          $fatal(1,"zero pipeline emitted nonzero data");
      tiles=tiles+1;
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
    if(tiles!=176) $fatal(1,"missing attention output tiles %0d",tiles);
    if(busy || staging_busy || attention_busy)
      $fatal(1,"connected QKV attention head remained busy");
    $display("tb_qkv_attention_head_pipeline: PASS cycles=%0d",cycles);
    $display("tb_qkv_attention_head_pipeline: PROFILE staging=%0d head=%0d qkv_projection=%0d rotary=%0d qk=%0d softmax=%0d pv=%0d",
      staging_cycles,head_cycles,qkv_projection_cycles,rotary_cycles,
      qk_cycles,softmax_cycles,pv_cycles);
    $finish;
  end
  initial begin repeat(70000) @(posedge clk);$fatal(1,"timeout");end
endmodule
