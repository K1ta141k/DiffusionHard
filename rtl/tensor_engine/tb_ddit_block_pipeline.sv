`timescale 1ns/1ps

module tb_ddit_block_pipeline;
  reg clk=0,rst_n=0,block_start=0,residual_load=0,constant_load=0;
  reg [3:0] residual_group=0;
  reg [6:0] residual_tile=0;
  reg [5:0] constant_token=0;
  reg [4:0] constant_pair=0,qkv_weight_k=0,projection_weight_k=0;
  reg normalized_data_valid=0;
  wire block_start_ready,busy,done,qkv_metadata_ready,qkv_weight_ready;
  wire [3:0] requested_qkv_head,requested_qkv_channel_tile;
  wire [1:0] requested_qkv_kind;
  wire [11:0] requested_qkv_row;
  wire normalized_read_valid;
  wire [3:0] normalized_group;
  wire [4:0] normalized_tile;
  wire projection_metadata_ready,projection_weight_ready;
  wire [6:0] requested_projection_tile;
  wire attention_tile_valid;
  wire [3:0] attention_tile_group;
  wire [6:0] attention_tile_output_tile;
  wire [575:0] attention_tile_data;
  wire [9:0] reciprocal_channel,requested_up_tile,requested_down_tile;
  wire requested_up_bank,requested_down_bank;
  wire up_weight_ready,up_metadata_ready,down_weight_ready,down_metadata_ready;
  wire output_valid;
  wire [9:0] output_tile;
  wire [0:0] output_group;
  wire [575:0] outputs;
  wire attention_busy,mlp_busy;
  integer index,lane,attention_tiles=0,outputs_seen=0,cycles=0;
  integer up_weights=0,down_weights=0;
  integer attention_array_requests=0,attention_array_responses=0;
  reg saw_attention=0,saw_mlp=0;

  ddit_block_pipeline #(
    .HEADS(1),.ATTENTION_OUTPUT_TILES(2),.TOKENS(4),
    .DOWN_INPUT_SIZE(768),.DOWN_OUTPUT_SIZE(6)
`ifdef PACKED_ATTENTION
    ,.PACKED_ATTENTION(1)
`endif
`ifdef FOLDED_N2
    ,.PHYSICAL_N_LANES(2)
`endif
  ) dut(
    .clk(clk),.rst_n(rst_n),.block_start(block_start),
    .block_start_ready(block_start_ready),.busy(busy),.done(done),
    .residual_load_valid(residual_load),.residual_load_group(residual_group),
    .residual_load_output_tile(residual_tile),
    .residual_load_q10_packed(576'b0),
    .qkv_metadata_valid(attention_busy),.qkv_metadata_ready(qkv_metadata_ready),
    .qkv_metadata_head(requested_qkv_head),
    .qkv_metadata_kind(requested_qkv_kind),
    .qkv_metadata_channel_tile(requested_qkv_channel_tile),
    .qkv_metadata_multipliers_packed(144'b0),
    .qkv_metadata_biases_q12_packed(108'b0),
    .qkv_weight_tile_valid(attention_busy),
    .qkv_weight_tile_ready(qkv_weight_ready),
    .qkv_weight_head(requested_qkv_head),.qkv_weight_kind(requested_qkv_kind),
    .qkv_weight_channel_tile(requested_qkv_channel_tile),
    .qkv_weight_input_tile(qkv_weight_k),.qkv_weight_int16_packed(3072'b0),
    .requested_qkv_head(requested_qkv_head),
    .requested_qkv_kind(requested_qkv_kind),
    .requested_qkv_channel_tile(requested_qkv_channel_tile),
    .requested_qkv_global_row(requested_qkv_row),
    .normalized_read_valid(normalized_read_valid),
    .normalized_read_group(normalized_group),
    .normalized_read_input_tile(normalized_tile),
    .normalized_read_data_valid(normalized_data_valid),
    .normalized_q12_packed(2304'b0),.constant_load_valid(constant_load),
    .constant_load_token(constant_token),.constant_load_pair(constant_pair),
    .constant_load_cosine_q15(16'sd32767),
    .constant_load_sine_q15(16'sd0),
    .projection_metadata_valid(attention_busy),
    .projection_metadata_ready(projection_metadata_ready),
    .projection_metadata_output_tile(requested_projection_tile),
    .projection_metadata_multipliers_packed(144'b0),
    .projection_weight_tile_valid(attention_busy),
    .projection_weight_tile_ready(projection_weight_ready),
    .projection_weight_output_tile(requested_projection_tile),
    .projection_weight_input_tile(projection_weight_k),
    .projection_weight_int8_packed(1536'b0),
    .requested_projection_output_tile(requested_projection_tile),
    .attention_tile_valid(attention_tile_valid),
    .attention_tile_group(attention_tile_group),
    .attention_tile_output_tile(attention_tile_output_tile),
    .attention_tile_q10_packed(attention_tile_data),
    .smoothing_reciprocal_q15(18'd32768),
    .smoothing_reciprocal_channel(reciprocal_channel),
    .requested_up_output_tile(requested_up_tile),
    .requested_up_bank(requested_up_bank),
    .up_weight_stream_valid(1'b1),.up_weight_stream_ready(up_weight_ready),
    .up_weight_stream_data(1536'b0),.up_metadata_stream_valid(1'b1),
    .up_metadata_stream_ready(up_metadata_ready),
    .up_metadata_stream_data(444'b0),
    .requested_down_output_tile(requested_down_tile),
    .requested_down_bank(requested_down_bank),
    .down_weight_stream_valid(1'b1),
    .down_weight_stream_ready(down_weight_ready),
    .down_weight_stream_data(1536'b0),
    .down_metadata_stream_valid(1'b1),
    .down_metadata_stream_ready(down_metadata_ready),
    .down_metadata_stream_data(1344'b0),
    .output_valid(output_valid),.output_tile(output_tile),
    .output_group(output_group),.outputs_packed(outputs),
    .attention_busy(attention_busy),.mlp_busy(mlp_busy));

  genvar canvas_bank;
  generate
    for(canvas_bank=0;canvas_bank<64;canvas_bank=canvas_bank+1) begin: pad_heads
      integer canvas_address;
      initial begin
        for(canvas_address=16;canvas_address<192;
            canvas_address=canvas_address+1)
`ifdef PACKED_ATTENTION
          dut.packed_attention_path.attention.producer.canvas
            .canvas_banks[canvas_bank].memory[canvas_address]=0;
`else
          dut.fixed_attention_path.attention.producer.canvas
            .canvas_banks[canvas_bank]
            .memory[canvas_address]=0;
`endif
      end
    end
  endgenerate

  always #2 clk=~clk;
  always @(posedge clk) begin
    cycles=cycles+1;
    normalized_data_valid<=normalized_read_valid;
    if(qkv_weight_ready)
      qkv_weight_k<=(qkv_weight_k==23)?0:qkv_weight_k+1'b1;
    if(projection_weight_ready)
      projection_weight_k<=(projection_weight_k==23)?0:projection_weight_k+1'b1;
    if(up_weight_ready) up_weights=up_weights+1;
    if(down_weight_ready) down_weights=down_weights+1;
    if(attention_busy) saw_attention=1;
    if(mlp_busy) saw_mlp=1;
    if(attention_busy && mlp_busy) $fatal(1,"DDiT phases overlapped");
    if(dut.attention_array_valid) begin
      attention_array_requests=attention_array_requests+1;
      if((^dut.attention_array_activations)===1'bx ||
         (^dut.attention_array_weights)===1'bx)
        $fatal(1,"unknown attention array request %0d tag=%h",
          attention_array_requests,dut.attention_array_tag);
    end
    #1;
    if(dut.attention_array_response_valid) begin
      attention_array_responses=attention_array_responses+1;
      if((^dut.attention_array_response_accumulators)===1'bx)
        $fatal(1,"unknown attention array response %0d tag=%h",
          attention_array_responses,dut.attention_array_response_tag);
    end
    if(attention_tile_valid) begin
      if(attention_tile_group!==(attention_tiles%16) ||
         attention_tile_output_tile!==(attention_tiles/16) ||
         attention_tile_data!==0) begin
        $display("attention mismatch count=%0d group=%0d tile=%0d data=%h",
          attention_tiles,attention_tile_group,attention_tile_output_tile,
          attention_tile_data);
        $fatal(1,"DDiT attention tile mismatch");
      end
      attention_tiles=attention_tiles+1;
    end
    if(output_valid) begin
      if(output_tile!==0 || output_group!==0 || outputs!==0)
        $fatal(1,"DDiT final output mismatch");
      outputs_seen=outputs_seen+1;
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(index=0;index<128;index=index+1) begin
      @(negedge clk);residual_load=1;residual_group=0;residual_tile=index;
    end
    for(index=0;index<30;index=index+1) begin
      @(negedge clk);residual_load=1;residual_group=1+index%15;
      residual_tile=index/15;
    end
    @(negedge clk);residual_load=0;
    for(index=0;index<2048;index=index+1) begin
      @(negedge clk);constant_load=1;
      constant_token=index/32;constant_pair=index%32;
    end
    @(negedge clk);constant_load=0;block_start=1;
    wait(block_start_ready);@(posedge clk);@(negedge clk);block_start=0;
    wait(done);repeat(4) @(posedge clk);
    if(!saw_attention || !saw_mlp || attention_tiles!=32 || outputs_seen!=1 ||
       up_weights!=3072 || down_weights!=24 || busy)
      $fatal(1,"DDiT connected count mismatch %0d %0d %0d %0d",
        attention_tiles,outputs_seen,up_weights,down_weights);
    $display("tb_ddit_block_pipeline: PASS cycles=%0d attention_requests=%0d attention_responses=%0d mlp_requests=%0d",
      cycles,attention_array_requests,attention_array_responses,
      up_weights+down_weights);
    $finish;
  end
  initial begin repeat(150000) @(posedge clk);$fatal(1,"timeout");end
endmodule
