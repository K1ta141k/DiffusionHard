`timescale 1ns/1ps

module tb_hidden_canvas_automatic_mlp_block;
  localparam integer CLIENT_TAG_WIDTH=12;
  reg clk=0,rst_n=0,block_start=0,canvas_data_valid=0;
  reg [575:0] canvas_data=0;
  wire block_start_ready,busy,done,canvas_read_valid;
  wire [3:0] canvas_group;
  wire [6:0] canvas_tile;
  wire [9:0] requested_up_tile,requested_down_tile,output_tile;
  wire requested_up_bank,requested_down_bank;
  wire up_weight_ready,up_metadata_ready;
  wire down_weight_ready,down_metadata_ready;
  wire output_valid;
  wire [0:0] output_group;
  wire [575:0] outputs;
  wire array_request_valid,array_request_clear,array_request_last;
  wire [CLIENT_TAG_WIDTH:0] array_request_tag,array_response_tag;
  wire [1023:0] array_activations;
  wire [1535:0] array_weights;
  wire array_response_valid;
  wire [767:0] array_accumulators;
  integer canvas_reads=0,up_weights=0,up_metadata=0;
  integer down_weights=0,down_metadata=0,up_requests=0;
  integer down_requests=0,outputs_seen=0,cycles=0;

  hidden_canvas_automatic_mlp_block #(
    .TOKENS(4),.DOWN_INPUT_SIZE(768),.DOWN_OUTPUT_SIZE(12),
    .CLIENT_TAG_WIDTH(CLIENT_TAG_WIDTH)
  ) dut(
    .clk(clk),.rst_n(rst_n),.block_start(block_start),
    .block_start_ready(block_start_ready),
    .smoothing_reciprocal_q15(18'd32768),.busy(busy),.done(done),
    .canvas_read_valid(canvas_read_valid),.canvas_read_group(canvas_group),
    .canvas_read_output_tile(canvas_tile),
    .canvas_read_data_valid(canvas_data_valid),
    .canvas_read_q10_packed(canvas_data),
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
    .array_request_valid(array_request_valid),
    .array_request_clear(array_request_clear),
    .array_request_last(array_request_last),.array_request_tag(array_request_tag),
    .array_request_activations(array_activations),
    .array_request_weights(array_weights),
    .array_response_valid(array_response_valid),
    .array_response_tag(array_response_tag),
    .array_response_accumulators(array_accumulators));

  int8_mac_tile_pipelined #(
    .M_LANES(4),.N_LANES(6),.DATA_WIDTH(8),.ACC_WIDTH(32),
    .TAG_WIDTH(CLIENT_TAG_WIDTH+1)
  ) shared_mac(
    .clk(clk),.rst_n(rst_n),.valid_in(array_request_valid),
    .clear_accumulators(array_request_clear),
    .last_k_tile(array_request_last),.tag_in(array_request_tag),
    .activations_packed(array_activations),.weights_packed(array_weights),
    .valid_out(array_response_valid),.tag_out(array_response_tag),
    .accumulators_packed(array_accumulators));

  always #2 clk=~clk;
  always @(posedge clk) begin
    cycles=cycles+1;
    canvas_data_valid<=canvas_read_valid;
    if(canvas_read_valid) begin canvas_reads=canvas_reads+1;canvas_data<=0;end
    if(up_weight_ready) begin
      if(requested_up_tile!==up_weights/24 ||
         requested_up_bank!==requested_up_tile[0])
        $fatal(1,"automatic up weight request mismatch");
      up_weights=up_weights+1;
    end
    if(up_metadata_ready) begin
      if(requested_up_tile!==up_metadata ||
         requested_up_bank!==requested_up_tile[0])
        $fatal(1,"automatic up metadata request mismatch");
      up_metadata=up_metadata+1;
    end
    if(down_weight_ready) begin
      if(requested_down_tile!==down_weights/24 ||
         requested_down_bank!==requested_down_tile[0])
        $fatal(1,"automatic down weight request mismatch");
      down_weights=down_weights+1;
    end
    if(down_metadata_ready) begin
      if(requested_down_tile!==down_metadata ||
         requested_down_bank!==requested_down_tile[0])
        $fatal(1,"automatic down metadata request mismatch");
      down_metadata=down_metadata+1;
    end
    if(array_request_valid) begin
      if(array_request_tag[CLIENT_TAG_WIDTH]) down_requests=down_requests+1;
      else up_requests=up_requests+1;
    end
    #1;
    if(output_valid) begin
      if(output_tile!==outputs_seen || output_group!==0 || outputs!==0)
        $fatal(1,"automatic MLP output mismatch");
      outputs_seen=outputs_seen+1;
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;block_start=1;
    wait(block_start_ready);@(posedge clk);@(negedge clk);block_start=0;
    wait(done);repeat(4) @(posedge clk);
    if(canvas_reads!=386 || up_weights!=3072 || up_metadata!=128 ||
       down_weights!=48 || down_metadata!=2 || up_requests!=3072 ||
       down_requests!=48 || outputs_seen!=2 || busy)
      $fatal(1,"automatic MLP final count mismatch %0d %0d %0d %0d %0d %0d %0d %0d",
        canvas_reads,up_weights,up_metadata,down_weights,down_metadata,
        up_requests,down_requests,outputs_seen);
    $display("tb_hidden_canvas_automatic_mlp_block: PASS cycles=%0d",cycles);
    $finish;
  end
  initial begin repeat(40000) @(posedge clk);$fatal(1,"timeout");end
endmodule
