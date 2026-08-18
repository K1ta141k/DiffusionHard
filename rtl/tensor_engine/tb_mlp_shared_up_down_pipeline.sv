`timescale 1ns/1ps

module tb_mlp_shared_up_down_pipeline;
  localparam integer CLIENT_TAG_WIDTH=12;
  reg clk=0,rst_n=0;
  reg up_activation_valid=0,up_weight_valid=0,up_metadata_valid=0;
  reg up_token_factor_valid=0,up_start=0;
  reg [4:0] up_activation_k=0,up_weight_k=0;
  reg [9:0] up_output_tile=0;
  reg down_weight_valid=0,down_metadata_valid=0,down_residual_valid=0;
  reg down_start=0;
  reg [4:0] down_weight_k=0;
  reg [9:0] down_output_tile=0;
  wire up_weight_ready,up_metadata_ready,up_token_factor_ready;
  wire up_start_ready,up_busy,up_tile_done,up_all_done;
  wire down_weight_ready,down_metadata_ready,down_residual_ready;
  wire down_start_ready,down_busy,output_valid,down_done;
  wire output_bank;
  wire [9:0] output_tile;
  wire [0:0] output_group;
  wire [575:0] outputs;
  wire array_request_valid,array_request_clear,array_request_last;
  wire [CLIENT_TAG_WIDTH:0] array_request_tag,array_response_tag;
  wire [1023:0] array_activations;
  wire [1535:0] array_weights;
  wire array_response_valid;
  wire [767:0] array_accumulators;
  integer tile,k,up_requests=0,down_requests=0,outputs_seen=0,cycles=0;

  mlp_shared_up_down_pipeline #(
    .TOKENS(4),.UP_INPUT_SIZE(768),.DOWN_INPUT_SIZE(768),
    .DOWN_OUTPUT_SIZE(12),.M_LANES(4),.N_LANES(6),
    .CLIENT_TAG_WIDTH(CLIENT_TAG_WIDTH)
  ) dut(
    .clk(clk),.rst_n(rst_n),
    .up_activation_load_valid(up_activation_valid),
    .up_activation_load_group(0),.up_activation_load_k_tile(up_activation_k),
    .up_activation_load_data(1024'b0),.up_weight_load_valid(up_weight_valid),
    .up_weight_load_bank(0),.up_weight_load_k_tile(up_weight_k),
    .up_weight_load_data(1536'b0),.up_weight_load_ready(up_weight_ready),
    .up_metadata_load_valid(up_metadata_valid),.up_metadata_load_bank(0),
    .up_metadata_output_factors(108'b0),.up_metadata_biases(192'b0),
    .up_metadata_interstage_multipliers(144'b0),
    .up_metadata_load_ready(up_metadata_ready),
    .up_token_factor_load_valid(up_token_factor_valid),
    .up_token_factor_load_group(0),.up_token_factor_load_factors(64'b0),
    .up_token_factor_load_ready(up_token_factor_ready),.up_start(up_start),
    .up_start_bank(0),.up_start_output_tile(up_output_tile),
    .up_start_ready(up_start_ready),.up_busy(up_busy),
    .up_tile_done(up_tile_done),.up_all_activations_done(up_all_done),
    .down_weight_load_valid(down_weight_valid),.down_weight_load_bank(0),
    .down_weight_load_k_tile(down_weight_k),.down_weight_load_data(1536'b0),
    .down_weight_load_ready(down_weight_ready),
    .down_metadata_load_valid(down_metadata_valid),
    .down_metadata_load_bank(0),.down_metadata_multipliers(576'b0),
    .down_metadata_biases(768'b0),
    .down_metadata_load_ready(down_metadata_ready),
    .down_residual_load_valid(down_residual_valid),
    .down_residual_load_group(0),
    .down_residual_load_output_tile(down_output_tile),
    .down_residual_load_data(576'b0),
    .down_residual_load_ready(down_residual_ready),.down_start(down_start),
    .down_start_bank(0),.down_start_output_tile(down_output_tile),
    .down_start_ready(down_start_ready),.down_busy(down_busy),
    .output_valid(output_valid),.output_bank(output_bank),
    .output_tile(output_tile),.output_group(output_group),
    .outputs_packed(outputs),.down_done(down_done),
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
    if(array_request_valid) begin
      if(array_request_tag[CLIENT_TAG_WIDTH]) down_requests=down_requests+1;
      else up_requests=up_requests+1;
    end
    if(up_busy || down_busy) cycles=cycles+1;
    #1;
    if(output_valid) begin
      if(output_bank!==0 || output_tile!==outputs_seen ||
         output_group!==0 || outputs!==0)
        $fatal(1,"shared MLP zero output mismatch");
      outputs_seen=outputs_seen+1;
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(k=0;k<24;k=k+1) begin
      @(negedge clk);up_activation_valid=1;up_activation_k=k;
    end
    @(negedge clk);up_activation_valid=0;up_token_factor_valid=1;
    @(posedge clk);#1;
    if(!up_token_factor_ready) $fatal(1,"token factor load blocked");
    @(negedge clk);up_token_factor_valid=0;
    for(tile=0;tile<128;tile=tile+1) begin
      for(k=0;k<24;k=k+1) begin
        wait(up_weight_ready);@(negedge clk);up_weight_valid=1;up_weight_k=k;
      end
      @(negedge clk);up_weight_valid=0;up_metadata_valid=1;
      wait(up_metadata_ready);@(posedge clk);@(negedge clk);up_metadata_valid=0;
      up_output_tile=tile;up_start=1;wait(up_start_ready);
      @(posedge clk);@(negedge clk);up_start=0;wait(up_tile_done);
    end
    wait(up_all_done);
    for(tile=0;tile<2;tile=tile+1) begin
      down_output_tile=tile;
      for(k=0;k<24;k=k+1) begin
        wait(down_weight_ready);@(negedge clk);down_weight_valid=1;
        down_weight_k=k;
      end
      @(negedge clk);down_weight_valid=0;down_metadata_valid=1;
      wait(down_metadata_ready);@(posedge clk);@(negedge clk);
      down_metadata_valid=0;down_residual_valid=1;
      wait(down_residual_ready);@(posedge clk);@(negedge clk);
      down_residual_valid=0;down_start=1;wait(down_start_ready);
      @(posedge clk);@(negedge clk);down_start=0;wait(down_done);
    end
    repeat(4) @(posedge clk);
    if(up_requests!=3072) $fatal(1,"up request count %0d",up_requests);
    if(down_requests!=48) $fatal(1,"down request count %0d",down_requests);
    if(outputs_seen!=2) $fatal(1,"output count %0d",outputs_seen);
    $display("tb_mlp_shared_up_down_pipeline: PASS cycles=%0d",cycles);
    $finish;
  end
  initial begin repeat(30000) @(posedge clk);$fatal(1,"timeout");end
endmodule
