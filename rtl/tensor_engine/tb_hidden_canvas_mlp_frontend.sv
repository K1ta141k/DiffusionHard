`timescale 1ns/1ps

module tb_hidden_canvas_mlp_frontend;
  reg clk=0,rst_n=0,start=0,canvas_data_valid=0;
  reg [575:0] canvas_data=0;
  wire start_ready,canvas_read_valid,token_factor_valid,activation_valid;
  wire [3:0] canvas_group,token_group,activation_group;
  wire [6:0] canvas_tile;
  wire [63:0] token_factors;
  wire [4:0] activation_k_tile;
  wire [1023:0] activation_data;
  wire busy,done;
  integer reads=0,factors=0,activations=0,cycles=0;

  hidden_canvas_mlp_frontend dut(
    .clk(clk),.rst_n(rst_n),.start(start),.group_in(3),
    .smoothing_reciprocal_q15(18'd32768),.start_ready(start_ready),
    .canvas_read_valid(canvas_read_valid),.canvas_read_group(canvas_group),
    .canvas_read_output_tile(canvas_tile),
    .canvas_read_data_valid(canvas_data_valid),
    .canvas_read_q10_packed(canvas_data),
    .token_factor_valid(token_factor_valid),.token_factor_group(token_group),
    .token_factors_packed(token_factors),
    .activation_load_valid(activation_valid),
    .activation_load_group(activation_group),
    .activation_load_k_tile(activation_k_tile),
    .activation_load_data(activation_data),.busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    canvas_data_valid<=canvas_read_valid;
    if(canvas_read_valid) begin
      canvas_data<=0;reads=reads+1;
    end
    if(busy) cycles=cycles+1;
    #1;
    if(token_factor_valid) begin
      if(token_group!==3) $fatal(1,"frontend token group mismatch");
      factors=factors+1;
    end
    if(activation_valid) begin
      if(activation_group!==3 || activation_k_tile!==activations ||
         activation_data!==0)
        $fatal(1,"frontend activation mismatch tile %0d",activations);
      activations=activations+1;
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;start=1;
    @(negedge clk);start=0;wait(done);repeat(3) @(posedge clk);
    if(reads!=384) $fatal(1,"frontend canvas reads %0d",reads);
    if(factors!=1) $fatal(1,"frontend factor count %0d",factors);
    if(activations!=24) $fatal(1,"frontend activation count %0d",activations);
    if(busy) $fatal(1,"hidden frontend remained busy");
    $display("tb_hidden_canvas_mlp_frontend: PASS cycles=%0d",cycles);
    $finish;
  end
  initial begin repeat(10000) @(posedge clk);$fatal(1,"timeout");end
endmodule
