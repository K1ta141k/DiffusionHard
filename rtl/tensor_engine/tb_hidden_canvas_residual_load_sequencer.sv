`timescale 1ns/1ps

module tb_hidden_canvas_residual_load_sequencer;
  reg clk=0,rst_n=0,command_valid=0;
  reg [6:0] command_tile=0;
  reg canvas_data_valid=0,residual_ready=0;
  reg [575:0] canvas_data=0;
  wire command_ready,canvas_read_valid,residual_valid,busy,done;
  wire [1:0] canvas_group,residual_group;
  wire [6:0] canvas_tile,residual_tile;
  wire [575:0] residual_data;
  integer reads=0,loads=0,cycles=0;

  hidden_canvas_residual_load_sequencer #(.TOKEN_GROUPS(4)) dut(
    .clk(clk),.rst_n(rst_n),.command_valid(command_valid),
    .command_output_tile(command_tile),.command_ready(command_ready),
    .canvas_read_valid(canvas_read_valid),.canvas_read_group(canvas_group),
    .canvas_read_output_tile(canvas_tile),
    .canvas_read_data_valid(canvas_data_valid),.canvas_read_data(canvas_data),
    .residual_load_valid(residual_valid),.residual_load_group(residual_group),
    .residual_load_output_tile(residual_tile),
    .residual_load_data(residual_data),.residual_load_ready(residual_ready),
    .busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    cycles=cycles+1;
    canvas_data_valid<=canvas_read_valid;
    if(canvas_read_valid) begin
      if(canvas_group!==reads || canvas_tile!==7'd73)
        $fatal(1,"canvas request mismatch");
      canvas_data<={576{1'b0}} | (reads+11);reads=reads+1;
    end
    residual_ready<=cycles[0] || cycles[2];
    if(residual_valid && residual_ready) begin
      if(residual_group!==loads || residual_tile!==7'd73 ||
         residual_data!==loads+11)
        $fatal(1,"residual load mismatch");
      loads=loads+1;
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    command_tile=73;command_valid=1;wait(command_ready);
    @(posedge clk);@(negedge clk);command_valid=0;
    wait(done);
    if(command_ready) $fatal(1,"done guard did not hold command closed");
    repeat(3) @(posedge clk);
    if(reads!=4 || loads!=4 || busy)
      $fatal(1,"residual loader count mismatch");
    $display("tb_hidden_canvas_residual_load_sequencer: PASS cycles=%0d",cycles);
    $finish;
  end
  initial begin repeat(100) @(posedge clk);$fatal(1,"timeout");end
endmodule
