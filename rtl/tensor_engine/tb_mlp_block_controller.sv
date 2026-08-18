`timescale 1ns/1ps

module tb_mlp_block_controller;
  reg clk=0,rst_n=0,block_start=0;
  reg frontend_done=0,up_load_done=0,up_tile_done=0,up_all_done=0;
  reg down_load_done=0,down_tile_done=0;
  wire block_start_ready,frontend_start,up_load_enable,up_start;
  wire down_load_enable,down_start,busy,done;
  wire [3:0] frontend_group;
  wire [3:0] up_load_tile,up_start_tile,down_load_tile,down_start_tile;
  wire up_load_bank,up_start_bank,down_load_bank,down_start_bank;
  integer frontend_count=0,up_load_count=0,up_start_count=0;
  integer down_load_count=0,down_start_count=0;
  integer frontend_delay=0,up_load_delay=0,up_run_delay=0;
  integer down_load_delay=0,down_run_delay=0;
  reg frontend_pending=0,up_load_pending=0,up_run_pending=0;
  reg down_load_pending=0,down_run_pending=0;
  reg up_load_armed=1,down_load_armed=1;
  reg saw_up_overlap=0,saw_down_overlap=0;

  mlp_block_controller #(
    .FRONTEND_GROUPS(2),.UP_OUTPUT_TILES(4),.DOWN_OUTPUT_TILES(3),
    .TILE_WIDTH(4)
  ) dut(
    .clk(clk),.rst_n(rst_n),.block_start(block_start),
    .block_start_ready(block_start_ready),.frontend_start(frontend_start),
    .frontend_group(frontend_group),.frontend_start_ready(1'b1),
    .frontend_done(frontend_done),.up_load_enable(up_load_enable),
    .up_load_tile(up_load_tile),.up_load_bank(up_load_bank),
    .up_load_done(up_load_done),.up_start(up_start),
    .up_start_tile(up_start_tile),.up_start_bank(up_start_bank),
    .up_start_ready(1'b1),.up_tile_done(up_tile_done),
    .up_all_activations_done(up_all_done),
    .down_load_enable(down_load_enable),.down_load_tile(down_load_tile),
    .down_load_bank(down_load_bank),.down_load_done(down_load_done),
    .down_start(down_start),.down_start_tile(down_start_tile),
    .down_start_bank(down_start_bank),.down_start_ready(1'b1),
    .down_tile_done(down_tile_done),.busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    frontend_done<=0;up_load_done<=0;up_tile_done<=0;up_all_done<=0;
    down_load_done<=0;down_tile_done<=0;
    if(frontend_start) begin
      if(frontend_group!==frontend_count) $fatal(1,"frontend group order");
      frontend_count=frontend_count+1;frontend_pending=1;frontend_delay=0;
    end
    if(frontend_pending) begin
      frontend_delay=frontend_delay+1;
      if(frontend_delay==3) begin frontend_done<=1;frontend_pending=0;end
    end
    if(!up_load_enable) up_load_armed=1;
    if(up_load_enable && up_load_armed && !up_load_pending) begin
      if(up_load_tile!==up_load_count || up_load_bank!==up_load_tile[0])
        $fatal(1,"up load order");
      up_load_armed=0;up_load_pending=1;
      up_load_count=up_load_count+1;up_load_delay=0;
    end
    if(up_load_pending) begin
      up_load_delay=up_load_delay+1;
      if(up_load_delay==2) begin up_load_done<=1;up_load_pending=0;end
    end
    if(up_start) begin
      if(up_start_tile!==up_start_count || up_start_bank!==up_start_tile[0])
        $fatal(1,"up start order");
      up_start_count=up_start_count+1;up_run_pending=1;up_run_delay=0;
    end
    if(up_run_pending) begin
      up_run_delay=up_run_delay+1;if(up_load_enable) saw_up_overlap=1;
      if(up_run_delay==6) begin
        up_tile_done<=1;up_run_pending=0;
        if(up_start_count==4) up_all_done<=1;
      end
    end
    if(!down_load_enable) down_load_armed=1;
    if(down_load_enable && down_load_armed && !down_load_pending) begin
      if(down_load_tile!==down_load_count || down_load_bank!==down_load_tile[0])
        $fatal(1,"down load order");
      down_load_armed=0;down_load_pending=1;
      down_load_count=down_load_count+1;
      down_load_delay=0;
    end
    if(down_load_pending) begin
      down_load_delay=down_load_delay+1;
      if(down_load_delay==2) begin down_load_done<=1;down_load_pending=0;end
    end
    if(down_start) begin
      if(down_start_tile!==down_start_count ||
         down_start_bank!==down_start_tile[0]) $fatal(1,"down start order");
      down_start_count=down_start_count+1;down_run_pending=1;
      down_run_delay=0;
    end
    if(down_run_pending) begin
      down_run_delay=down_run_delay+1;if(down_load_enable) saw_down_overlap=1;
      if(down_run_delay==5) begin down_tile_done<=1;down_run_pending=0;end
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;block_start=1;
    @(negedge clk);block_start=0;wait(done);repeat(3) @(posedge clk);
    if(frontend_count!=2 || up_load_count!=4 || up_start_count!=4 ||
       down_load_count!=3 || down_start_count!=3)
      $fatal(1,"MLP controller count mismatch");
    if(!saw_up_overlap || !saw_down_overlap)
      $fatal(1,"MLP controller did not preload during execution");
    if(busy) $fatal(1,"MLP controller remained busy");
    $display("tb_mlp_block_controller: PASS");$finish;
  end
  initial begin repeat(500) @(posedge clk);$fatal(1,"timeout");end
endmodule
