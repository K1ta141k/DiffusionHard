`timescale 1ns/1ps

module tb_hidden_canvas_group_replay;
  reg clk=0,rst_n=0,start=0,canvas_data_valid=0;
  reg [575:0] canvas_data=0;
  reg output_ready=1;
  wire start_ready,canvas_read_valid,output_valid,busy,done;
  wire [3:0] canvas_group,output_group;
  wire [6:0] canvas_tile;
  wire [9:0] output_channel;
  wire [95:0] output_data;
  integer token,lane,channel_count=0,read_count=0,cycle_count=0;

  hidden_canvas_group_replay dut(
    .clk(clk),.rst_n(rst_n),.start(start),.group_in(7),
    .start_ready(start_ready),.canvas_read_valid(canvas_read_valid),
    .canvas_read_group(canvas_group),
    .canvas_read_output_tile(canvas_tile),
    .canvas_read_data_valid(canvas_data_valid),
    .canvas_read_q10_packed(canvas_data),.output_valid(output_valid),
    .output_ready(output_ready),.output_group(output_group),
    .output_channel(output_channel),.output_q10_packed(output_data),
    .busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    canvas_data_valid<=canvas_read_valid;
    if(canvas_read_valid) begin
      canvas_data<=0;read_count=read_count+1;
      for(token=0;token<4;token=token+1)
        for(lane=0;lane<6;lane=lane+1)
          canvas_data[(token*6+lane)*24 +: 24]<=
            canvas_group*10000+token*1000+canvas_tile*6+lane;
    end
    if(busy) begin
      cycle_count=cycle_count+1;
      output_ready<=(cycle_count%9)!=0;
    end else output_ready<=1;
    #1;
    if(output_valid && output_ready) begin
      if(output_group!==7 || output_channel!==channel_count)
        $fatal(1,"hidden replay tag mismatch");
      for(token=0;token<4;token=token+1)
        if($signed(output_data[token*24 +: 24])!==
           70000+token*1000+channel_count)
          $fatal(1,"hidden replay mismatch token %0d channel %0d",
                 token,channel_count);
      channel_count=channel_count+1;
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;start=1;
    @(negedge clk);start=0;wait(done);repeat(3) @(posedge clk);
    if(channel_count!=768) $fatal(1,"hidden replay channel count %0d",channel_count);
    if(read_count!=128) $fatal(1,"hidden replay read count %0d",read_count);
    if(busy) $fatal(1,"hidden replay remained busy");
    $display("tb_hidden_canvas_group_replay: PASS cycles=%0d",cycle_count);
    $finish;
  end
  initial begin repeat(2000) @(posedge clk);$fatal(1,"timeout");end
endmodule
