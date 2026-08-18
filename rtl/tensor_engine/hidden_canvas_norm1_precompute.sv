`timescale 1ns/1ps

module hidden_canvas_norm1_precompute(
    input wire clk,input wire rst_n,input wire start,output wire start_ready,
    output wire canvas_read_valid,output wire [3:0] canvas_read_group,
    output wire [6:0] canvas_read_output_tile,
    input wire canvas_read_data_valid,
    input wire [4*6*24-1:0] canvas_read_q10_packed,
    input wire normalized_read_valid,input wire [3:0] normalized_read_group,
    input wire [4:0] normalized_read_input_tile,
    output reg normalized_read_data_valid,
    output wire [4*32*18-1:0] normalized_q12_packed,
    output wire busy,output reg done
);
    localparam [2:0] IDLE=0,STATS=1,WAIT_FINAL=2,FINAL=3,WAIT_NEXT=4;
    reg [2:0] state;reg [3:0] active_group;
    wire replay_start_ready,replay_output_valid,replay_output_ready;
    wire [3:0] replay_output_group;wire [9:0] replay_output_channel;
    wire [4*24-1:0] replay_output_q10;wire replay_busy,replay_done;
    wire norm_start_ready,norm_replay_ready,norm_input_ready;
    wire norm_output_valid;wire [3:0] norm_output_group;
    wire [9:0] norm_output_channel;wire [4*18-1:0] norm_output_q12;
    wire norm_busy,norm_done;
    wire launch_initial=state==IDLE && start && start_ready;
    wire launch_next=state==WAIT_NEXT && replay_start_ready && norm_start_ready;
    wire launch_group=launch_initial || launch_next;
    wire launch_final=state==WAIT_FINAL && replay_start_ready && norm_replay_ready;
    wire [8:0] write_group_base={norm_output_group,4'b0}
      +{norm_output_group,3'b0};
    wire [8:0] read_group_base={normalized_read_group,4'b0}
      +{normalized_read_group,3'b0};
    wire [8:0] write_address=write_group_base+norm_output_channel[9:5];
    wire [8:0] read_address=read_group_base+normalized_read_input_tile;
    genvar channel_lane,token_lane;

    assign start_ready=state==IDLE && replay_start_ready && norm_start_ready;
    assign busy=state!=IDLE || replay_busy || norm_busy;

    hidden_canvas_group_replay replay(.clk(clk),.rst_n(rst_n),
      .start(launch_group||launch_final),.group_in(active_group),
      .start_ready(replay_start_ready),.canvas_read_valid(canvas_read_valid),
      .canvas_read_group(canvas_read_group),
      .canvas_read_output_tile(canvas_read_output_tile),
      .canvas_read_data_valid(canvas_read_data_valid),
      .canvas_read_q10_packed(canvas_read_q10_packed),
      .output_valid(replay_output_valid),.output_ready(replay_output_ready),
      .output_group(replay_output_group),.output_channel(replay_output_channel),
      .output_q10_packed(replay_output_q10),.busy(replay_busy),
      .done(replay_done));

    layer_norm_q12_group norm(.clk(clk),.rst_n(rst_n),
      .start(launch_group),.group_in(active_group),.start_ready(norm_start_ready),
      .start_replay(launch_final),.final_replay(1'b1),
      .replay_ready(norm_replay_ready),.input_valid(replay_output_valid),
      .input_ready(norm_input_ready),.input_q10_packed(replay_output_q10),
      .output_valid(norm_output_valid),.output_group(norm_output_group),
      .output_channel(norm_output_channel),
      .output_q12_packed(norm_output_q12),.busy(norm_busy),.done(norm_done));
    assign replay_output_ready=norm_input_ready;

    generate for(channel_lane=0;channel_lane<32;
      channel_lane=channel_lane+1) begin: normalized_banks
      reg [4*18-1:0] memory[0:383];reg [4*18-1:0] read_value;
      always @(posedge clk) begin
        if(norm_output_valid && norm_output_channel[4:0]==channel_lane)
          memory[write_address]<=norm_output_q12;
        if(normalized_read_valid) read_value<=memory[read_address];
      end
      for(token_lane=0;token_lane<4;token_lane=token_lane+1) begin: pack_tokens
        assign normalized_q12_packed[(token_lane*32+channel_lane)*18+:18]=
          read_value[token_lane*18+:18];
      end
    end endgenerate

    always @(posedge clk) begin
      if(!rst_n) normalized_read_data_valid<=1'b0;
      else normalized_read_data_valid<=normalized_read_valid;
    end
    always @(posedge clk) begin
      if(!rst_n) begin state<=IDLE;active_group<=0;done<=0;end
      else begin
        done<=0;
        if(launch_initial) begin active_group<=0;state<=STATS;end
        else if(launch_next) state<=STATS;
        else if(state==STATS && replay_done) state<=WAIT_FINAL;
        else if(launch_final) state<=FINAL;
        else if(state==FINAL && norm_done) begin
          if(active_group==15) begin state<=IDLE;done<=1;end
          else begin active_group<=active_group+1'b1;state<=WAIT_NEXT;end
        end
      end
    end
    always @(posedge clk) begin
`ifndef SYNTHESIS
      if(rst_n && replay_output_valid && replay_output_group!=active_group)
        $error("norm1 replay changed groups");
      if(rst_n && norm_output_valid && norm_output_group!=active_group)
        $error("norm1 output changed groups");
`endif
    end
endmodule
