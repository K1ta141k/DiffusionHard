`timescale 1ns/1ps

module attention_canvas_grouped_scratchpad_banked #(
    parameter integer DATA_WIDTH=18,
    parameter integer BANKS=64,
    parameter integer GROUP_DEPTH=192
)(
    input wire clk,input wire rst_n,
    input wire tile_valid,output wire tile_ready,
    input wire [3:0] tile_head,input wire [3:0] tile_group,
    input wire [3:0] tile_channel_tile,input wire [2:0] tile_valid_channels,
    input wire [4*6*DATA_WIDTH-1:0] tile_data_packed,
    output reg tile_done,
    input wire read_valid,input wire [3:0] read_head,
    input wire [5:0] read_token,output reg read_data_valid,
    output wire [BANKS*DATA_WIDTH-1:0] read_data_packed,
    input wire group_read_valid,input wire [3:0] group_read_head,
    input wire [3:0] group_read_group,output reg group_read_data_valid,
    output wire [4*BANKS*DATA_WIDTH-1:0] group_read_data_packed
);
    wire any_read=read_valid||group_read_valid;
    wire [7:0] active_read_address=group_read_valid
      ? {group_read_head,group_read_group}:{read_head,read_token[5:2]};
    reg [1:0] active_read_token_lane;
    assign tile_ready=!any_read;

    genvar bank_index;
    generate for(bank_index=0;bank_index<BANKS;
      bank_index=bank_index+1)begin:canvas_banks
      localparam integer BANK_TILE=bank_index/6;
      localparam integer BANK_LANE=bank_index%6;
      wire bank_write=tile_valid&&tile_ready
        &&tile_channel_tile==BANK_TILE&&BANK_LANE<tile_valid_channels;
      wire [4*DATA_WIDTH-1:0] bank_write_word={
        tile_data_packed[(3*6+BANK_LANE)*DATA_WIDTH+:DATA_WIDTH],
        tile_data_packed[(2*6+BANK_LANE)*DATA_WIDTH+:DATA_WIDTH],
        tile_data_packed[(1*6+BANK_LANE)*DATA_WIDTH+:DATA_WIDTH],
        tile_data_packed[(0*6+BANK_LANE)*DATA_WIDTH+:DATA_WIDTH]};
      (* ram_style="block" *) reg [4*DATA_WIDTH-1:0] memory [0:191];
      reg [4*DATA_WIDTH-1:0] read_word;
      always @(posedge clk)begin
        if(bank_write)memory[{tile_head,tile_group}]<=bank_write_word;
        else if(any_read)read_word<=memory[active_read_address];
      end
      assign read_data_packed[bank_index*DATA_WIDTH+:DATA_WIDTH]=
        read_word[active_read_token_lane*DATA_WIDTH+:DATA_WIDTH];
      genvar token_lane;
      for(token_lane=0;token_lane<4;token_lane=token_lane+1)begin:group_tokens
        assign group_read_data_packed[
          (token_lane*BANKS+bank_index)*DATA_WIDTH+:DATA_WIDTH]=
          read_word[token_lane*DATA_WIDTH+:DATA_WIDTH];
      end
    end endgenerate

    always @(posedge clk)begin
      if(!rst_n)begin tile_done<=0;read_data_valid<=0;
        group_read_data_valid<=0;active_read_token_lane<=0;end
      else begin
        tile_done<=tile_valid&&tile_ready;
        read_data_valid<=read_valid;
        group_read_data_valid<=group_read_valid;
        if(read_valid)active_read_token_lane<=read_token[1:0];
      end
    end

    initial begin
      if(BANKS!=64||GROUP_DEPTH!=192)
        $error("grouped attention canvas requires 64 banks and 192 groups");
    end
endmodule
