`timescale 1ns/1ps

module tb_attention_head_scratchpad_banked;
    reg clk=0,load_valid=0;
    reg [5:0] load_token=0,load_channel=0;
    reg signed [17:0] load_query_q12=0,load_key_q12=0,load_value_q12=0;
    reg query_read_valid=0,key_read_valid=0,value_read_valid=0;
    reg [5:0] query_read_token=0,key_read_token=0,value_read_channel=0;
    reg [1:0] query_read_channel_block=0,key_read_channel_block=0;
    reg [1:0] value_read_key_block=0;
    wire query_data_valid,key_data_valid,value_data_valid;
    wire [287:0] query_data_packed,key_data_packed,value_data_packed;
    integer token,channel,lane;

    attention_head_scratchpad_banked dut(
      .clk(clk),.load_valid(load_valid),.load_token(load_token),
      .load_channel(load_channel),.load_query_q12(load_query_q12),
      .load_key_q12(load_key_q12),.load_value_q12(load_value_q12),
      .query_read_valid(query_read_valid),.query_read_token(query_read_token),
      .query_read_channel_block(query_read_channel_block),
      .query_data_valid(query_data_valid),.query_data_packed(query_data_packed),
      .key_read_valid(key_read_valid),.key_read_token(key_read_token),
      .key_read_channel_block(key_read_channel_block),
      .key_data_valid(key_data_valid),.key_data_packed(key_data_packed),
      .value_read_valid(value_read_valid),.value_read_key_block(value_read_key_block),
      .value_read_channel(value_read_channel),.value_data_valid(value_data_valid),
      .value_data_packed(value_data_packed));
    always #2 clk=~clk;

    function integer q_expected(input integer t,input integer c);
      q_expected=t*101+c*3-2000;
    endfunction
    function integer k_expected(input integer t,input integer c);
      k_expected=t*53-c*7+900;
    endfunction
    function integer v_expected(input integer t,input integer c);
      v_expected=t*17+c*11-700;
    endfunction

    initial begin
      repeat(2) @(posedge clk);
      for(token=0;token<64;token=token+1)
        for(channel=0;channel<64;channel=channel+1) begin
          @(negedge clk);load_valid=1;load_token=token;load_channel=channel;
          load_query_q12=q_expected(token,channel);
          load_key_q12=k_expected(token,channel);
          load_value_q12=v_expected(token,channel);
        end
      @(negedge clk);load_valid=0;
      query_read_valid=1;query_read_token=6'd7;query_read_channel_block=2'd2;
      key_read_valid=1;key_read_token=6'd41;key_read_channel_block=2'd1;
      value_read_valid=1;value_read_key_block=2'd3;value_read_channel=6'd21;
      @(posedge clk);#1;
      if(!query_data_valid || !key_data_valid || !value_data_valid)
        $fatal(1,"missing scratchpad read valid");
      for(lane=0;lane<16;lane=lane+1) begin
        if($signed(query_data_packed[lane*18 +: 18])!==q_expected(7,32+lane))
          $fatal(1,"query bank mismatch %0d",lane);
        if($signed(key_data_packed[lane*18 +: 18])!==k_expected(41,16+lane))
          $fatal(1,"key bank mismatch %0d",lane);
        if($signed(value_data_packed[lane*18 +: 18])!==v_expected(48+lane,21))
          $fatal(1,"value bank mismatch %0d",lane);
      end
      @(negedge clk);query_read_valid=0;key_read_valid=0;value_read_valid=0;
      $display("tb_attention_head_scratchpad_banked: PASS");$finish;
    end
    initial begin repeat(5000) @(posedge clk);$fatal(1,"timeout");end
endmodule
