`timescale 1ns/1ps

module tb_mixed_precision_mac_tile_pipelined;
    localparam integer M_LANES = 2;
    localparam integer N_LANES = 2;
    localparam integer WIDTH = 18;
    localparam integer ACC_WIDTH = 48;

    reg clk=0, rst_n=0, valid_in=0, narrow_int8_mode=0;
    reg clear_accumulators=0, last_k_tile=0;
    reg [3:0] tag_in=0;
    reg [M_LANES*32*WIDTH-1:0] activations_packed=0;
    reg [N_LANES*32*WIDTH-1:0] weights_packed=0;
    wire valid_out;
    wire [3:0] tag_out;
    wire [M_LANES*N_LANES*ACC_WIDTH-1:0] accumulators_packed;
    integer expected [0:2][0:M_LANES-1][0:N_LANES-1];
    integer output_count=0,m,n,k,tile;
    integer av,wv;
    reg signed [WIDTH-1:0] actual;

    mixed_precision_mac_tile_pipelined #(
        .M_LANES(M_LANES),.N_LANES(N_LANES),.ACC_WIDTH(ACC_WIDTH),
        .TAG_WIDTH(4)
    ) dut(
        .clk(clk),.rst_n(rst_n),.valid_in(valid_in),
        .narrow_int8_mode(narrow_int8_mode),
        .clear_accumulators(clear_accumulators),.last_k_tile(last_k_tile),
        .tag_in(tag_in),.activations_packed(activations_packed),
        .weights_packed(weights_packed),.valid_out(valid_out),
        .tag_out(tag_out),.accumulators_packed(accumulators_packed));
    always #2 clk=~clk;

    task drive_fixed_tile;
      input integer transaction;
      input integer tile_number;
      input integer probability_mode;
      begin
        @(negedge clk); activations_packed=0;weights_packed=0;
        narrow_int8_mode=0;
        for(m=0;m<M_LANES;m=m+1) for(k=0;k<32;k=k+1) begin
          if(probability_mode) av=60000-m*7000-k*3;
          else av=(m+1)*500+tile_number*71+k-16;
          activations_packed[(m*32+k)*WIDTH +: WIDTH]=av;
        end
        for(n=0;n<N_LANES;n=n+1) for(k=0;k<32;k=k+1) begin
          if(probability_mode) wv=(n+1)*3-(k%7);
          else wv=(n+1)*-300+tile_number*43+k;
          weights_packed[(n*32+k)*WIDTH +: WIDTH]=wv;
        end
        for(m=0;m<M_LANES;m=m+1) for(n=0;n<N_LANES;n=n+1)
          for(k=0;k<32;k=k+1) begin
            av=$signed(activations_packed[(m*32+k)*WIDTH +: WIDTH]);
            wv=$signed(weights_packed[(n*32+k)*WIDTH +: WIDTH]);
            expected[transaction][m][n]=expected[transaction][m][n]+av*wv;
          end
        valid_in=1;clear_accumulators=(tile_number==0);last_k_tile=(tile_number==1);
        if(probability_mode) last_k_tile=1;
        tag_in=transaction;
      end
    endtask

    task drive_int8_tile;
      input integer transaction;
      begin
        @(negedge clk);activations_packed={M_LANES*32*WIDTH{1'b1}};
        weights_packed={N_LANES*32*WIDTH{1'b1}};narrow_int8_mode=1;
        for(m=0;m<M_LANES;m=m+1) for(k=0;k<32;k=k+1) begin
          av=((m*11+k)%17)-8;
          activations_packed[(m*32+k)*WIDTH +: 8]=av;
        end
        for(n=0;n<N_LANES;n=n+1) for(k=0;k<32;k=k+1) begin
          wv=((n*7+k*3)%15)-7;
          weights_packed[(n*32+k)*WIDTH +: 8]=wv;
        end
        for(m=0;m<M_LANES;m=m+1) for(n=0;n<N_LANES;n=n+1)
          for(k=0;k<32;k=k+1) begin
            av=$signed(activations_packed[(m*32+k)*WIDTH +: 8]);
            wv=$signed(weights_packed[(n*32+k)*WIDTH +: 8]);
            expected[transaction][m][n]=expected[transaction][m][n]+av*wv;
          end
        valid_in=1;clear_accumulators=1;last_k_tile=1;tag_in=transaction;
      end
    endtask

    always @(posedge clk) begin
      #1;
      if(valid_out) begin
        if(tag_out!==output_count) $fatal(1,"tag mismatch");
        for(m=0;m<M_LANES;m=m+1) for(n=0;n<N_LANES;n=n+1)
          if($signed(accumulators_packed[(m*N_LANES+n)*ACC_WIDTH +: ACC_WIDTH])
             !== expected[output_count][m][n])
            $fatal(1,"mixed MAC mismatch transaction %0d lane %0d,%0d",
                   output_count,m,n);
        output_count=output_count+1;
      end
    end

    initial begin
      for(tile=0;tile<3;tile=tile+1) for(m=0;m<M_LANES;m=m+1)
        for(n=0;n<N_LANES;n=n+1) expected[tile][m][n]=0;
      repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
      drive_fixed_tile(0,0,0); drive_fixed_tile(0,1,0);
      drive_int8_tile(1); drive_fixed_tile(2,0,1);
      @(negedge clk);valid_in=0;repeat(12) @(posedge clk);
      if(output_count!=3) $fatal(1,"missing mixed MAC outputs");
      $display("tb_mixed_precision_mac_tile_pipelined: PASS");$finish;
    end
endmodule
