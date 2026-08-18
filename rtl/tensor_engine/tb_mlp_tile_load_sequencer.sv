`timescale 1ns/1ps

module tb_mlp_tile_load_sequencer;
  localparam integer WEIGHT_WIDTH=6*32*8;
  localparam integer METADATA_WIDTH=37;
  reg clk=0,rst_n=0,command_valid=0,command_bank=0;
  reg weight_valid=0,metadata_valid=0;
  reg [WEIGHT_WIDTH-1:0] weight_data=0;
  reg [METADATA_WIDTH-1:0] metadata_data=0;
  reg weight_sink_ready=0,metadata_sink_ready=0;
  wire command_ready,weight_ready,metadata_ready;
  wire weight_load_valid,weight_load_bank,metadata_load_valid;
  wire metadata_load_bank,busy,done;
  wire [1:0] weight_load_k_tile;
  wire [WEIGHT_WIDTH-1:0] weight_load_data;
  wire [METADATA_WIDTH-1:0] metadata_load_data;
  integer accepted_weights=0,accepted_metadata=0,commands=0,cycles=0;
  reg expected_bank=0;

  mlp_tile_load_sequencer #(
    .INPUT_SIZE(96),.METADATA_WIDTH(METADATA_WIDTH)
  ) dut(
    .clk(clk),.rst_n(rst_n),.command_valid(command_valid),
    .command_bank(command_bank),.command_ready(command_ready),
    .weight_stream_valid(weight_valid),.weight_stream_ready(weight_ready),
    .weight_stream_data(weight_data),.metadata_stream_valid(metadata_valid),
    .metadata_stream_ready(metadata_ready),.metadata_stream_data(metadata_data),
    .weight_load_valid(weight_load_valid),.weight_load_bank(weight_load_bank),
    .weight_load_k_tile(weight_load_k_tile),.weight_load_data(weight_load_data),
    .weight_load_ready(weight_sink_ready),
    .metadata_load_valid(metadata_load_valid),
    .metadata_load_bank(metadata_load_bank),
    .metadata_load_data(metadata_load_data),
    .metadata_load_ready(metadata_sink_ready),.busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    cycles=cycles+1;
    weight_sink_ready <= cycles[0] || cycles[2];
    metadata_sink_ready <= !cycles[1];
    if(command_valid && command_ready) begin
      commands=commands+1;expected_bank=command_bank;
    end
    if(weight_load_valid && weight_sink_ready) begin
      if(weight_load_bank!==expected_bank ||
         weight_load_k_tile!==accepted_weights%3 ||
         weight_load_data!==weight_data)
        $fatal(1,"weight load mismatch");
      accepted_weights=accepted_weights+1;
    end
    if(metadata_load_valid && metadata_sink_ready) begin
      if(metadata_load_bank!==expected_bank ||
         metadata_load_data!==metadata_data)
        $fatal(1,"metadata load mismatch");
      accepted_metadata=accepted_metadata+1;
    end
  end

  task run_command;
    input bank;
    input metadata_first;
    integer sent_weights;
    reg sent_metadata;
    begin
      sent_weights=0;sent_metadata=0;
      @(negedge clk);command_bank=bank;command_valid=1;
      wait(command_ready);@(posedge clk);@(negedge clk);command_valid=0;
      while(!done) begin
        weight_valid=(sent_weights<3);
        weight_data={WEIGHT_WIDTH{1'b0}} | (bank*100+sent_weights+1);
        metadata_valid=!sent_metadata &&
          (metadata_first ? sent_weights==0 : sent_weights==2);
        metadata_data=37'h123400000 | (bank*7+3);
        @(posedge clk);
        if(weight_valid && weight_ready) sent_weights=sent_weights+1;
        if(metadata_valid && metadata_ready) sent_metadata=1;
        @(negedge clk);
      end
      weight_valid=0;metadata_valid=0;
      if(sent_weights!=3 || !sent_metadata) $fatal(1,"source count mismatch");
      if(command_ready) $fatal(1,"done guard did not hold command closed");
      @(posedge clk);@(negedge clk);
    end
  endtask

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    run_command(0,1);run_command(1,0);
    repeat(3) @(posedge clk);
    if(commands!=2 || accepted_weights!=6 || accepted_metadata!=2 || busy)
      $fatal(1,"tile loader final count mismatch");
    $display("tb_mlp_tile_load_sequencer: PASS cycles=%0d",cycles);
    $finish;
  end
  initial begin repeat(200) @(posedge clk);$fatal(1,"timeout");end
endmodule
