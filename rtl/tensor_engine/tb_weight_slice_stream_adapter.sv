`timescale 1ns/1ps

module tb_weight_slice_stream_adapter;
    localparam integer INPUT_SIZE = 64;
    localparam integer N_LANES = 1;
    localparam integer DATA_WIDTH = 8;
    localparam integer STREAM_WIDTH = 64;
    localparam integer K_TILE_WIDTH = 1;
    localparam integer TILE_WIDTH = N_LANES * 32 * DATA_WIDTH;
    localparam integer BEATS_PER_TILE = TILE_WIDTH / STREAM_WIDTH;
    localparam integer K_TILES = INPUT_SIZE / 32;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg command_valid = 1'b0;
    reg command_bank = 1'b0;
    wire command_ready;
    reg stream_valid = 1'b0;
    wire stream_ready;
    reg [STREAM_WIDTH-1:0] stream_data = '0;
    reg stream_last = 1'b0;
    wire weight_load_valid;
    wire weight_load_bank;
    wire [K_TILE_WIDTH-1:0] weight_load_k_tile;
    wire [TILE_WIDTH-1:0] weight_load_data;
    reg weight_load_ready = 1'b1;
    wire busy;
    wire done;
    wire protocol_error;

    integer beat_index;
    integer tile_index;
    integer observed_tiles = 0;
    integer done_count = 0;
    reg [STREAM_WIDTH-1:0] expected_beat;

    weight_slice_stream_adapter #(
        .INPUT_SIZE(INPUT_SIZE),
        .N_LANES(N_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .STREAM_WIDTH(STREAM_WIDTH),
        .K_TILE_WIDTH(K_TILE_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(command_valid),
        .command_bank(command_bank),
        .command_ready(command_ready),
        .stream_valid(stream_valid),
        .stream_ready(stream_ready),
        .stream_data(stream_data),
        .stream_last(stream_last),
        .weight_load_valid(weight_load_valid),
        .weight_load_bank(weight_load_bank),
        .weight_load_k_tile(weight_load_k_tile),
        .weight_load_data(weight_load_data),
        .weight_load_ready(weight_load_ready),
        .busy(busy),
        .done(done),
        .protocol_error(protocol_error)
    );

    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (weight_load_valid) begin
            if (weight_load_bank !== 1'b1) $fatal(1, "bank mismatch");
            if (weight_load_k_tile !== observed_tiles[K_TILE_WIDTH-1:0]) begin
                $fatal(1, "K tile mismatch");
            end
            for (beat_index = 0; beat_index < BEATS_PER_TILE; beat_index = beat_index + 1) begin
                expected_beat = 64'h1000
                    + observed_tiles*BEATS_PER_TILE + beat_index;
                if (weight_load_data[beat_index*STREAM_WIDTH +: STREAM_WIDTH]
                    !== expected_beat) begin
                    $fatal(1, "tile %0d beat %0d data mismatch", observed_tiles, beat_index);
                end
            end
            observed_tiles = observed_tiles + 1;
        end
        #1;
        if (done) done_count = done_count + 1;
    end

    initial begin
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        command_bank = 1'b1;
        command_valid = 1'b1;
        @(posedge clk);
        #1;
        @(negedge clk);
        command_valid = 1'b0;

        for (tile_index = 0; tile_index < K_TILES; tile_index = tile_index + 1) begin
            for (beat_index = 0; beat_index < BEATS_PER_TILE; beat_index = beat_index + 1) begin
                @(negedge clk);
                stream_data = 64'h1000 + tile_index*BEATS_PER_TILE + beat_index;
                stream_last = (tile_index == K_TILES-1)
                    && (beat_index == BEATS_PER_TILE-1);
                stream_valid = 1'b1;
                if (tile_index == 0 && beat_index == BEATS_PER_TILE-1) begin
                    weight_load_ready = 1'b0;
                    #1;
                    if (stream_ready) $fatal(1, "third beat ignored downstream stall");
                    @(posedge clk);
                    #1;
                    @(negedge clk);
                    weight_load_ready = 1'b1;
                end
                #1;
                if (!stream_ready) $fatal(1, "stream did not resume");
                @(posedge clk);
                #1;
            end
        end
        @(negedge clk);
        stream_valid = 1'b0;
        stream_last = 1'b0;
        repeat (3) @(posedge clk);
        #1;

        if (observed_tiles !== K_TILES) $fatal(1, "missing output tiles");
        if (done_count !== 1) $fatal(1, "done pulse mismatch");
        if (protocol_error) $fatal(1, "protocol error asserted");
        $display("tb_weight_slice_stream_adapter: PASS");
        $finish;
    end
endmodule
