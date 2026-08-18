`timescale 1ns/1ps

module int8_mac_tile_256 (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire clear_accumulators,
    input  wire last_k_tile,
    input  wire [4*16*8-1:0] activations_packed,
    input  wire [4*16*8-1:0] weights_packed,
    output wire valid_out,
    output wire [4*4*32-1:0] accumulators_packed
);
    int8_mac_tile #(
        .M_LANES(4), .N_LANES(4), .K_LANES(16),
        .DATA_WIDTH(8), .ACC_WIDTH(32)
    ) implementation (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .clear_accumulators(clear_accumulators), .last_k_tile(last_k_tile),
        .activations_packed(activations_packed), .weights_packed(weights_packed),
        .valid_out(valid_out), .accumulators_packed(accumulators_packed)
    );
endmodule

module mixed_precision_mac_tile_pipelined_192 (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire narrow_int8_mode,
    input  wire clear_accumulators,
    input  wire last_k_tile,
    input  wire [7:0] tag_in,
    input  wire [1*32*18-1:0] activations_packed,
    input  wire [6*32*18-1:0] weights_packed,
    output wire valid_out,
    output wire [7:0] tag_out,
    output wire [1*6*48-1:0] accumulators_packed
);
    mixed_precision_mac_tile_pipelined #(
        .M_LANES(1), .N_LANES(6), .STORAGE_WIDTH(18),
        .ACC_WIDTH(48), .TAG_WIDTH(8)
    ) implementation (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .narrow_int8_mode(narrow_int8_mode),
        .clear_accumulators(clear_accumulators), .last_k_tile(last_k_tile),
        .tag_in(tag_in), .activations_packed(activations_packed),
        .weights_packed(weights_packed), .valid_out(valid_out),
        .tag_out(tag_out), .accumulators_packed(accumulators_packed)
    );
endmodule

module int8_mac_tile_512 (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire clear_accumulators,
    input  wire last_k_tile,
    input  wire [4*32*8-1:0] activations_packed,
    input  wire [4*32*8-1:0] weights_packed,
    output wire valid_out,
    output wire [4*4*32-1:0] accumulators_packed
);
    int8_mac_tile #(
        .M_LANES(4), .N_LANES(4), .K_LANES(32),
        .DATA_WIDTH(8), .ACC_WIDTH(32)
    ) implementation (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .clear_accumulators(clear_accumulators), .last_k_tile(last_k_tile),
        .activations_packed(activations_packed), .weights_packed(weights_packed),
        .valid_out(valid_out), .accumulators_packed(accumulators_packed)
    );
endmodule

module int8_mac_tile_768 (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire clear_accumulators,
    input  wire last_k_tile,
    input  wire [4*32*8-1:0] activations_packed,
    input  wire [6*32*8-1:0] weights_packed,
    output wire valid_out,
    output wire [4*6*32-1:0] accumulators_packed
);
    int8_mac_tile #(
        .M_LANES(4), .N_LANES(6), .K_LANES(32),
        .DATA_WIDTH(8), .ACC_WIDTH(32)
    ) implementation (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .clear_accumulators(clear_accumulators), .last_k_tile(last_k_tile),
        .activations_packed(activations_packed), .weights_packed(weights_packed),
        .valid_out(valid_out), .accumulators_packed(accumulators_packed)
    );
endmodule

module int8_mac_tile_pipelined_192 (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire clear_accumulators,
    input  wire last_k_tile,
    input  wire [7:0] tag_in,
    input  wire [1*32*8-1:0] activations_packed,
    input  wire [6*32*8-1:0] weights_packed,
    output wire valid_out,
    output wire [7:0] tag_out,
    output wire [1*6*32-1:0] accumulators_packed
);
    int8_mac_tile_pipelined #(
        .M_LANES(1), .N_LANES(6),
        .DATA_WIDTH(8), .ACC_WIDTH(32), .TAG_WIDTH(8)
    ) implementation (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .clear_accumulators(clear_accumulators), .last_k_tile(last_k_tile),
        .tag_in(tag_in), .activations_packed(activations_packed),
        .weights_packed(weights_packed), .valid_out(valid_out),
        .tag_out(tag_out), .accumulators_packed(accumulators_packed)
    );
endmodule
