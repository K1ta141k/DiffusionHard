`timescale 1ns/1ps

module mdlm_weight_slice_dma #(
    parameter integer SECTION_ID = 7,
    parameter integer INPUT_SIZE = 768,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 8,
    parameter integer OUTPUT_TILE_WIDTH = 9,
    parameter integer TAG_WIDTH = 16,
    parameter integer K_TILES = INPUT_SIZE / 32,
    parameter integer K_TILE_WIDTH = (K_TILES <= 1) ? 1 : $clog2(K_TILES),
    parameter integer RECORD_WIDTH = N_LANES * 32 * DATA_WIDTH,
    parameter integer EXPECTED_PAYLOAD_BYTES = RECORD_WIDTH / 8
) (
    input  wire clk,
    input  wire rst_n,
    input  wire [63:0] block_base_address,

    input  wire command_valid,
    output wire command_ready,
    input  wire command_bank,
    input  wire [OUTPUT_TILE_WIDTH-1:0] command_output_tile,

    output wire weight_load_valid,
    input  wire weight_load_ready,
    output wire weight_load_bank,
    output wire [K_TILE_WIDTH-1:0] weight_load_k_tile,
    output wire [RECORD_WIDTH-1:0] weight_load_data,

    output wire [63:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [1:0] m_axi_arburst,
    output wire m_axi_arvalid,
    input  wire m_axi_arready,
    input  wire [511:0] m_axi_rdata,
    input  wire [1:0] m_axi_rresp,
    input  wire m_axi_rlast,
    input  wire m_axi_rvalid,
    output wire m_axi_rready,

    output wire busy,
    output reg  done,
    output wire protocol_error,
    output wire [63:0] bytes_read,
    output wire [63:0] address_stall_cycles,
    output wire [63:0] data_stall_cycles
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_REQUEST = 2'd1;
    localparam [1:0] STATE_RECORD = 2'd2;

    reg [1:0] state;
    reg active_bank;
    reg [OUTPUT_TILE_WIDTH-1:0] active_output_tile;
    reg [K_TILE_WIDTH-1:0] active_k_tile;
    reg geometry_error;
    reg transfer_error;

    wire dma_request_ready;
    wire dma_request_valid = state == STATE_REQUEST;
    wire [13:0] dma_record_index = active_output_tile * K_TILES
        + active_k_tile;
    wire dma_stream_valid;
    wire dma_stream_ready;
    wire [511:0] dma_stream_data;
    wire dma_stream_last;
    wire [TAG_WIDTH-1:0] dma_stream_tag;
    wire [5:0] dma_byte_offset;
    wire [9:0] dma_payload_bytes;
    wire dma_busy;
    wire dma_done;
    wire dma_invalid_request;
    wire dma_protocol_error;

    wire record_valid;
    wire record_ready = state == STATE_RECORD && weight_load_ready;
    wire [TAG_WIDTH-1:0] record_tag;
    wire adapter_protocol_error;

    assign command_ready = state == STATE_IDLE;
    assign weight_load_valid = state == STATE_RECORD && record_valid;
    assign weight_load_bank = active_bank;
    assign weight_load_k_tile = active_k_tile;
    assign busy = state != STATE_IDLE;
    assign protocol_error = dma_protocol_error || adapter_protocol_error
        || geometry_error || transfer_error;

    mdlm_block_parameter_dma #(
        .TAG_WIDTH(TAG_WIDTH)
    ) parameter_dma (
        .clk(clk),.rst_n(rst_n),.block_base_address(block_base_address),
        .request_valid(dma_request_valid),.request_ready(dma_request_ready),
        .request_section_id(SECTION_ID[3:0]),
        .request_record_index(dma_record_index),
        .request_tag({{(TAG_WIDTH-K_TILE_WIDTH){1'b0}}, active_k_tile}),
        .m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),.m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),.stream_valid(dma_stream_valid),
        .stream_ready(dma_stream_ready),.stream_data(dma_stream_data),
        .stream_last(dma_stream_last),.stream_tag(dma_stream_tag),
        .stream_record_byte_offset(dma_byte_offset),
        .stream_record_payload_bytes(dma_payload_bytes),.busy(dma_busy),
        .done(dma_done),.invalid_request(dma_invalid_request),
        .protocol_error(dma_protocol_error),.bytes_read(bytes_read),
        .address_stall_cycles(address_stall_cycles),
        .data_stall_cycles(data_stall_cycles)
    );

    fixed_weight_record_adapter #(
        .N_LANES(N_LANES),.DATA_WIDTH(DATA_WIDTH),.TAG_WIDTH(TAG_WIDTH)
    ) record_adapter (
        .clk(clk),.rst_n(rst_n),.stream_valid(dma_stream_valid),
        .stream_ready(dma_stream_ready),.stream_data(dma_stream_data),
        .stream_last(dma_stream_last),.stream_tag(dma_stream_tag),
        .record_valid(record_valid),.record_ready(record_ready),
        .record_data(weight_load_data),.record_tag(record_tag),
        .protocol_error(adapter_protocol_error)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_bank <= 1'b0;
            active_output_tile <= 0;
            active_k_tile <= 0;
            geometry_error <= 1'b0;
            transfer_error <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && command_valid) begin
                state <= STATE_REQUEST;
                active_bank <= command_bank;
                active_output_tile <= command_output_tile;
                active_k_tile <= 0;
                geometry_error <= 1'b0;
                transfer_error <= 1'b0;
            end else if (state == STATE_REQUEST && dma_request_ready) begin
                state <= STATE_RECORD;
            end else if (state == STATE_RECORD) begin
                if (dma_stream_valid && dma_stream_ready
                    && (dma_byte_offset != 0
                        || dma_payload_bytes != EXPECTED_PAYLOAD_BYTES))
                    geometry_error <= 1'b1;
                if (dma_done && (dma_invalid_request || dma_protocol_error)) begin
                    transfer_error <= 1'b1;
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else if (record_valid && weight_load_ready) begin
                    if (record_tag[K_TILE_WIDTH-1:0] != active_k_tile)
                        transfer_error <= 1'b1;
                    if (active_k_tile == K_TILES-1) begin
                        state <= STATE_IDLE;
                        done <= 1'b1;
                    end else begin
                        active_k_tile <= active_k_tile + 1'b1;
                        state <= STATE_REQUEST;
                    end
                end
            end
        end
    end

    initial begin
        if (INPUT_SIZE < 32 || INPUT_SIZE % 32 != 0)
            $error("weight slice input size must be divisible by 32");
        if (TAG_WIDTH < K_TILE_WIDTH)
            $error("tag width must carry the input tile index");
        if (EXPECTED_PAYLOAD_BYTES > 1023)
            $error("record payload exceeds DMA metadata width");
    end

endmodule
