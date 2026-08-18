`timescale 1ns/1ps

module mdlm_parameter_record_dma #(
    parameter integer SECTION_ID = 0,
    parameter integer RECORD_WIDTH = 252,
    parameter integer EXPECTED_PAYLOAD_BYTES = (RECORD_WIDTH + 7) / 8,
    parameter integer COMPACT_RECORD = 0,
    parameter integer TAG_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,
    input  wire [63:0] block_base_address,
    input  wire request_valid,
    output wire request_ready,
    input  wire [13:0] request_record_index,
    input  wire [TAG_WIDTH-1:0] request_tag,
    output wire record_valid,
    input  wire record_ready,
    output wire [RECORD_WIDTH-1:0] record_data,
    output wire [TAG_WIDTH-1:0] record_tag,
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
    output wire done,
    output wire invalid_request,
    output wire protocol_error,
    output wire [63:0] bytes_read,
    output wire [63:0] address_stall_cycles,
    output wire [63:0] data_stall_cycles
);

    wire dma_request_ready;
    wire dma_stream_valid;
    wire dma_stream_ready;
    wire [511:0] dma_stream_data;
    wire dma_stream_last;
    wire [TAG_WIDTH-1:0] dma_stream_tag;
    wire [5:0] dma_byte_offset;
    wire [9:0] dma_payload_bytes;
    wire dma_busy;
    wire dma_done;
    wire dma_protocol_error;
    wire adapter_protocol_error;

    assign request_ready = dma_request_ready && !record_valid;
    assign busy = dma_busy || record_valid;
    assign done = (record_valid && record_ready)
        || (dma_done && (invalid_request || dma_protocol_error));
    assign protocol_error = dma_protocol_error || adapter_protocol_error;

    mdlm_block_parameter_dma #(.TAG_WIDTH(TAG_WIDTH)) parameter_dma (
        .clk(clk),.rst_n(rst_n),.block_base_address(block_base_address),
        .request_valid(request_valid && !record_valid),
        .request_ready(dma_request_ready),.request_section_id(SECTION_ID[3:0]),
        .request_record_index(request_record_index),.request_tag(request_tag),
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
        .done(dma_done),.invalid_request(invalid_request),
        .protocol_error(dma_protocol_error),.bytes_read(bytes_read),
        .address_stall_cycles(address_stall_cycles),
        .data_stall_cycles(data_stall_cycles)
    );

    generate
        if (COMPACT_RECORD) begin : compact
            compact_table_record_adapter #(
                .RECORD_WIDTH(RECORD_WIDTH),
                .EXPECTED_PAYLOAD_BYTES(EXPECTED_PAYLOAD_BYTES),
                .TAG_WIDTH(TAG_WIDTH)
            ) adapter (
                .clk(clk),.rst_n(rst_n),.stream_valid(dma_stream_valid),
                .stream_ready(dma_stream_ready),.stream_data(dma_stream_data),
                .stream_last(dma_stream_last),.stream_tag(dma_stream_tag),
                .stream_record_byte_offset(dma_byte_offset),
                .stream_record_payload_bytes(dma_payload_bytes),
                .record_valid(record_valid),.record_ready(record_ready),
                .record_data(record_data),.record_tag(record_tag),
                .protocol_error(adapter_protocol_error)
            );
        end else begin : aligned
            fixed_aligned_record_adapter #(
                .RECORD_WIDTH(RECORD_WIDTH),
                .EXPECTED_PAYLOAD_BYTES(EXPECTED_PAYLOAD_BYTES),
                .TAG_WIDTH(TAG_WIDTH)
            ) adapter (
                .clk(clk),.rst_n(rst_n),.stream_valid(dma_stream_valid),
                .stream_ready(dma_stream_ready),.stream_data(dma_stream_data),
                .stream_last(dma_stream_last),.stream_tag(dma_stream_tag),
                .stream_record_byte_offset(dma_byte_offset),
                .stream_record_payload_bytes(dma_payload_bytes),
                .record_valid(record_valid),.record_ready(record_ready),
                .record_data(record_data),.record_tag(record_tag),
                .protocol_error(adapter_protocol_error)
            );
        end
    endgenerate

endmodule
