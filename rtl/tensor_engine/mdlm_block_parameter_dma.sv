`timescale 1ns/1ps

module mdlm_block_parameter_dma #(
    parameter integer TAG_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,

    input  wire [63:0] block_base_address,
    input  wire request_valid,
    output wire request_ready,
    input  wire [3:0] request_section_id,
    input  wire [13:0] request_record_index,
    input  wire [TAG_WIDTH-1:0] request_tag,

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

    output wire stream_valid,
    input  wire stream_ready,
    output wire [511:0] stream_data,
    output wire stream_last,
    output wire [TAG_WIDTH-1:0] stream_tag,
    output wire [5:0] stream_record_byte_offset,
    output wire [9:0] stream_record_payload_bytes,

    output wire busy,
    output wire done,
    output reg  invalid_request,
    output wire protocol_error,
    output wire [63:0] bytes_read,
    output wire [63:0] address_stall_cycles,
    output wire [63:0] data_stall_cycles
);

    wire generated_valid;
    wire [63:0] generated_address;
    wire [6:0] generated_beats;
    wire [5:0] generated_byte_offset;
    wire [9:0] generated_payload_bytes;
    wire master_command_ready;
    wire master_done;
    reg [5:0] active_byte_offset;
    reg [9:0] active_payload_bytes;

    assign request_ready = master_command_ready;
    assign done = master_done || invalid_request;
    assign stream_record_byte_offset = active_byte_offset;
    assign stream_record_payload_bytes = active_payload_bytes;

    mdlm_block_parameter_address_generator address_generator (
        .block_base_address(block_base_address),
        .section_id(request_section_id),
        .record_index(request_record_index),
        .record_valid(generated_valid),
        .axi_address(generated_address),
        .burst_beats(generated_beats),
        .record_byte_offset(generated_byte_offset),
        .record_payload_bytes(generated_payload_bytes)
    );

    axi512_read_burst_master #(
        .TAG_WIDTH(TAG_WIDTH)
    ) read_master (
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(request_valid && generated_valid),
        .command_ready(master_command_ready),
        .command_address(generated_address),
        .command_beats(generated_beats),
        .command_tag(request_tag),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .stream_valid(stream_valid),
        .stream_ready(stream_ready),
        .stream_data(stream_data),
        .stream_last(stream_last),
        .stream_tag(stream_tag),
        .busy(busy),
        .done(master_done),
        .protocol_error(protocol_error),
        .bytes_read(bytes_read),
        .address_stall_cycles(address_stall_cycles),
        .data_stall_cycles(data_stall_cycles)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            active_byte_offset <= 0;
            active_payload_bytes <= 0;
            invalid_request <= 1'b0;
        end else begin
            invalid_request <= 1'b0;
            if (request_valid && request_ready) begin
                if (generated_valid) begin
                    active_byte_offset <= generated_byte_offset;
                    active_payload_bytes <= generated_payload_bytes;
                end else begin
                    invalid_request <= 1'b1;
                end
            end
        end
    end

endmodule
