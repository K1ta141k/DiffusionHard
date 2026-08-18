`timescale 1ns/1ps

module axi512_read_burst_master #(
    parameter integer TAG_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,

    input  wire command_valid,
    output wire command_ready,
    input  wire [63:0] command_address,
    input  wire [6:0] command_beats,
    input  wire [TAG_WIDTH-1:0] command_tag,

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

    output wire busy,
    output reg  done,
    output reg  protocol_error,
    output reg  [63:0] bytes_read,
    output reg  [63:0] address_stall_cycles,
    output reg  [63:0] data_stall_cycles
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_ADDRESS = 2'd1;
    localparam [1:0] STATE_DATA = 2'd2;

    reg [1:0] state;
    reg [63:0] active_address;
    reg [6:0] active_beats;
    reg [6:0] accepted_beats;
    reg [TAG_WIDTH-1:0] active_tag;
    wire accepting_data = m_axi_rvalid && m_axi_rready;
    wire expected_last = accepted_beats == active_beats-1'b1;

    assign command_ready = state == STATE_IDLE;
    assign m_axi_araddr = active_address;
    assign m_axi_arlen = {1'b0, active_beats} - 1'b1;
    assign m_axi_arsize = 3'd6;
    assign m_axi_arburst = 2'b01;
    assign m_axi_arvalid = state == STATE_ADDRESS;
    assign m_axi_rready = state == STATE_DATA && stream_ready;
    assign stream_valid = state == STATE_DATA && m_axi_rvalid;
    assign stream_data = m_axi_rdata;
    assign stream_last = m_axi_rlast;
    assign stream_tag = active_tag;
    assign busy = state != STATE_IDLE;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_address <= 0;
            active_beats <= 0;
            accepted_beats <= 0;
            active_tag <= 0;
            done <= 1'b0;
            protocol_error <= 1'b0;
            bytes_read <= 0;
            address_stall_cycles <= 0;
            data_stall_cycles <= 0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && command_valid && command_ready) begin
                if (command_beats == 0) begin
                    protocol_error <= 1'b1;
                    done <= 1'b1;
                end else begin
                    active_address <= command_address;
                    active_beats <= command_beats;
                    accepted_beats <= 0;
                    active_tag <= command_tag;
                    protocol_error <= 1'b0;
                    state <= STATE_ADDRESS;
                end
            end else if (state == STATE_ADDRESS) begin
                if (m_axi_arready)
                    state <= STATE_DATA;
                else
                    address_stall_cycles <= address_stall_cycles + 1'b1;
            end else if (state == STATE_DATA) begin
                if (m_axi_rvalid && !stream_ready)
                    data_stall_cycles <= data_stall_cycles + 1'b1;
                if (accepting_data) begin
                    bytes_read <= bytes_read + 7'd64;
                    if (m_axi_rresp != 2'b00)
                        protocol_error <= 1'b1;
                    if (m_axi_rlast != expected_last)
                        protocol_error <= 1'b1;
                    if (m_axi_rlast || expected_last) begin
                        state <= STATE_IDLE;
                        done <= 1'b1;
                    end else begin
                        accepted_beats <= accepted_beats + 1'b1;
                    end
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && command_valid && command_beats == 0)
            $error("AXI read command must contain at least one beat");
        if (rst_n && state == STATE_DATA && m_axi_rvalid
            && accepted_beats >= active_beats)
            $error("AXI read master received more beats than requested");
`endif
    end

endmodule
