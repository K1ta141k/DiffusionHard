`timescale 1ns/1ps

module mdlm_qkv_output_tile_loader (
    input  wire clk,
    input  wire rst_n,
    input  wire [63:0] block_base_address,
    input  wire command_valid,
    output wire command_ready,
    input  wire [3:0] command_head,
    input  wire [1:0] command_kind,
    input  wire [3:0] command_channel_tile,

    output wire metadata_valid,
    input  wire metadata_ready,
    output wire [3:0] metadata_head,
    output wire [1:0] metadata_kind,
    output wire [3:0] metadata_channel_tile,
    output wire [6*24-1:0] metadata_multipliers_packed,
    output wire [6*18-1:0] metadata_biases_q12_packed,
    output wire weight_tile_valid,
    input  wire weight_tile_ready,
    output wire [3:0] weight_head,
    output wire [1:0] weight_kind,
    output wire [3:0] weight_channel_tile,
    output wire [4:0] weight_input_tile,
    output wire [6*32*16-1:0] weight_int16_packed,

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

    localparam [2:0] STATE_IDLE = 3'd0;
    localparam [2:0] STATE_METADATA_REQUEST = 3'd1;
    localparam [2:0] STATE_METADATA_RECORD = 3'd2;
    localparam [2:0] STATE_WEIGHT_REQUEST = 3'd3;
    localparam [2:0] STATE_WEIGHT_RECORD = 3'd4;

    reg [2:0] state;
    reg [3:0] active_head;
    reg [1:0] active_kind;
    reg [3:0] active_channel_tile;
    reg [4:0] active_input_tile;
    reg local_error;
    wire [8:0] head_times_33 = ({5'd0,active_head} << 5)
        + {5'd0,active_head};
    wire [8:0] kind_times_11 = ({7'd0,active_kind} << 3)
        + ({7'd0,active_kind} << 1) + {7'd0,active_kind};
    wire [8:0] output_tile_index = head_times_33 + kind_times_11
        + active_channel_tile;
    wire [13:0] output_tile_extended = {5'd0,output_tile_index};
    wire [13:0] weight_record_index = (output_tile_extended << 4)
        + (output_tile_extended << 3) + active_input_tile;

    wire dma_request_ready;
    wire dma_stream_valid;
    wire dma_stream_ready;
    wire [511:0] dma_stream_data;
    wire dma_stream_last;
    wire [15:0] dma_stream_tag;
    wire [5:0] dma_byte_offset;
    wire [9:0] dma_payload_bytes;
    wire dma_done;
    wire dma_invalid;
    wire dma_protocol_error;

    wire metadata_record_valid;
    wire metadata_record_ready = state == STATE_METADATA_RECORD
        && metadata_ready;
    wire [251:0] metadata_record_data;
    wire [15:0] metadata_record_tag;
    wire metadata_adapter_error;
    wire weight_record_valid;
    wire weight_record_ready = state == STATE_WEIGHT_RECORD
        && weight_tile_ready;
    wire [3071:0] weight_record_data;
    wire [15:0] weight_record_tag;
    wire weight_adapter_error;

    wire requesting_metadata = state == STATE_METADATA_REQUEST;
    wire requesting_weight = state == STATE_WEIGHT_REQUEST;
    wire receiving_metadata = state == STATE_METADATA_RECORD;
    wire receiving_weight = state == STATE_WEIGHT_RECORD;

    assign command_ready = state == STATE_IDLE;
    assign busy = state != STATE_IDLE;
    assign protocol_error = local_error || dma_protocol_error
        || metadata_adapter_error || weight_adapter_error;
    assign metadata_valid = receiving_metadata && metadata_record_valid;
    assign metadata_head = active_head;
    assign metadata_kind = active_kind;
    assign metadata_channel_tile = active_channel_tile;
    assign metadata_multipliers_packed = metadata_record_data[143:0];
    assign metadata_biases_q12_packed = metadata_record_data[251:144];
    assign weight_tile_valid = receiving_weight && weight_record_valid;
    assign weight_head = active_head;
    assign weight_kind = active_kind;
    assign weight_channel_tile = active_channel_tile;
    assign weight_input_tile = active_input_tile;
    assign weight_int16_packed = weight_record_data;
    assign dma_stream_ready = receiving_metadata
        ? metadata_record_ready || !metadata_record_valid
        : receiving_weight ? weight_record_ready || !weight_record_valid : 1'b0;

    mdlm_block_parameter_dma parameter_dma (
        .clk(clk),.rst_n(rst_n),.block_base_address(block_base_address),
        .request_valid(requesting_metadata || requesting_weight),
        .request_ready(dma_request_ready),
        .request_section_id(requesting_metadata ? 4'd0 : 4'd1),
        .request_record_index(requesting_metadata
            ? {5'd0, output_tile_index} : weight_record_index),
        .request_tag(requesting_metadata
            ? {1'b1, 15'd0} : {11'd0, active_input_tile}),
        .m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),.m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),.stream_valid(dma_stream_valid),
        .stream_ready(dma_stream_ready),.stream_data(dma_stream_data),
        .stream_last(dma_stream_last),.stream_tag(dma_stream_tag),
        .stream_record_byte_offset(dma_byte_offset),
        .stream_record_payload_bytes(dma_payload_bytes),.busy(),
        .done(dma_done),.invalid_request(dma_invalid),
        .protocol_error(dma_protocol_error),.bytes_read(bytes_read),
        .address_stall_cycles(address_stall_cycles),
        .data_stall_cycles(data_stall_cycles)
    );

    fixed_aligned_record_adapter #(
        .RECORD_WIDTH(252),.EXPECTED_PAYLOAD_BYTES(32)
    ) metadata_adapter (
        .clk(clk),.rst_n(rst_n),
        .stream_valid(dma_stream_valid && receiving_metadata),
        .stream_ready(),.stream_data(dma_stream_data),
        .stream_last(dma_stream_last),.stream_tag(dma_stream_tag),
        .stream_record_byte_offset(dma_byte_offset),
        .stream_record_payload_bytes(dma_payload_bytes),
        .record_valid(metadata_record_valid),.record_ready(metadata_record_ready),
        .record_data(metadata_record_data),.record_tag(metadata_record_tag),
        .protocol_error(metadata_adapter_error)
    );

    fixed_weight_record_adapter #(.DATA_WIDTH(16)) weight_adapter (
        .clk(clk),.rst_n(rst_n),
        .stream_valid(dma_stream_valid && receiving_weight),
        .stream_ready(),.stream_data(dma_stream_data),
        .stream_last(dma_stream_last),.stream_tag(dma_stream_tag),
        .record_valid(weight_record_valid),.record_ready(weight_record_ready),
        .record_data(weight_record_data),.record_tag(weight_record_tag),
        .protocol_error(weight_adapter_error)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_head <= 0;
            active_kind <= 0;
            active_channel_tile <= 0;
            active_input_tile <= 0;
            local_error <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && command_valid) begin
                active_head <= command_head;
                active_kind <= command_kind;
                active_channel_tile <= command_channel_tile;
                active_input_tile <= 0;
                local_error <= command_head >= 12 || command_kind >= 3
                    || command_channel_tile >= 11;
                if (command_head >= 12 || command_kind >= 3
                    || command_channel_tile >= 11) begin
                    done <= 1'b1;
                end else begin
                    state <= STATE_METADATA_REQUEST;
                end
            end else if (state == STATE_METADATA_REQUEST
                && dma_request_ready) begin
                state <= STATE_METADATA_RECORD;
            end else if (state == STATE_METADATA_RECORD) begin
                if (dma_done && (dma_invalid || dma_protocol_error)) begin
                    local_error <= 1'b1;
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else if (metadata_record_valid && metadata_ready) begin
                    if (metadata_record_tag != 16'h8000)
                        local_error <= 1'b1;
                    state <= STATE_WEIGHT_REQUEST;
                end
            end else if (state == STATE_WEIGHT_REQUEST
                && dma_request_ready) begin
                state <= STATE_WEIGHT_RECORD;
            end else if (state == STATE_WEIGHT_RECORD) begin
                if (dma_done && (dma_invalid || dma_protocol_error)) begin
                    local_error <= 1'b1;
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else if (weight_record_valid && weight_tile_ready) begin
                    if (weight_record_tag[4:0] != active_input_tile)
                        local_error <= 1'b1;
                    if (active_input_tile == 23) begin
                        state <= STATE_IDLE;
                        done <= 1'b1;
                    end else begin
                        active_input_tile <= active_input_tile + 1'b1;
                        state <= STATE_WEIGHT_REQUEST;
                    end
                end
            end
        end
    end

endmodule
