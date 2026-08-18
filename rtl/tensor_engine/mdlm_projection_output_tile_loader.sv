`timescale 1ns/1ps

module mdlm_projection_output_tile_loader (
    input  wire clk,
    input  wire rst_n,
    input  wire [63:0] block_base_address,
    input  wire command_valid,
    output wire command_ready,
    input  wire [6:0] command_output_tile,
    output wire metadata_valid,
    input  wire metadata_ready,
    output wire [6:0] metadata_output_tile,
    output wire [6*24-1:0] metadata_multipliers_packed,
    output wire weight_tile_valid,
    input  wire weight_tile_ready,
    output wire [6:0] weight_output_tile,
    output wire [4:0] weight_input_tile,
    output wire [6*32*8-1:0] weight_int8_packed,
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

    localparam [2:0] IDLE=0,META_REQUEST=1,META_RECORD=2,
        WEIGHT_REQUEST=3,WEIGHT_RECORD=4;
    reg [2:0] state;
    reg [6:0] active_output_tile;
    reg [4:0] active_input_tile;
    reg local_error;

    wire dma_request_ready,dma_stream_valid,dma_stream_last,dma_done;
    wire dma_invalid,dma_protocol_error;
    wire [511:0] dma_stream_data;
    wire [15:0] dma_stream_tag;
    wire [5:0] dma_byte_offset;
    wire [9:0] dma_payload_bytes;
    wire metadata_record_valid,weight_record_valid;
    wire [143:0] metadata_record_data;
    wire [1535:0] weight_record_data;
    wire [15:0] metadata_record_tag,weight_record_tag;
    wire metadata_error,weight_error;
    wire receiving_metadata=state==META_RECORD;
    wire receiving_weight=state==WEIGHT_RECORD;
    wire metadata_record_ready=receiving_metadata && metadata_ready;
    wire weight_record_ready=receiving_weight && weight_tile_ready;
    wire dma_stream_ready=receiving_metadata
        ? metadata_record_ready || !metadata_record_valid
        : receiving_weight ? weight_record_ready || !weight_record_valid : 1'b0;
    wire [13:0] output_tile_extended={7'd0,active_output_tile};
    wire [13:0] weight_record_index=(output_tile_extended<<4)
        +(output_tile_extended<<3)+active_input_tile;

    assign command_ready=state==IDLE;
    assign busy=state!=IDLE;
    assign protocol_error=local_error || dma_protocol_error
        || metadata_error || weight_error;
    assign metadata_valid=receiving_metadata && metadata_record_valid;
    assign metadata_output_tile=active_output_tile;
    assign metadata_multipliers_packed=metadata_record_data;
    assign weight_tile_valid=receiving_weight && weight_record_valid;
    assign weight_output_tile=active_output_tile;
    assign weight_input_tile=active_input_tile;
    assign weight_int8_packed=weight_record_data;

    mdlm_block_parameter_dma parameter_dma(
        .clk(clk),.rst_n(rst_n),.block_base_address(block_base_address),
        .request_valid(state==META_REQUEST || state==WEIGHT_REQUEST),
        .request_ready(dma_request_ready),
        .request_section_id(state==META_REQUEST ? 4'd3 : 4'd4),
        .request_record_index(state==META_REQUEST
            ? {7'd0,active_output_tile} : weight_record_index),
        .request_tag(state==META_REQUEST ? 16'h8000 : {11'd0,active_input_tile}),
        .m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),.m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),.stream_valid(dma_stream_valid),
        .stream_ready(dma_stream_ready),.stream_data(dma_stream_data),
        .stream_last(dma_stream_last),.stream_tag(dma_stream_tag),
        .stream_record_byte_offset(dma_byte_offset),
        .stream_record_payload_bytes(dma_payload_bytes),.busy(),.done(dma_done),
        .invalid_request(dma_invalid),.protocol_error(dma_protocol_error),
        .bytes_read(bytes_read),.address_stall_cycles(address_stall_cycles),
        .data_stall_cycles(data_stall_cycles));

    fixed_aligned_record_adapter #(.RECORD_WIDTH(144),
        .EXPECTED_PAYLOAD_BYTES(18)) metadata_adapter(
        .clk(clk),.rst_n(rst_n),
        .stream_valid(dma_stream_valid && receiving_metadata),.stream_ready(),
        .stream_data(dma_stream_data),.stream_last(dma_stream_last),
        .stream_tag(dma_stream_tag),.stream_record_byte_offset(dma_byte_offset),
        .stream_record_payload_bytes(dma_payload_bytes),
        .record_valid(metadata_record_valid),.record_ready(metadata_record_ready),
        .record_data(metadata_record_data),.record_tag(metadata_record_tag),
        .protocol_error(metadata_error));
    fixed_weight_record_adapter #(.DATA_WIDTH(8)) weight_adapter(
        .clk(clk),.rst_n(rst_n),
        .stream_valid(dma_stream_valid && receiving_weight),.stream_ready(),
        .stream_data(dma_stream_data),.stream_last(dma_stream_last),
        .stream_tag(dma_stream_tag),.record_valid(weight_record_valid),
        .record_ready(weight_record_ready),.record_data(weight_record_data),
        .record_tag(weight_record_tag),.protocol_error(weight_error));

    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;active_output_tile<=0;active_input_tile<=0;
            local_error<=0;done<=0;
        end else begin
            done<=0;
            if(state==IDLE && command_valid) begin
                active_output_tile<=command_output_tile;active_input_tile<=0;
                local_error<=0;state<=META_REQUEST;
            end else if(state==META_REQUEST && dma_request_ready)
                state<=META_RECORD;
            else if(state==META_RECORD) begin
                if(dma_done && (dma_invalid || dma_protocol_error)) begin
                    local_error<=1;state<=IDLE;done<=1;
                end else if(metadata_record_valid && metadata_ready) begin
                    if(metadata_record_tag!=16'h8000) local_error<=1;
                    state<=WEIGHT_REQUEST;
                end
            end else if(state==WEIGHT_REQUEST && dma_request_ready)
                state<=WEIGHT_RECORD;
            else if(state==WEIGHT_RECORD) begin
                if(dma_done && (dma_invalid || dma_protocol_error)) begin
                    local_error<=1;state<=IDLE;done<=1;
                end else if(weight_record_valid && weight_tile_ready) begin
                    if(weight_record_tag[4:0]!=active_input_tile) local_error<=1;
                    if(active_input_tile==23) begin state<=IDLE;done<=1;end
                    else begin
                        active_input_tile<=active_input_tile+1'b1;
                        state<=WEIGHT_REQUEST;
                    end
                end
            end
        end
    end
endmodule
