`timescale 1ns/1ps

module mdlm_mlp_output_tile_loader #(
    parameter integer METADATA_SECTION_ID = 6,
    parameter integer WEIGHT_SECTION_ID = 7,
    parameter integer INPUT_SIZE = 768,
    parameter integer METADATA_WIDTH = 444,
    parameter integer METADATA_PAYLOAD_BYTES = 56,
    parameter integer OUTPUT_TILE_WIDTH = 10,
    parameter integer K_TILES = INPUT_SIZE / 32,
    parameter integer K_TILE_WIDTH = (K_TILES <= 1) ? 1 : $clog2(K_TILES)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire [63:0] block_base_address,
    input  wire command_valid,
    output wire command_ready,
    input  wire [OUTPUT_TILE_WIDTH-1:0] command_output_tile,
    output wire metadata_stream_valid,
    input  wire metadata_stream_ready,
    output wire [METADATA_WIDTH-1:0] metadata_stream_data,
    output wire weight_stream_valid,
    input  wire weight_stream_ready,
    output wire [6*32*8-1:0] weight_stream_data,
    output wire [K_TILE_WIDTH-1:0] weight_stream_k_tile,
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
    reg [OUTPUT_TILE_WIDTH-1:0] active_output_tile;
    reg [K_TILE_WIDTH-1:0] active_k_tile;
    reg local_error;
    wire [13:0] output_tile_extended={{(14-OUTPUT_TILE_WIDTH){1'b0}},
        active_output_tile};
    wire [13:0] weight_record_index;
    generate
        if(K_TILES==24) begin : stride_24
            assign weight_record_index=(output_tile_extended<<4)
                +(output_tile_extended<<3)+active_k_tile;
        end else if(K_TILES==96) begin : stride_96
            assign weight_record_index=(output_tile_extended<<6)
                +(output_tile_extended<<5)+active_k_tile;
        end else begin : generic_stride
            assign weight_record_index=output_tile_extended*K_TILES+active_k_tile;
        end
    endgenerate
    wire dma_request_ready,dma_stream_valid,dma_stream_last,dma_done;
    wire dma_invalid,dma_protocol_error;
    wire [511:0] dma_stream_data;wire [15:0] dma_stream_tag;
    wire [5:0] dma_byte_offset;wire [9:0] dma_payload_bytes;
    wire meta_valid,weight_valid,meta_error,weight_error;
    wire [METADATA_WIDTH-1:0] meta_data;wire [1535:0] weight_data;
    wire [15:0] meta_tag,weight_tag;
    wire receiving_meta=state==META_RECORD;
    wire receiving_weight=state==WEIGHT_RECORD;
    wire meta_ready=receiving_meta && metadata_stream_ready;
    wire weight_ready=receiving_weight && weight_stream_ready;
    wire dma_stream_ready=receiving_meta ? meta_ready || !meta_valid
        : receiving_weight ? weight_ready || !weight_valid : 1'b0;

    assign command_ready=state==IDLE;
    assign busy=state!=IDLE;
    assign protocol_error=local_error || dma_protocol_error
        || meta_error || weight_error;
    assign metadata_stream_valid=receiving_meta && meta_valid;
    assign metadata_stream_data=meta_data;
    assign weight_stream_valid=receiving_weight && weight_valid;
    assign weight_stream_data=weight_data;
    assign weight_stream_k_tile=active_k_tile;

    mdlm_block_parameter_dma parameter_dma(
        .clk(clk),.rst_n(rst_n),.block_base_address(block_base_address),
        .request_valid(state==META_REQUEST || state==WEIGHT_REQUEST),
        .request_ready(dma_request_ready),
        .request_section_id(state==META_REQUEST
            ? METADATA_SECTION_ID[3:0] : WEIGHT_SECTION_ID[3:0]),
        .request_record_index(state==META_REQUEST
            ? active_output_tile : weight_record_index),
        .request_tag(state==META_REQUEST ? 16'h8000
            : {{(16-K_TILE_WIDTH){1'b0}},active_k_tile}),
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
    fixed_aligned_record_adapter #(.RECORD_WIDTH(METADATA_WIDTH),
        .EXPECTED_PAYLOAD_BYTES(METADATA_PAYLOAD_BYTES)) metadata_adapter(
        .clk(clk),.rst_n(rst_n),
        .stream_valid(dma_stream_valid && receiving_meta),.stream_ready(),
        .stream_data(dma_stream_data),.stream_last(dma_stream_last),
        .stream_tag(dma_stream_tag),.stream_record_byte_offset(dma_byte_offset),
        .stream_record_payload_bytes(dma_payload_bytes),.record_valid(meta_valid),
        .record_ready(meta_ready),.record_data(meta_data),.record_tag(meta_tag),
        .protocol_error(meta_error));
    fixed_weight_record_adapter #(.DATA_WIDTH(8)) weight_adapter(
        .clk(clk),.rst_n(rst_n),
        .stream_valid(dma_stream_valid && receiving_weight),.stream_ready(),
        .stream_data(dma_stream_data),.stream_last(dma_stream_last),
        .stream_tag(dma_stream_tag),.record_valid(weight_valid),
        .record_ready(weight_ready),.record_data(weight_data),
        .record_tag(weight_tag),.protocol_error(weight_error));

    always @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;active_output_tile<=0;active_k_tile<=0;
            local_error<=0;done<=0;
        end else begin
            done<=0;
            if(state==IDLE && command_valid) begin
                active_output_tile<=command_output_tile;active_k_tile<=0;
                local_error<=0;state<=META_REQUEST;
            end else if(state==META_REQUEST && dma_request_ready)
                state<=META_RECORD;
            else if(state==META_RECORD) begin
                if(dma_done && (dma_invalid || dma_protocol_error)) begin
                    local_error<=1;state<=IDLE;done<=1;
                end else if(meta_valid && metadata_stream_ready) begin
                    if(meta_tag!=16'h8000) local_error<=1;
                    state<=WEIGHT_REQUEST;
                end
            end else if(state==WEIGHT_REQUEST && dma_request_ready)
                state<=WEIGHT_RECORD;
            else if(state==WEIGHT_RECORD) begin
                if(dma_done && (dma_invalid || dma_protocol_error)) begin
                    local_error<=1;state<=IDLE;done<=1;
                end else if(weight_valid && weight_stream_ready) begin
                    if(weight_tag[K_TILE_WIDTH-1:0]!=active_k_tile) local_error<=1;
                    if(active_k_tile==K_TILES-1) begin state<=IDLE;done<=1;end
                    else begin active_k_tile<=active_k_tile+1'b1;
                        state<=WEIGHT_REQUEST;end
                end
            end
        end
    end

    initial begin
        if(INPUT_SIZE<32 || INPUT_SIZE%32!=0)
            $error("MLP loader input size must be divisible by 32");
        if(K_TILE_WIDTH>16) $error("MLP loader tag is too narrow");
    end
endmodule
