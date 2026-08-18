`timescale 1ns/1ps

module mdlm_block_parameter_address_generator (
    input  wire [63:0] block_base_address,
    input  wire [3:0] section_id,
    input  wire [13:0] record_index,
    output reg  record_valid,
    output reg  [63:0] axi_address,
    output reg  [6:0] burst_beats,
    output reg  [5:0] record_byte_offset,
    output reg  [9:0] record_payload_bytes
);

    localparam [3:0] SECTION_QKV_METADATA = 4'd0;
    localparam [3:0] SECTION_QKV_WEIGHTS = 4'd1;
    localparam [3:0] SECTION_ROTARY_CONSTANTS = 4'd2;
    localparam [3:0] SECTION_PROJECTION_METADATA = 4'd3;
    localparam [3:0] SECTION_PROJECTION_WEIGHTS = 4'd4;
    localparam [3:0] SECTION_MLP_RECIPROCAL = 4'd5;
    localparam [3:0] SECTION_MLP_UP_METADATA = 4'd6;
    localparam [3:0] SECTION_MLP_UP_WEIGHTS = 4'd7;
    localparam [3:0] SECTION_MLP_DOWN_METADATA = 4'd8;
    localparam [3:0] SECTION_MLP_DOWN_WEIGHTS = 4'd9;

    reg [31:0] section_offset;
    reg [13:0] record_count;
    reg [31:0] record_offset;
    reg compact_four_byte_records;

    always @* begin
        section_offset = 0;
        record_count = 0;
        record_offset = 0;
        compact_four_byte_records = 1'b0;
        burst_beats = 0;
        record_byte_offset = 0;
        record_payload_bytes = 0;
        case (section_id)
            SECTION_QKV_METADATA: begin
                section_offset = 32'd0;
                record_count = 14'd396;
                record_offset = {18'd0, record_index} << 6;
                burst_beats = 7'd1;
                record_payload_bytes = 10'd32;
            end
            SECTION_QKV_WEIGHTS: begin
                section_offset = 32'd28672;
                record_count = 14'd9504;
                record_offset = ({18'd0, record_index} << 8)
                    + ({18'd0, record_index} << 7);
                burst_beats = 7'd6;
                record_payload_bytes = 10'd384;
            end
            SECTION_ROTARY_CONSTANTS: begin
                section_offset = 32'd3678208;
                record_count = 14'd2048;
                record_offset = {22'd0, record_index[13:4]} << 6;
                compact_four_byte_records = 1'b1;
                burst_beats = 7'd1;
                record_payload_bytes = 10'd4;
            end
            SECTION_PROJECTION_METADATA: begin
                section_offset = 32'd3686400;
                record_count = 14'd128;
                record_offset = {18'd0, record_index} << 6;
                burst_beats = 7'd1;
                record_payload_bytes = 10'd18;
            end
            SECTION_PROJECTION_WEIGHTS: begin
                section_offset = 32'd3694592;
                record_count = 14'd3072;
                record_offset = ({18'd0, record_index} << 7)
                    + ({18'd0, record_index} << 6);
                burst_beats = 7'd3;
                record_payload_bytes = 10'd192;
            end
            SECTION_MLP_RECIPROCAL: begin
                section_offset = 32'd4284416;
                record_count = 14'd768;
                record_offset = {22'd0, record_index[13:4]} << 6;
                compact_four_byte_records = 1'b1;
                burst_beats = 7'd1;
                record_payload_bytes = 10'd3;
            end
            SECTION_MLP_UP_METADATA: begin
                section_offset = 32'd4288512;
                record_count = 14'd512;
                record_offset = {18'd0, record_index} << 6;
                burst_beats = 7'd1;
                record_payload_bytes = 10'd56;
            end
            SECTION_MLP_UP_WEIGHTS: begin
                section_offset = 32'd4321280;
                record_count = 14'd12288;
                record_offset = ({18'd0, record_index} << 7)
                    + ({18'd0, record_index} << 6);
                burst_beats = 7'd3;
                record_payload_bytes = 10'd192;
            end
            SECTION_MLP_DOWN_METADATA: begin
                section_offset = 32'd6680576;
                record_count = 14'd128;
                record_offset = ({18'd0, record_index} << 7)
                    + ({18'd0, record_index} << 6);
                burst_beats = 7'd3;
                record_payload_bytes = 10'd168;
            end
            SECTION_MLP_DOWN_WEIGHTS: begin
                section_offset = 32'd6705152;
                record_count = 14'd12288;
                record_offset = ({18'd0, record_index} << 7)
                    + ({18'd0, record_index} << 6);
                burst_beats = 7'd3;
                record_payload_bytes = 10'd192;
            end
            default: begin
                section_offset = 0;
                record_count = 0;
                record_offset = 0;
            end
        endcase
        record_valid = record_index < record_count;
        axi_address = block_base_address + section_offset + record_offset;
        if (compact_four_byte_records)
            record_byte_offset = {record_index[3:0], 2'b00};
        if (!record_valid) begin
            burst_beats = 0;
            record_byte_offset = 0;
            record_payload_bytes = 0;
        end
    end

endmodule
