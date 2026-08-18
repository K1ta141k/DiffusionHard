`timescale 1ns/1ps

module compact_table_record_adapter #(
    parameter integer RECORD_WIDTH = 18,
    parameter integer EXPECTED_PAYLOAD_BYTES = (RECORD_WIDTH + 7) / 8,
    parameter integer TAG_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,
    input  wire stream_valid,
    output wire stream_ready,
    input  wire [511:0] stream_data,
    input  wire stream_last,
    input  wire [TAG_WIDTH-1:0] stream_tag,
    input  wire [5:0] stream_record_byte_offset,
    input  wire [9:0] stream_record_payload_bytes,
    output reg  record_valid,
    input  wire record_ready,
    output reg  [RECORD_WIDTH-1:0] record_data,
    output reg  [TAG_WIDTH-1:0] record_tag,
    output reg  protocol_error
);

    reg [31:0] selected_word;
    wire accepting = stream_valid && stream_ready;
    wire geometry_error = !stream_last
        || stream_record_byte_offset[1:0] != 0
        || stream_record_byte_offset > 60
        || stream_record_payload_bytes != EXPECTED_PAYLOAD_BYTES;

    assign stream_ready = !record_valid || record_ready;

    always @* begin
        case (stream_record_byte_offset[5:2])
            4'd0: selected_word = stream_data[31:0];
            4'd1: selected_word = stream_data[63:32];
            4'd2: selected_word = stream_data[95:64];
            4'd3: selected_word = stream_data[127:96];
            4'd4: selected_word = stream_data[159:128];
            4'd5: selected_word = stream_data[191:160];
            4'd6: selected_word = stream_data[223:192];
            4'd7: selected_word = stream_data[255:224];
            4'd8: selected_word = stream_data[287:256];
            4'd9: selected_word = stream_data[319:288];
            4'd10: selected_word = stream_data[351:320];
            4'd11: selected_word = stream_data[383:352];
            4'd12: selected_word = stream_data[415:384];
            4'd13: selected_word = stream_data[447:416];
            4'd14: selected_word = stream_data[479:448];
            default: selected_word = stream_data[511:480];
        endcase
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            record_valid <= 1'b0;
            record_data <= 0;
            record_tag <= 0;
            protocol_error <= 1'b0;
        end else begin
            if (record_valid && record_ready)
                record_valid <= 1'b0;
            if (accepting) begin
                if (geometry_error) begin
                    record_valid <= 1'b0;
                    protocol_error <= 1'b1;
                end else begin
                    record_data <= selected_word[RECORD_WIDTH-1:0];
                    record_tag <= stream_tag;
                    record_valid <= 1'b1;
                end
            end
        end
    end

    initial begin
        if (RECORD_WIDTH < 1 || RECORD_WIDTH > 32)
            $error("compact record width must fit one four-byte slot");
        if (EXPECTED_PAYLOAD_BYTES > 4)
            $error("compact record payload must fit one four-byte slot");
    end

endmodule
