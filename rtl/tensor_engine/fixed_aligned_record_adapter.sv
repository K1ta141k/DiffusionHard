`timescale 1ns/1ps

module fixed_aligned_record_adapter #(
    parameter integer RECORD_WIDTH = 252,
    parameter integer EXPECTED_PAYLOAD_BYTES = (RECORD_WIDTH + 7) / 8,
    parameter integer STREAM_WIDTH = 512,
    parameter integer TAG_WIDTH = 16,
    parameter integer BEATS_PER_RECORD =
        (EXPECTED_PAYLOAD_BYTES + STREAM_WIDTH/8 - 1) / (STREAM_WIDTH/8),
    parameter integer BEAT_COUNTER_WIDTH = (BEATS_PER_RECORD <= 1)
        ? 1 : $clog2(BEATS_PER_RECORD),
    parameter integer PADDED_WIDTH = BEATS_PER_RECORD * STREAM_WIDTH
) (
    input  wire clk,
    input  wire rst_n,
    input  wire stream_valid,
    output wire stream_ready,
    input  wire [STREAM_WIDTH-1:0] stream_data,
    input  wire stream_last,
    input  wire [TAG_WIDTH-1:0] stream_tag,
    input  wire [5:0] stream_record_byte_offset,
    input  wire [9:0] stream_record_payload_bytes,
    output reg  record_valid,
    input  wire record_ready,
    output wire [RECORD_WIDTH-1:0] record_data,
    output wire [TAG_WIDTH-1:0] record_tag,
    output reg  protocol_error
);

    reg [STREAM_WIDTH-1:0] beat_buffer [0:BEATS_PER_RECORD-1];
    reg [BEAT_COUNTER_WIDTH-1:0] beat_counter;
    reg [TAG_WIDTH-1:0] active_tag;
    reg current_record_error;
    wire accepting = stream_valid && stream_ready;
    wire expected_last = beat_counter == BEATS_PER_RECORD-1;
    wire beat_geometry_error = stream_record_byte_offset != 0
        || stream_record_payload_bytes != EXPECTED_PAYLOAD_BYTES;
    wire beat_tag_error = beat_counter != 0 && stream_tag != active_tag;
    wire beat_error = beat_geometry_error || beat_tag_error
        || stream_last != expected_last;
    wire [PADDED_WIDTH-1:0] padded_record;

    genvar beat_index;
    generate
        for (beat_index = 0; beat_index < BEATS_PER_RECORD; beat_index = beat_index + 1) begin : pack_record
            assign padded_record[beat_index*STREAM_WIDTH +: STREAM_WIDTH]
                = beat_buffer[beat_index];
        end
    endgenerate

    assign stream_ready = !record_valid || record_ready;
    assign record_data = padded_record[RECORD_WIDTH-1:0];
    assign record_tag = active_tag;

    always @(posedge clk) begin
        if (!rst_n) begin
            beat_counter <= 0;
            active_tag <= 0;
            current_record_error <= 1'b0;
            record_valid <= 1'b0;
            protocol_error <= 1'b0;
        end else begin
            if (record_valid && record_ready)
                record_valid <= 1'b0;
            if (accepting) begin
                beat_buffer[beat_counter] <= stream_data;
                if (beat_counter == 0)
                    active_tag <= stream_tag;
                if (beat_error || current_record_error)
                    protocol_error <= 1'b1;
                if (stream_last || expected_last) begin
                    record_valid <= !(beat_error || current_record_error);
                    beat_counter <= 0;
                    current_record_error <= 1'b0;
                end else begin
                    beat_counter <= beat_counter + 1'b1;
                    current_record_error <= current_record_error || beat_error;
                end
            end
        end
    end

    initial begin
        if (RECORD_WIDTH < 1 || RECORD_WIDTH > PADDED_WIDTH)
            $error("aligned record width exceeds buffered beats");
        if (EXPECTED_PAYLOAD_BYTES < (RECORD_WIDTH + 7) / 8)
            $error("aligned record payload is too small");
    end

endmodule
