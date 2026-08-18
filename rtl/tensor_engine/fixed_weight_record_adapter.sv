`timescale 1ns/1ps

module fixed_weight_record_adapter #(
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 8,
    parameter integer STREAM_WIDTH = 512,
    parameter integer TAG_WIDTH = 16,
    parameter integer RECORD_WIDTH = N_LANES * 32 * DATA_WIDTH,
    parameter integer BEATS_PER_RECORD = RECORD_WIDTH / STREAM_WIDTH,
    parameter integer BEAT_COUNTER_WIDTH = (BEATS_PER_RECORD <= 1)
        ? 1 : $clog2(BEATS_PER_RECORD)
) (
    input  wire clk,
    input  wire rst_n,

    input  wire stream_valid,
    output wire stream_ready,
    input  wire [STREAM_WIDTH-1:0] stream_data,
    input  wire stream_last,
    input  wire [TAG_WIDTH-1:0] stream_tag,

    output reg  record_valid,
    input  wire record_ready,
    output wire [RECORD_WIDTH-1:0] record_data,
    output wire [TAG_WIDTH-1:0] record_tag,
    output reg  protocol_error
);

    reg [STREAM_WIDTH-1:0] beat_buffer [0:BEATS_PER_RECORD-1];
    reg [BEAT_COUNTER_WIDTH-1:0] beat_counter;
    reg [TAG_WIDTH-1:0] active_tag;
    wire accepting_beat = stream_valid && stream_ready;
    wire expected_last = beat_counter == BEATS_PER_RECORD-1;

    initial begin
        if (RECORD_WIDTH % STREAM_WIDTH != 0)
            $error("record width must be divisible by stream width");
        if (BEATS_PER_RECORD < 1)
            $error("record must contain at least one stream beat");
    end

    genvar beat_index;
    generate
        for (beat_index = 0; beat_index < BEATS_PER_RECORD; beat_index = beat_index + 1) begin : pack_record
            assign record_data[beat_index*STREAM_WIDTH +: STREAM_WIDTH]
                = beat_buffer[beat_index];
        end
    endgenerate

    assign stream_ready = !record_valid || record_ready;
    assign record_tag = active_tag;

    always @(posedge clk) begin
        if (!rst_n) begin
            beat_counter <= 0;
            active_tag <= 0;
            record_valid <= 1'b0;
            protocol_error <= 1'b0;
        end else begin
            if (record_valid && record_ready)
                record_valid <= 1'b0;

            if (accepting_beat) begin
                beat_buffer[beat_counter] <= stream_data;
                if (beat_counter == 0)
                    active_tag <= stream_tag;
                else if (stream_tag != active_tag)
                    protocol_error <= 1'b1;

                if (stream_last != expected_last) begin
                    beat_counter <= 0;
                    record_valid <= 1'b0;
                    protocol_error <= 1'b1;
                end else if (expected_last) begin
                    beat_counter <= 0;
                    record_valid <= 1'b1;
                end else begin
                    beat_counter <= beat_counter + 1'b1;
                end
            end
        end
    end

endmodule
