`timescale 1ns/1ps

module attention_dynamic_vector_scale_tracker #(
    parameter integer DATA_WIDTH = 18,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer VECTOR_LENGTH = 64,
    parameter integer TOKEN_WIDTH = 6,
    parameter integer CHANNEL_WIDTH = 6,
    parameter integer LEVELS = 127,
    parameter integer FRACTION_BITS = 17
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    output wire ready_in,
    input  wire [TOKEN_WIDTH-1:0] token_in,
    input  wire [CHANNEL_WIDTH-1:0] channel_in,
    input  wire signed [DATA_WIDTH-1:0] query_q12_in,
    input  wire signed [DATA_WIDTH-1:0] key_q12_in,
    output reg  scale_valid,
    output reg  [TOKEN_WIDTH-1:0] scale_token,
    output reg  [DATA_WIDTH-1:0] query_maximum,
    output reg  [DATA_WIDTH-1:0] key_maximum,
    output reg  [MULTIPLIER_WIDTH-1:0] query_multiplier_q17,
    output reg  [MULTIPLIER_WIDTH-1:0] key_multiplier_q17
);

    localparam [MULTIPLIER_WIDTH-1:0] DIVIDEND =
        LEVELS << FRACTION_BITS;

    reg [DATA_WIDTH-1:0] query_maximum_work;
    reg [DATA_WIDTH-1:0] key_maximum_work;
    reg [DATA_WIDTH-1:0] pending_query_divisor;
    reg [DATA_WIDTH-1:0] pending_key_divisor;
    reg [TOKEN_WIDTH-1:0] pending_token;

    wire [DATA_WIDTH-1:0] query_magnitude = query_q12_in[DATA_WIDTH-1]
        ? (~query_q12_in + 1'b1) : query_q12_in;
    wire [DATA_WIDTH-1:0] key_magnitude = key_q12_in[DATA_WIDTH-1]
        ? (~key_q12_in + 1'b1) : key_q12_in;
    wire first_channel = channel_in == {CHANNEL_WIDTH{1'b0}};
    wire last_channel = channel_in == VECTOR_LENGTH-1;
    wire [DATA_WIDTH-1:0] query_observed_maximum = first_channel
        ? query_magnitude
        : ((query_magnitude > query_maximum_work)
            ? query_magnitude : query_maximum_work);
    wire [DATA_WIDTH-1:0] key_observed_maximum = first_channel
        ? key_magnitude
        : ((key_magnitude > key_maximum_work)
            ? key_magnitude : key_maximum_work);
    wire [DATA_WIDTH-1:0] query_effective_maximum =
        (query_observed_maximum == 0) ? 1 : query_observed_maximum;
    wire [DATA_WIDTH-1:0] key_effective_maximum =
        (key_observed_maximum == 0) ? 1 : key_observed_maximum;

    wire query_divider_ready;
    wire key_divider_ready;
    wire divider_start = valid_in && ready_in && last_channel;
    wire query_divider_valid;
    wire key_divider_valid;
    wire [MULTIPLIER_WIDTH-1:0] query_quotient;
    wire [MULTIPLIER_WIDTH-1:0] key_quotient;
    wire [MULTIPLIER_WIDTH-1:0] query_remainder;
    wire [MULTIPLIER_WIDTH-1:0] key_remainder;
    wire [MULTIPLIER_WIDTH:0] doubled_query_remainder =
        {query_remainder, 1'b0};
    wire [MULTIPLIER_WIDTH:0] doubled_key_remainder =
        {key_remainder, 1'b0};
    wire [MULTIPLIER_WIDTH:0] extended_query_divisor =
        {{(MULTIPLIER_WIDTH+1-DATA_WIDTH){1'b0}}, pending_query_divisor};
    wire [MULTIPLIER_WIDTH:0] extended_key_divisor =
        {{(MULTIPLIER_WIDTH+1-DATA_WIDTH){1'b0}}, pending_key_divisor};
    wire round_query_up =
        (doubled_query_remainder > extended_query_divisor)
        || ((doubled_query_remainder == extended_query_divisor)
            && query_quotient[0]);
    wire round_key_up =
        (doubled_key_remainder > extended_key_divisor)
        || ((doubled_key_remainder == extended_key_divisor)
            && key_quotient[0]);

    assign ready_in = !last_channel
        || (query_divider_ready && key_divider_ready);

    unsigned_divider_iterative #(
        .WIDTH(MULTIPLIER_WIDTH)
    ) query_divider (
        .clk(clk), .rst_n(rst_n), .start(divider_start),
        .ready(query_divider_ready), .dividend(DIVIDEND),
        .divisor({{(MULTIPLIER_WIDTH-DATA_WIDTH){1'b0}},
                  query_effective_maximum}), .busy(),
        .valid_out(query_divider_valid), .quotient(query_quotient),
        .remainder(query_remainder)
    );

    unsigned_divider_iterative #(
        .WIDTH(MULTIPLIER_WIDTH)
    ) key_divider (
        .clk(clk), .rst_n(rst_n), .start(divider_start),
        .ready(key_divider_ready), .dividend(DIVIDEND),
        .divisor({{(MULTIPLIER_WIDTH-DATA_WIDTH){1'b0}},
                  key_effective_maximum}), .busy(),
        .valid_out(key_divider_valid), .quotient(key_quotient),
        .remainder(key_remainder)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            query_maximum_work <= 0;
            key_maximum_work <= 0;
            pending_query_divisor <= 1;
            pending_key_divisor <= 1;
            pending_token <= 0;
            scale_valid <= 1'b0;
            scale_token <= 0;
            query_maximum <= 0;
            key_maximum <= 0;
            query_multiplier_q17 <= 0;
            key_multiplier_q17 <= 0;
        end else begin
            scale_valid <= 1'b0;
            if (valid_in && ready_in) begin
                query_maximum_work <= query_observed_maximum;
                key_maximum_work <= key_observed_maximum;
                if (last_channel) begin
                    pending_token <= token_in;
                    pending_query_divisor <= query_effective_maximum;
                    pending_key_divisor <= key_effective_maximum;
                    query_maximum_work <= 0;
                    key_maximum_work <= 0;
                end
            end
            if (query_divider_valid && key_divider_valid) begin
                scale_valid <= 1'b1;
                scale_token <= pending_token;
                query_maximum <= pending_query_divisor;
                key_maximum <= pending_key_divisor;
                query_multiplier_q17 <= query_quotient + round_query_up;
                key_multiplier_q17 <= key_quotient + round_key_up;
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && query_divider_valid != key_divider_valid)
            $error("dynamic Q/K scale dividers lost alignment");
        if (rst_n && valid_in && !ready_in)
            $error("dynamic Q/K scale tracker input overflow");
`endif
    end

endmodule
