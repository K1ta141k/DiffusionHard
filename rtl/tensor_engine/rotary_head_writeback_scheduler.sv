`timescale 1ns/1ps

module rotary_head_writeback_scheduler (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output wire start_ready,
    input  wire [3:0] head_in,
    output wire qk_read_valid,
    output wire [5:0] qk_read_token,
    output wire [4:0] qk_read_pair,
    input  wire qk_read_data_valid,
    input  wire [5:0] qk_read_token_out,
    input  wire [4:0] qk_read_pair_out,
    input  wire signed [17:0] query_first_q12,
    input  wire signed [17:0] query_second_q12,
    input  wire signed [17:0] key_first_q12,
    input  wire signed [17:0] key_second_q12,
    output wire constant_read_valid,
    output wire [5:0] constant_read_token,
    output wire [4:0] constant_read_pair,
    input  wire constant_read_data_valid,
    input  wire [5:0] constant_read_token_out,
    input  wire [4:0] constant_read_pair_out,
    input  wire signed [15:0] cosine_q15,
    input  wire signed [15:0] sine_q15,
    output wire query_write_valid,
    output wire key_write_valid,
    output wire [5:0] write_token,
    output wire [5:0] write_channel,
    output wire signed [17:0] write_query_q12,
    output wire signed [17:0] write_key_q12,
    output wire busy,
    output reg  done
);

    reg active;
    reg request_gap;
    reg [11:0] requests_issued;
    reg [5:0] next_token;
    reg [4:0] next_pair;
    reg [3:0] active_head;
    reg second_pending;
    reg [5:0] pending_token;
    reg [4:0] pending_pair;
    reg signed [17:0] pending_query_second;
    reg signed [17:0] pending_key_second;
    reg pending_last;

    wire issue_request = active && !request_gap && requests_issued < 2048;
    wire rotary_input_valid = qk_read_data_valid && constant_read_data_valid;
    wire rotary_output_valid;
    wire [3:0] rotary_group;
    wire [1:0] rotary_token_lane;
    wire [3:0] rotary_head;
    wire [4:0] rotary_pair;
    wire signed [17:0] rotated_query_first;
    wire signed [17:0] rotated_query_second;
    wire signed [17:0] rotated_key_first;
    wire signed [17:0] rotated_key_second;
    wire [5:0] rotary_token = {rotary_group, rotary_token_lane};

    assign start_ready = !active && !second_pending;
    assign busy = active || second_pending;
    assign qk_read_valid = issue_request;
    assign qk_read_token = next_token;
    assign qk_read_pair = next_pair;
    assign constant_read_valid = issue_request;
    assign constant_read_token = next_token;
    assign constant_read_pair = next_pair;
    assign query_write_valid = rotary_output_valid || second_pending;
    assign key_write_valid = rotary_output_valid || second_pending;
    assign write_token = rotary_output_valid ? rotary_token : pending_token;
    assign write_channel = rotary_output_valid
        ? {1'b0, rotary_pair} : {1'b1, pending_pair};
    assign write_query_q12 = rotary_output_valid
        ? rotated_query_first : pending_query_second;
    assign write_key_q12 = rotary_output_valid
        ? rotated_key_first : pending_key_second;

    rotary_qk_pair_serial rotary (
        .clk(clk), .rst_n(rst_n), .valid_in(rotary_input_valid), .ready_in(),
        .group_in(qk_read_token_out[5:2]),
        .token_in(qk_read_token_out[1:0]), .head_in(active_head),
        .pair_in(qk_read_pair_out), .query_first_q12(query_first_q12),
        .query_second_q12(query_second_q12), .key_first_q12(key_first_q12),
        .key_second_q12(key_second_q12), .cosine_q15(cosine_q15),
        .sine_q15(sine_q15), .valid_out(rotary_output_valid),
        .group_out(rotary_group), .token_out(rotary_token_lane),
        .head_out(rotary_head), .pair_out(rotary_pair),
        .query_first_rotated_q12(rotated_query_first),
        .query_second_rotated_q12(rotated_query_second),
        .key_first_rotated_q12(rotated_key_first),
        .key_second_rotated_q12(rotated_key_second)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            active <= 1'b0;
            request_gap <= 1'b0;
            requests_issued <= 0;
            next_token <= 0;
            next_pair <= 0;
            active_head <= 0;
            second_pending <= 1'b0;
            pending_token <= 0;
            pending_pair <= 0;
            pending_query_second <= 0;
            pending_key_second <= 0;
            pending_last <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start && start_ready) begin
                active <= 1'b1;
                request_gap <= 1'b0;
                requests_issued <= 0;
                next_token <= 0;
                next_pair <= 0;
                active_head <= head_in;
            end
            if (issue_request) begin
                requests_issued <= requests_issued + 1'b1;
                request_gap <= 1'b1;
                if (next_pair == 31) begin
                    next_pair <= 0;
                    next_token <= next_token + 1'b1;
                end else begin
                    next_pair <= next_pair + 1'b1;
                end
            end else if (request_gap) begin
                request_gap <= 1'b0;
            end
            if (rotary_output_valid) begin
                pending_token <= rotary_token;
                pending_pair <= rotary_pair;
                pending_query_second <= rotated_query_second;
                pending_key_second <= rotated_key_second;
                pending_last <= rotary_token == 63 && rotary_pair == 31;
                second_pending <= 1'b1;
            end else if (second_pending) begin
                second_pending <= 1'b0;
                if (pending_last) begin
                    active <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && qk_read_data_valid != constant_read_data_valid)
            $error("rotary QK and constant responses lost alignment");
        if (rst_n && rotary_input_valid
            && (qk_read_token_out != constant_read_token_out
                || qk_read_pair_out != constant_read_pair_out))
            $error("rotary response tags did not match");
        if (rst_n && rotary_output_valid && second_pending)
            $error("rotary writeback serializer overflow");
`endif
    end

endmodule
