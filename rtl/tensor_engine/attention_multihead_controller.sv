`timescale 1ns/1ps

module attention_multihead_controller #(
    parameter integer HEADS = 12,
    parameter integer LOADS_PER_HEAD = 4096
) (
    input  wire clk,
    input  wire rst_n,
    input  wire block_start,
    output wire block_start_ready,
    input  wire load_fire,
    output wire load_enable,
    output wire [3:0] expected_head,
    input  wire head_start_ready,
    output wire head_start,
    input  wire head_done,
    input  wire canvas_idle,
    output wire busy,
    output reg  done
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_LOAD = 2'd1;
    localparam [1:0] STATE_RUN = 2'd2;
    localparam [1:0] STATE_DRAIN = 2'd3;
    localparam integer LOAD_COUNT_WIDTH = $clog2(LOADS_PER_HEAD);

    reg [1:0] state;
    reg [3:0] active_head;
    reg [LOAD_COUNT_WIDTH-1:0] load_count;
    reg start_pending;

    assign block_start_ready = (state == STATE_IDLE);
    assign load_enable = (state == STATE_LOAD);
    assign expected_head = active_head;
    assign head_start = (state == STATE_RUN) && start_pending
        && head_start_ready;
    assign busy = (state != STATE_IDLE);

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            active_head <= 0;
            load_count <= 0;
            start_pending <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (state == STATE_IDLE && block_start) begin
                state <= STATE_LOAD;
                active_head <= 0;
                load_count <= 0;
            end else if (state == STATE_LOAD && load_fire) begin
                if (load_count == LOADS_PER_HEAD-1) begin
                    state <= STATE_RUN;
                    load_count <= 0;
                    start_pending <= 1'b1;
                end else begin
                    load_count <= load_count + 1'b1;
                end
            end else if (state == STATE_RUN) begin
                if (head_start)
                    start_pending <= 1'b0;
                if (head_done)
                    state <= STATE_DRAIN;
            end else if (state == STATE_DRAIN && canvas_idle) begin
                if (active_head == HEADS-1) begin
                    state <= STATE_IDLE;
                    done <= 1'b1;
                end else begin
                    active_head <= active_head + 1'b1;
                    state <= STATE_LOAD;
                end
            end
        end
    end

    initial begin
        if (HEADS < 1 || HEADS > 12)
            $error("multihead controller supports one through twelve heads");
        if (LOADS_PER_HEAD < 2)
            $error("LOADS_PER_HEAD must be at least two");
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && block_start && !block_start_ready)
            $error("attention block start arrived while busy");
`endif
    end

endmodule
