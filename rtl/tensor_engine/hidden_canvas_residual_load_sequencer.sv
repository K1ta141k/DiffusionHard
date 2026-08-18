`timescale 1ns/1ps

module hidden_canvas_residual_load_sequencer #(
    parameter integer TOKEN_GROUPS = 16,
    parameter integer OUTPUT_TILE_WIDTH = 7,
    parameter integer GROUP_WIDTH = (TOKEN_GROUPS <= 1)
        ? 1 : $clog2(TOKEN_GROUPS),
    parameter integer DATA_WIDTH = 4*6*24
) (
    input  wire clk,
    input  wire rst_n,
    input  wire command_valid,
    input  wire [OUTPUT_TILE_WIDTH-1:0] command_output_tile,
    output wire command_ready,

    output wire canvas_read_valid,
    output wire [GROUP_WIDTH-1:0] canvas_read_group,
    output wire [OUTPUT_TILE_WIDTH-1:0] canvas_read_output_tile,
    input  wire canvas_read_data_valid,
    input  wire [DATA_WIDTH-1:0] canvas_read_data,

    output wire residual_load_valid,
    output wire [GROUP_WIDTH-1:0] residual_load_group,
    output wire [OUTPUT_TILE_WIDTH-1:0] residual_load_output_tile,
    output wire [DATA_WIDTH-1:0] residual_load_data,
    input  wire residual_load_ready,

    output reg  busy,
    output reg  done
);

    reg [OUTPUT_TILE_WIDTH-1:0] active_output_tile;
    reg [GROUP_WIDTH-1:0] active_group;
    reg read_pending;
    reg response_pending;
    reg [DATA_WIDTH-1:0] response_data;
    wire residual_accept = residual_load_valid && residual_load_ready;

    assign command_ready = !busy && !done;
    assign canvas_read_valid = busy && !read_pending && !response_pending;
    assign canvas_read_group = active_group;
    assign canvas_read_output_tile = active_output_tile;
    assign residual_load_valid = busy && response_pending;
    assign residual_load_group = active_group;
    assign residual_load_output_tile = active_output_tile;
    assign residual_load_data = response_data;

    always @(posedge clk) begin
        if (!rst_n) begin
            active_output_tile <= {OUTPUT_TILE_WIDTH{1'b0}};
            active_group <= {GROUP_WIDTH{1'b0}};
            read_pending <= 1'b0;
            response_pending <= 1'b0;
            response_data <= {DATA_WIDTH{1'b0}};
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (command_valid && command_ready) begin
                active_output_tile <= command_output_tile;
                active_group <= {GROUP_WIDTH{1'b0}};
                read_pending <= 1'b0;
                response_pending <= 1'b0;
                busy <= 1'b1;
            end else if (busy) begin
                if (canvas_read_valid)
                    read_pending <= 1'b1;
                if (canvas_read_data_valid) begin
                    response_data <= canvas_read_data;
                    read_pending <= 1'b0;
                    response_pending <= 1'b1;
                end
                if (residual_accept) begin
                    response_pending <= 1'b0;
                    if (active_group == TOKEN_GROUPS-1) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                    end else begin
                        active_group <= active_group + 1'b1;
                    end
                end
            end
        end
    end

    always @(posedge clk) begin
`ifndef SYNTHESIS
        if (rst_n && canvas_read_data_valid && !read_pending)
            $error("hidden canvas residual loader received an unexpected response");
`endif
    end

    initial begin
        if (TOKEN_GROUPS < 1)
            $error("hidden canvas residual loader requires token groups");
    end

endmodule
