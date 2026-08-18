`timescale 1ns/1ps

module unsigned_sqrt_iterative #(
    parameter integer RADICAND_WIDTH = 44,
    parameter integer ROOT_WIDTH = RADICAND_WIDTH / 2,
    parameter integer REMAINDER_WIDTH = ROOT_WIDTH + 2,
    parameter integer COUNT_WIDTH = (ROOT_WIDTH <= 1)
        ? 1 : $clog2(ROOT_WIDTH + 1)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output wire ready,
    input  wire [RADICAND_WIDTH-1:0] radicand,
    output reg  busy,
    output reg  valid_out,
    output reg  [ROOT_WIDTH-1:0] root,
    output reg  [REMAINDER_WIDTH-1:0] remainder
);

    reg [RADICAND_WIDTH-1:0] radicand_shift;
    reg [ROOT_WIDTH-1:0] root_work;
    reg [REMAINDER_WIDTH-1:0] remainder_work;
    reg [COUNT_WIDTH-1:0] iterations_left;
    reg [REMAINDER_WIDTH-1:0] shifted_remainder;
    reg [REMAINDER_WIDTH-1:0] trial_subtractor;
    reg [REMAINDER_WIDTH-1:0] next_remainder;
    reg [ROOT_WIDTH-1:0] next_root;

    assign ready = !busy;

    initial begin
        if (RADICAND_WIDTH % 2 != 0) begin
            $error("RADICAND_WIDTH must be even");
        end
    end

    always @* begin
        shifted_remainder = {
            remainder_work[REMAINDER_WIDTH-3:0],
            radicand_shift[RADICAND_WIDTH-1 -: 2]
        };
        trial_subtractor = {
            {(REMAINDER_WIDTH-ROOT_WIDTH-2){1'b0}},
            root_work,
            2'b01
        };
        if (shifted_remainder >= trial_subtractor) begin
            next_remainder = shifted_remainder - trial_subtractor;
            next_root = {root_work[ROOT_WIDTH-2:0], 1'b1};
        end else begin
            next_remainder = shifted_remainder;
            next_root = {root_work[ROOT_WIDTH-2:0], 1'b0};
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0;
            valid_out <= 1'b0;
            root <= {ROOT_WIDTH{1'b0}};
            remainder <= {REMAINDER_WIDTH{1'b0}};
            radicand_shift <= {RADICAND_WIDTH{1'b0}};
            root_work <= {ROOT_WIDTH{1'b0}};
            remainder_work <= {REMAINDER_WIDTH{1'b0}};
            iterations_left <= {COUNT_WIDTH{1'b0}};
        end else begin
            valid_out <= 1'b0;
            if (start && ready) begin
                busy <= 1'b1;
                radicand_shift <= radicand;
                root_work <= {ROOT_WIDTH{1'b0}};
                remainder_work <= {REMAINDER_WIDTH{1'b0}};
                iterations_left <= ROOT_WIDTH;
            end else if (busy) begin
                radicand_shift <= {
                    radicand_shift[RADICAND_WIDTH-3:0], 2'b00
                };
                root_work <= next_root;
                remainder_work <= next_remainder;
                iterations_left <= iterations_left - 1'b1;
                if (iterations_left == 1) begin
                    busy <= 1'b0;
                    root <= next_root;
                    remainder <= next_remainder;
                    valid_out <= 1'b1;
                end
            end
        end
    end

endmodule
