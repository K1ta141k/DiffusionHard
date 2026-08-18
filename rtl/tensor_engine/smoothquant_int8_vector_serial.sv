`timescale 1ns/1ps

module smoothquant_int8_vector_serial #(
    parameter integer LANES = 24,
    parameter integer INPUT_WIDTH = 16,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer RIGHT_SHIFT = 20,
    parameter integer TAG_WIDTH = 16,
    parameter integer LANE_WIDTH = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter integer PRODUCT_WIDTH = INPUT_WIDTH + MULTIPLIER_WIDTH + 1
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    output wire ready_in,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [LANES*INPUT_WIDTH-1:0] inputs_packed,
    input  wire [LANES*MULTIPLIER_WIDTH-1:0] multipliers_packed,
    output reg  valid_out,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output reg  [LANES*8-1:0] outputs_packed
);

    localparam signed [PRODUCT_WIDTH-1:0] ROUNDING_OFFSET =
        {{(PRODUCT_WIDTH-RIGHT_SHIFT){1'b0}}, 1'b1,
         {(RIGHT_SHIFT-1){1'b0}}};
    localparam signed [PRODUCT_WIDTH-1:0] INT8_MAX = 127;
    localparam signed [PRODUCT_WIDTH-1:0] INT8_MIN = -127;

    reg active;
    reg [LANE_WIDTH-1:0] lane_counter;
    reg [TAG_WIDTH-1:0] working_tag;
    reg [LANES*INPUT_WIDTH-1:0] working_inputs;
    reg [LANES*MULTIPLIER_WIDTH-1:0] working_multipliers;

    reg pending_valid;
    reg [TAG_WIDTH-1:0] pending_tag;
    reg [LANES*INPUT_WIDTH-1:0] pending_inputs;
    reg [LANES*MULTIPLIER_WIDTH-1:0] pending_multipliers;

    wire signed [INPUT_WIDTH-1:0] selected_input = working_inputs[
        lane_counter*INPUT_WIDTH +: INPUT_WIDTH
    ];
    wire [MULTIPLIER_WIDTH-1:0] selected_multiplier = working_multipliers[
        lane_counter*MULTIPLIER_WIDTH +: MULTIPLIER_WIDTH
    ];
    reg signed [PRODUCT_WIDTH-1:0] product_next;
    reg signed [PRODUCT_WIDTH-1:0] rounded_next;
    reg signed [7:0] output_next;

    assign ready_in = !pending_valid;

    initial begin
        if (RIGHT_SHIFT <= 0 || RIGHT_SHIFT >= PRODUCT_WIDTH) begin
            $error("RIGHT_SHIFT must be between zero and PRODUCT_WIDTH");
        end
    end

    always @* begin
        product_next = $signed(selected_input) * $signed({
            1'b0, selected_multiplier
        });
        if (product_next >= 0) begin
            rounded_next =
                (product_next + ROUNDING_OFFSET) >>> RIGHT_SHIFT;
        end else begin
            rounded_next = -(
                ((-product_next) + ROUNDING_OFFSET) >>> RIGHT_SHIFT
            );
        end
        if (rounded_next > INT8_MAX) begin
            output_next = 8'sd127;
        end else if (rounded_next < INT8_MIN) begin
            output_next = -8'sd127;
        end else begin
            output_next = rounded_next[7:0];
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            active <= 1'b0;
            lane_counter <= {LANE_WIDTH{1'b0}};
            working_tag <= {TAG_WIDTH{1'b0}};
            working_inputs <= {LANES*INPUT_WIDTH{1'b0}};
            working_multipliers <= {LANES*MULTIPLIER_WIDTH{1'b0}};
            pending_valid <= 1'b0;
            pending_tag <= {TAG_WIDTH{1'b0}};
            pending_inputs <= {LANES*INPUT_WIDTH{1'b0}};
            pending_multipliers <= {LANES*MULTIPLIER_WIDTH{1'b0}};
            valid_out <= 1'b0;
            tag_out <= {TAG_WIDTH{1'b0}};
            outputs_packed <= {LANES*8{1'b0}};
        end else begin
            valid_out <= 1'b0;

            if (!active) begin
                if (pending_valid) begin
                    active <= 1'b1;
                    lane_counter <= {LANE_WIDTH{1'b0}};
                    working_tag <= pending_tag;
                    working_inputs <= pending_inputs;
                    working_multipliers <= pending_multipliers;
                    pending_valid <= 1'b0;
                end else if (valid_in) begin
                    active <= 1'b1;
                    lane_counter <= {LANE_WIDTH{1'b0}};
                    working_tag <= tag_in;
                    working_inputs <= inputs_packed;
                    working_multipliers <= multipliers_packed;
                end
            end else begin
                outputs_packed[lane_counter*8 +: 8] <= output_next;
                if (lane_counter == LANES-1) begin
                    valid_out <= 1'b1;
                    tag_out <= working_tag;
                    lane_counter <= {LANE_WIDTH{1'b0}};
                    if (pending_valid) begin
                        working_tag <= pending_tag;
                        working_inputs <= pending_inputs;
                        working_multipliers <= pending_multipliers;
                        pending_valid <= 1'b0;
                    end else if (valid_in) begin
                        working_tag <= tag_in;
                        working_inputs <= inputs_packed;
                        working_multipliers <= multipliers_packed;
                    end else begin
                        active <= 1'b0;
                    end
                end else begin
                    lane_counter <= lane_counter + 1'b1;
                    if (valid_in && ready_in) begin
                        pending_valid <= 1'b1;
                        pending_tag <= tag_in;
                        pending_inputs <= inputs_packed;
                        pending_multipliers <= multipliers_packed;
                    end
                end
            end
        end
    end

endmodule
