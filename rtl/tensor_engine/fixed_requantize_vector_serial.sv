`timescale 1ns/1ps

module fixed_requantize_vector_serial #(
    parameter integer LANES = 24,
    parameter integer ACC_WIDTH = 32,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer OUTPUT_WIDTH = 24,
    parameter integer RIGHT_SHIFT = 20,
    parameter integer TAG_WIDTH = 16,
    parameter integer LANE_WIDTH = (LANES <= 1) ? 1 : $clog2(LANES)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    output wire ready_in,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [LANES*ACC_WIDTH-1:0] accumulators_packed,
    input  wire [LANES*MULTIPLIER_WIDTH-1:0] multipliers_packed,
    input  wire [LANES*ACC_WIDTH-1:0] biases_packed,
    output reg  valid_out,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output reg  [LANES*OUTPUT_WIDTH-1:0] outputs_packed
);

    localparam integer SCALAR_TAG_WIDTH = TAG_WIDTH + LANE_WIDTH;

    reg [LANES*ACC_WIDTH-1:0] pending_accumulators;
    reg [LANES*MULTIPLIER_WIDTH-1:0] pending_multipliers;
    reg [LANES*ACC_WIDTH-1:0] pending_biases;
    reg [TAG_WIDTH-1:0] pending_tag;
    reg pending_valid;

    reg active;
    reg [LANE_WIDTH-1:0] lane_counter;
    reg [LANES*ACC_WIDTH-1:0] working_accumulators;
    reg [LANES*MULTIPLIER_WIDTH-1:0] working_multipliers;
    reg [LANES*ACC_WIDTH-1:0] working_biases;
    reg [TAG_WIDTH-1:0] working_tag;
    reg [SCALAR_TAG_WIDTH-1:0] requant_tag;

    wire scalar_valid;
    wire [OUTPUT_WIDTH-1:0] scalar_output;
    wire [SCALAR_TAG_WIDTH-1:0] scalar_input_tag;
    wire [ACC_WIDTH-1:0] scalar_accumulator;
    wire [MULTIPLIER_WIDTH-1:0] scalar_multiplier;
    wire [ACC_WIDTH-1:0] scalar_bias;
    wire [LANE_WIDTH-1:0] output_lane;
    wire [TAG_WIDTH-1:0] output_tag;

    assign ready_in = !pending_valid;
    assign scalar_input_tag = {working_tag, lane_counter};
    assign scalar_accumulator = working_accumulators[
        lane_counter*ACC_WIDTH +: ACC_WIDTH
    ];
    assign scalar_multiplier = working_multipliers[
        lane_counter*MULTIPLIER_WIDTH +: MULTIPLIER_WIDTH
    ];
    assign scalar_bias = working_biases[
        lane_counter*ACC_WIDTH +: ACC_WIDTH
    ];
    assign output_lane = requant_tag[LANE_WIDTH-1:0];
    assign output_tag = requant_tag[SCALAR_TAG_WIDTH-1:LANE_WIDTH];

    fixed_requantize #(
        .LANES(1),
        .ACC_WIDTH(ACC_WIDTH),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH),
        .RIGHT_SHIFT(RIGHT_SHIFT)
    ) scalar_requantizer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(active),
        .accumulators_packed(scalar_accumulator),
        .multipliers_packed(scalar_multiplier),
        .biases_packed(scalar_bias),
        .valid_out(scalar_valid),
        .outputs_packed(scalar_output)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            pending_accumulators <= {LANES*ACC_WIDTH{1'b0}};
            pending_multipliers <= {LANES*MULTIPLIER_WIDTH{1'b0}};
            pending_biases <= {LANES*ACC_WIDTH{1'b0}};
            pending_tag <= {TAG_WIDTH{1'b0}};
            pending_valid <= 1'b0;
            active <= 1'b0;
            lane_counter <= {LANE_WIDTH{1'b0}};
            working_accumulators <= {LANES*ACC_WIDTH{1'b0}};
            working_multipliers <= {LANES*MULTIPLIER_WIDTH{1'b0}};
            working_biases <= {LANES*ACC_WIDTH{1'b0}};
            working_tag <= {TAG_WIDTH{1'b0}};
            requant_tag <= {SCALAR_TAG_WIDTH{1'b0}};
            valid_out <= 1'b0;
            tag_out <= {TAG_WIDTH{1'b0}};
            outputs_packed <= {LANES*OUTPUT_WIDTH{1'b0}};
        end else begin
            valid_out <= 1'b0;
            if (active) begin
                requant_tag <= scalar_input_tag;
            end

            if (!active) begin
                if (pending_valid) begin
                    working_accumulators <= pending_accumulators;
                    working_multipliers <= pending_multipliers;
                    working_biases <= pending_biases;
                    working_tag <= pending_tag;
                    pending_valid <= 1'b0;
                    lane_counter <= {LANE_WIDTH{1'b0}};
                    active <= 1'b1;
                end else if (valid_in) begin
                    working_accumulators <= accumulators_packed;
                    working_multipliers <= multipliers_packed;
                    working_biases <= biases_packed;
                    working_tag <= tag_in;
                    lane_counter <= {LANE_WIDTH{1'b0}};
                    active <= 1'b1;
                end
            end else if (lane_counter == LANES-1) begin
                if (pending_valid) begin
                    working_accumulators <= pending_accumulators;
                    working_multipliers <= pending_multipliers;
                    working_biases <= pending_biases;
                    working_tag <= pending_tag;
                    pending_valid <= 1'b0;
                    lane_counter <= {LANE_WIDTH{1'b0}};
                end else if (valid_in) begin
                    working_accumulators <= accumulators_packed;
                    working_multipliers <= multipliers_packed;
                    working_biases <= biases_packed;
                    working_tag <= tag_in;
                    lane_counter <= {LANE_WIDTH{1'b0}};
                end else begin
                    active <= 1'b0;
                end
            end else begin
                lane_counter <= lane_counter + 1'b1;
                if (valid_in && ready_in) begin
                    pending_accumulators <= accumulators_packed;
                    pending_multipliers <= multipliers_packed;
                    pending_biases <= biases_packed;
                    pending_tag <= tag_in;
                    pending_valid <= 1'b1;
                end
            end

            if (scalar_valid) begin
                outputs_packed[
                    output_lane*OUTPUT_WIDTH +: OUTPUT_WIDTH
                ] <= scalar_output;
                if (output_lane == LANES-1) begin
                    valid_out <= 1'b1;
                    tag_out <= output_tag;
                end
            end
        end
    end

endmodule

module fixed_requantize_vector_parallel #(
    parameter integer LANES=24,
    parameter integer ACC_WIDTH=32,
    parameter integer MULTIPLIER_WIDTH=24,
    parameter integer OUTPUT_WIDTH=24,
    parameter integer RIGHT_SHIFT=20,
    parameter integer TAG_WIDTH=16,
    parameter integer PARALLEL_LANES=4,
    parameter integer BATCHES=LANES/PARALLEL_LANES,
    parameter integer BATCH_WIDTH=(BATCHES<=1)?1:$clog2(BATCHES)
)(
    input wire clk,input wire rst_n,input wire valid_in,output wire ready_in,
    input wire [TAG_WIDTH-1:0] tag_in,
    input wire [LANES*ACC_WIDTH-1:0] accumulators_packed,
    input wire [LANES*MULTIPLIER_WIDTH-1:0] multipliers_packed,
    input wire [LANES*ACC_WIDTH-1:0] biases_packed,
    output reg valid_out,output reg [TAG_WIDTH-1:0] tag_out,
    output reg [LANES*OUTPUT_WIDTH-1:0] outputs_packed
);
    reg active;reg [BATCH_WIDTH-1:0] batch_counter,output_batch;
    reg [TAG_WIDTH-1:0] working_tag;
    reg [LANES*ACC_WIDTH-1:0] working_accumulators,working_biases;
    reg [LANES*MULTIPLIER_WIDTH-1:0] working_multipliers;
    wire batch_valid;wire [PARALLEL_LANES*OUTPUT_WIDTH-1:0] batch_outputs;
    wire [PARALLEL_LANES*ACC_WIDTH-1:0] batch_accumulators=
      working_accumulators[batch_counter*PARALLEL_LANES*ACC_WIDTH
        +:PARALLEL_LANES*ACC_WIDTH];
    wire [PARALLEL_LANES*MULTIPLIER_WIDTH-1:0] batch_multipliers=
      working_multipliers[batch_counter*PARALLEL_LANES*MULTIPLIER_WIDTH
        +:PARALLEL_LANES*MULTIPLIER_WIDTH];
    wire [PARALLEL_LANES*ACC_WIDTH-1:0] batch_biases=
      working_biases[batch_counter*PARALLEL_LANES*ACC_WIDTH
        +:PARALLEL_LANES*ACC_WIDTH];
    assign ready_in=!active;
    fixed_requantize #(.LANES(PARALLEL_LANES),.ACC_WIDTH(ACC_WIDTH),
      .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),.OUTPUT_WIDTH(OUTPUT_WIDTH),
      .RIGHT_SHIFT(RIGHT_SHIFT)) batch_requantizer(.clk(clk),.rst_n(rst_n),
      .valid_in(active),.accumulators_packed(batch_accumulators),
      .multipliers_packed(batch_multipliers),.biases_packed(batch_biases),
      .valid_out(batch_valid),.outputs_packed(batch_outputs));
    always @(posedge clk)begin
      if(!rst_n)begin active<=0;batch_counter<=0;output_batch<=0;
        working_tag<=0;working_accumulators<=0;working_multipliers<=0;
        working_biases<=0;valid_out<=0;tag_out<=0;outputs_packed<=0;end
      else begin
        valid_out<=0;
        if(valid_in&&ready_in)begin active<=1;batch_counter<=0;
          working_tag<=tag_in;working_accumulators<=accumulators_packed;
          working_multipliers<=multipliers_packed;working_biases<=biases_packed;
        end
        if(active)begin
          output_batch<=batch_counter;
          if(batch_counter==BATCHES-1)active<=0;
          else batch_counter<=batch_counter+1'b1;
        end
        if(batch_valid)begin
          outputs_packed[output_batch*PARALLEL_LANES*OUTPUT_WIDTH
            +:PARALLEL_LANES*OUTPUT_WIDTH]<=batch_outputs;
          if(output_batch==BATCHES-1)begin valid_out<=1;tag_out<=working_tag;end
        end
      end
    end
    initial if(PARALLEL_LANES<1||LANES%PARALLEL_LANES)
      $error("parallel requantizer lanes must divide vector lanes");
endmodule
