`timescale 1ns/1ps

module mlp_up_postprocess_serial #(
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer LANES = M_LANES * N_LANES,
    parameter integer ACC_WIDTH = 32,
    parameter integer MULTIPLIER_WIDTH = 24,
    parameter integer TOKEN_FACTOR_WIDTH = 16,
    parameter integer OUTPUT_FACTOR_WIDTH = 18,
    parameter integer FACTOR_SHIFT = 8,
    parameter integer SIDEBAND_WIDTH = N_LANES * 24,
    parameter integer DATA_WIDTH = 16,
    parameter integer RIGHT_SHIFT = 20,
    parameter integer TAG_WIDTH = 16,
    parameter integer LANE_WIDTH = (LANES <= 1) ? 1 : $clog2(LANES),
    parameter integer TOKEN_COUNTER_WIDTH = (M_LANES <= 1)
        ? 1 : $clog2(M_LANES),
    parameter integer OUTPUT_COUNTER_WIDTH = (N_LANES <= 1)
        ? 1 : $clog2(N_LANES)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    output wire ready_in,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [LANES*ACC_WIDTH-1:0] accumulators_packed,
    input  wire [M_LANES*TOKEN_FACTOR_WIDTH-1:0] token_factors_packed,
    input  wire [N_LANES*OUTPUT_FACTOR_WIDTH-1:0] output_factors_packed,
    input  wire [N_LANES*ACC_WIDTH-1:0] biases_packed,
    input  wire [SIDEBAND_WIDTH-1:0] sideband_in,
    output reg  valid_out,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output reg  [LANES*DATA_WIDTH-1:0] gelu_packed,
    output reg  [SIDEBAND_WIDTH-1:0] sideband_out
);

    localparam integer SCALAR_TAG_WIDTH = TAG_WIDTH + LANE_WIDTH;

    reg [LANES*ACC_WIDTH-1:0] pending_accumulators;
    reg [M_LANES*TOKEN_FACTOR_WIDTH-1:0] pending_token_factors;
    reg [N_LANES*OUTPUT_FACTOR_WIDTH-1:0] pending_output_factors;
    reg [N_LANES*ACC_WIDTH-1:0] pending_biases;
    reg [SIDEBAND_WIDTH-1:0] pending_sideband;
    reg [TAG_WIDTH-1:0] pending_tag;
    reg pending_valid;

    reg active;
    reg [LANE_WIDTH-1:0] lane_counter;
    reg [TOKEN_COUNTER_WIDTH-1:0] token_counter;
    reg [OUTPUT_COUNTER_WIDTH-1:0] output_counter;
    reg [LANES*ACC_WIDTH-1:0] working_accumulators;
    reg [M_LANES*TOKEN_FACTOR_WIDTH-1:0] working_token_factors;
    reg [N_LANES*OUTPUT_FACTOR_WIDTH-1:0] working_output_factors;
    reg [N_LANES*ACC_WIDTH-1:0] working_biases;
    reg [SIDEBAND_WIDTH-1:0] working_sideband;
    reg [TAG_WIDTH-1:0] working_tag;

    wire scalar_valid;
    wire [SCALAR_TAG_WIDTH-1:0] scalar_tag;
    wire [DATA_WIDTH-1:0] scalar_gelu;
    wire [SCALAR_TAG_WIDTH-1:0] scalar_input_tag;
    wire [ACC_WIDTH-1:0] scalar_accumulator;
    wire [TOKEN_FACTOR_WIDTH-1:0] scalar_token_factor;
    wire [OUTPUT_FACTOR_WIDTH-1:0] scalar_output_factor;
    wire [TOKEN_FACTOR_WIDTH+OUTPUT_FACTOR_WIDTH-1:0] factor_product;
    wire [MULTIPLIER_WIDTH-1:0] scalar_multiplier;
    wire [ACC_WIDTH-1:0] scalar_bias;
    wire [LANE_WIDTH-1:0] output_lane;
    wire [TAG_WIDTH-1:0] output_tag;
    wire requant_valid;
    wire [DATA_WIDTH-1:0] requantized_value;
    reg [SCALAR_TAG_WIDTH-1:0] requant_tag;
    reg [SCALAR_TAG_WIDTH-1:0] gelu_tag_stage_0;
    reg [SCALAR_TAG_WIDTH-1:0] gelu_tag_stage_1;
    reg [SCALAR_TAG_WIDTH-1:0] gelu_tag_stage_2;
    reg [SIDEBAND_WIDTH-1:0] requant_sideband;
    reg [SIDEBAND_WIDTH-1:0] gelu_sideband_stage_0;
    reg [SIDEBAND_WIDTH-1:0] gelu_sideband_stage_1;
    reg [SIDEBAND_WIDTH-1:0] gelu_sideband_stage_2;

    assign ready_in = !pending_valid;
    assign scalar_input_tag = {working_tag, lane_counter};
    assign scalar_accumulator = working_accumulators[
        lane_counter*ACC_WIDTH +: ACC_WIDTH
    ];
    assign scalar_token_factor = working_token_factors[
        token_counter*TOKEN_FACTOR_WIDTH +: TOKEN_FACTOR_WIDTH
    ];
    assign scalar_output_factor = working_output_factors[
        output_counter*OUTPUT_FACTOR_WIDTH +: OUTPUT_FACTOR_WIDTH
    ];
    assign factor_product = scalar_token_factor * scalar_output_factor;
    assign scalar_multiplier =
        (factor_product + (1 << (FACTOR_SHIFT-1))) >> FACTOR_SHIFT;
    assign scalar_bias = working_biases[
        output_counter*ACC_WIDTH +: ACC_WIDTH
    ];
    assign output_lane = scalar_tag[LANE_WIDTH-1:0];
    assign output_tag = scalar_tag[SCALAR_TAG_WIDTH-1:LANE_WIDTH];
    assign scalar_tag = gelu_tag_stage_2;

    fixed_requantize #(
        .LANES(1),
        .ACC_WIDTH(ACC_WIDTH),
        .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),
        .OUTPUT_WIDTH(DATA_WIDTH),
        .RIGHT_SHIFT(RIGHT_SHIFT)
    ) scalar_requantizer (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(active),
        .accumulators_packed(scalar_accumulator),
        .multipliers_packed(scalar_multiplier),
        .biases_packed(scalar_bias),
        .valid_out(requant_valid),
        .outputs_packed(requantized_value)
    );

    gelu_q10_lut_scalar_bram #(
        .DATA_WIDTH(DATA_WIDTH)
    ) scalar_gelu_rom (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(requant_valid),
        .input_value(requantized_value),
        .valid_out(scalar_valid),
        .output_value(scalar_gelu)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            pending_valid <= 1'b0;
            pending_accumulators <= {LANES*ACC_WIDTH{1'b0}};
            pending_token_factors <= {M_LANES*TOKEN_FACTOR_WIDTH{1'b0}};
            pending_output_factors <= {N_LANES*OUTPUT_FACTOR_WIDTH{1'b0}};
            pending_biases <= {N_LANES*ACC_WIDTH{1'b0}};
            pending_sideband <= {SIDEBAND_WIDTH{1'b0}};
            pending_tag <= {TAG_WIDTH{1'b0}};
            active <= 1'b0;
            lane_counter <= {LANE_WIDTH{1'b0}};
            token_counter <= {TOKEN_COUNTER_WIDTH{1'b0}};
            output_counter <= {OUTPUT_COUNTER_WIDTH{1'b0}};
            working_accumulators <= {LANES*ACC_WIDTH{1'b0}};
            working_token_factors <= {M_LANES*TOKEN_FACTOR_WIDTH{1'b0}};
            working_output_factors <= {N_LANES*OUTPUT_FACTOR_WIDTH{1'b0}};
            working_biases <= {N_LANES*ACC_WIDTH{1'b0}};
            working_sideband <= {SIDEBAND_WIDTH{1'b0}};
            working_tag <= {TAG_WIDTH{1'b0}};
            requant_tag <= {SCALAR_TAG_WIDTH{1'b0}};
            gelu_tag_stage_0 <= {SCALAR_TAG_WIDTH{1'b0}};
            gelu_tag_stage_1 <= {SCALAR_TAG_WIDTH{1'b0}};
            gelu_tag_stage_2 <= {SCALAR_TAG_WIDTH{1'b0}};
            requant_sideband <= {SIDEBAND_WIDTH{1'b0}};
            gelu_sideband_stage_0 <= {SIDEBAND_WIDTH{1'b0}};
            gelu_sideband_stage_1 <= {SIDEBAND_WIDTH{1'b0}};
            gelu_sideband_stage_2 <= {SIDEBAND_WIDTH{1'b0}};
            valid_out <= 1'b0;
            tag_out <= {TAG_WIDTH{1'b0}};
            gelu_packed <= {LANES*DATA_WIDTH{1'b0}};
            sideband_out <= {SIDEBAND_WIDTH{1'b0}};
        end else begin
            valid_out <= 1'b0;

            if (active) begin
                requant_tag <= scalar_input_tag;
                requant_sideband <= working_sideband;
            end
            if (requant_valid) begin
                gelu_tag_stage_0 <= requant_tag;
                gelu_sideband_stage_0 <= requant_sideband;
            end
            gelu_tag_stage_1 <= gelu_tag_stage_0;
            gelu_tag_stage_2 <= gelu_tag_stage_1;
            gelu_sideband_stage_1 <= gelu_sideband_stage_0;
            gelu_sideband_stage_2 <= gelu_sideband_stage_1;

            if (!active) begin
                if (pending_valid) begin
                    working_accumulators <= pending_accumulators;
                    working_token_factors <= pending_token_factors;
                    working_output_factors <= pending_output_factors;
                    working_biases <= pending_biases;
                    working_sideband <= pending_sideband;
                    working_tag <= pending_tag;
                    pending_valid <= 1'b0;
                    lane_counter <= {LANE_WIDTH{1'b0}};
                    token_counter <= {TOKEN_COUNTER_WIDTH{1'b0}};
                    output_counter <= {OUTPUT_COUNTER_WIDTH{1'b0}};
                    active <= 1'b1;
                end else if (valid_in) begin
                    working_accumulators <= accumulators_packed;
                    working_token_factors <= token_factors_packed;
                    working_output_factors <= output_factors_packed;
                    working_biases <= biases_packed;
                    working_sideband <= sideband_in;
                    working_tag <= tag_in;
                    lane_counter <= {LANE_WIDTH{1'b0}};
                    token_counter <= {TOKEN_COUNTER_WIDTH{1'b0}};
                    output_counter <= {OUTPUT_COUNTER_WIDTH{1'b0}};
                    active <= 1'b1;
                end
            end else if (lane_counter == LANES-1) begin
                if (pending_valid) begin
                    working_accumulators <= pending_accumulators;
                    working_token_factors <= pending_token_factors;
                    working_output_factors <= pending_output_factors;
                    working_biases <= pending_biases;
                    working_sideband <= pending_sideband;
                    working_tag <= pending_tag;
                    pending_valid <= 1'b0;
                    lane_counter <= {LANE_WIDTH{1'b0}};
                    token_counter <= {TOKEN_COUNTER_WIDTH{1'b0}};
                    output_counter <= {OUTPUT_COUNTER_WIDTH{1'b0}};
                end else if (valid_in) begin
                    working_accumulators <= accumulators_packed;
                    working_token_factors <= token_factors_packed;
                    working_output_factors <= output_factors_packed;
                    working_biases <= biases_packed;
                    working_sideband <= sideband_in;
                    working_tag <= tag_in;
                    lane_counter <= {LANE_WIDTH{1'b0}};
                    token_counter <= {TOKEN_COUNTER_WIDTH{1'b0}};
                    output_counter <= {OUTPUT_COUNTER_WIDTH{1'b0}};
                end else begin
                    active <= 1'b0;
                end
            end else begin
                lane_counter <= lane_counter + 1'b1;
                if (output_counter == N_LANES-1) begin
                    output_counter <= {OUTPUT_COUNTER_WIDTH{1'b0}};
                    token_counter <= token_counter + 1'b1;
                end else begin
                    output_counter <= output_counter + 1'b1;
                end
                if (valid_in && ready_in) begin
                    pending_accumulators <= accumulators_packed;
                    pending_token_factors <= token_factors_packed;
                    pending_output_factors <= output_factors_packed;
                    pending_biases <= biases_packed;
                    pending_sideband <= sideband_in;
                    pending_tag <= tag_in;
                    pending_valid <= 1'b1;
                end
            end

            if (scalar_valid) begin
                gelu_packed[output_lane*DATA_WIDTH +: DATA_WIDTH] <= scalar_gelu;
                if (output_lane == LANES-1) begin
                    valid_out <= 1'b1;
                    tag_out <= output_tag;
                    sideband_out <= gelu_sideband_stage_2;
                end
            end
        end
    end

endmodule

module mlp_up_postprocess_parallel4 #(
    parameter integer M_LANES=4,parameter integer N_LANES=6,
    parameter integer LANES=M_LANES*N_LANES,
    parameter integer ACC_WIDTH=32,parameter integer MULTIPLIER_WIDTH=24,
    parameter integer TOKEN_FACTOR_WIDTH=16,
    parameter integer OUTPUT_FACTOR_WIDTH=18,parameter integer FACTOR_SHIFT=8,
    parameter integer SIDEBAND_WIDTH=N_LANES*24,
    parameter integer DATA_WIDTH=16,parameter integer RIGHT_SHIFT=20,
    parameter integer TAG_WIDTH=16,parameter integer PARALLEL_LANES=4,
    parameter integer BATCHES=LANES/PARALLEL_LANES,
    parameter integer BATCH_WIDTH=(BATCHES<=1)?1:$clog2(BATCHES),
    parameter integer LANE_WIDTH=(LANES<=1)?1:$clog2(LANES)
)(
    input wire clk,input wire rst_n,input wire valid_in,output wire ready_in,
    input wire [TAG_WIDTH-1:0] tag_in,
    input wire [LANES*ACC_WIDTH-1:0] accumulators_packed,
    input wire [M_LANES*TOKEN_FACTOR_WIDTH-1:0] token_factors_packed,
    input wire [N_LANES*OUTPUT_FACTOR_WIDTH-1:0] output_factors_packed,
    input wire [N_LANES*ACC_WIDTH-1:0] biases_packed,
    input wire [SIDEBAND_WIDTH-1:0] sideband_in,
    output reg valid_out,output reg [TAG_WIDTH-1:0] tag_out,
    output reg [LANES*DATA_WIDTH-1:0] gelu_packed,
    output reg [SIDEBAND_WIDTH-1:0] sideband_out
);
    reg active,issuing;reg [BATCH_WIDTH-1:0] batch_counter;
    reg [BATCH_WIDTH-1:0] requant_batch,gelu_batch_0,gelu_batch_1,gelu_batch_2;
    reg [TAG_WIDTH-1:0] working_tag;
    reg [SIDEBAND_WIDTH-1:0] working_sideband;
    reg [LANES*ACC_WIDTH-1:0] working_accumulators;
    reg [M_LANES*TOKEN_FACTOR_WIDTH-1:0] working_token_factors;
    reg [N_LANES*OUTPUT_FACTOR_WIDTH-1:0] working_output_factors;
    reg [N_LANES*ACC_WIDTH-1:0] working_biases;
    wire [PARALLEL_LANES*ACC_WIDTH-1:0] batch_accumulators=
      working_accumulators[batch_counter*PARALLEL_LANES*ACC_WIDTH
        +:PARALLEL_LANES*ACC_WIDTH];
    wire [PARALLEL_LANES*MULTIPLIER_WIDTH-1:0] batch_multipliers;
    wire [PARALLEL_LANES*ACC_WIDTH-1:0] batch_biases;
    wire requant_valid;
    wire [PARALLEL_LANES*DATA_WIDTH-1:0] requantized;
    wire [PARALLEL_LANES-1:0] gelu_valid;
    wire [PARALLEL_LANES*DATA_WIDTH-1:0] gelu_outputs;
    assign ready_in=!active;
    genvar parallel_lane;
    generate for(parallel_lane=0;parallel_lane<PARALLEL_LANES;
      parallel_lane=parallel_lane+1)begin:parallel_lanes
      wire [LANE_WIDTH:0] lane_number=batch_counter*PARALLEL_LANES+parallel_lane;
      wire [$clog2(M_LANES)-1:0] token_number=lane_number/N_LANES;
      wire [$clog2(N_LANES)-1:0] output_number=lane_number%N_LANES;
      wire [TOKEN_FACTOR_WIDTH+OUTPUT_FACTOR_WIDTH-1:0] factor_product=
        working_token_factors[token_number*TOKEN_FACTOR_WIDTH+:TOKEN_FACTOR_WIDTH]
        *working_output_factors[output_number*OUTPUT_FACTOR_WIDTH+:OUTPUT_FACTOR_WIDTH];
      assign batch_multipliers[
        parallel_lane*MULTIPLIER_WIDTH+:MULTIPLIER_WIDTH]=
        (factor_product+(1<<(FACTOR_SHIFT-1)))>>FACTOR_SHIFT;
      assign batch_biases[parallel_lane*ACC_WIDTH+:ACC_WIDTH]=
        working_biases[output_number*ACC_WIDTH+:ACC_WIDTH];
      gelu_q10_lut_scalar_bram #(.DATA_WIDTH(DATA_WIDTH)) gelu_rom(
        .clk(clk),.rst_n(rst_n),.valid_in(requant_valid),
        .input_value(requantized[parallel_lane*DATA_WIDTH+:DATA_WIDTH]),
        .valid_out(gelu_valid[parallel_lane]),
        .output_value(gelu_outputs[parallel_lane*DATA_WIDTH+:DATA_WIDTH]));
    end endgenerate
    fixed_requantize #(.LANES(PARALLEL_LANES),.ACC_WIDTH(ACC_WIDTH),
      .MULTIPLIER_WIDTH(MULTIPLIER_WIDTH),.OUTPUT_WIDTH(DATA_WIDTH),
      .RIGHT_SHIFT(RIGHT_SHIFT)) requantizer(.clk(clk),.rst_n(rst_n),
      .valid_in(issuing),.accumulators_packed(batch_accumulators),
      .multipliers_packed(batch_multipliers),.biases_packed(batch_biases),
      .valid_out(requant_valid),.outputs_packed(requantized));
    always @(posedge clk)begin
      if(!rst_n)begin active<=0;issuing<=0;batch_counter<=0;
        requant_batch<=0;gelu_batch_0<=0;gelu_batch_1<=0;gelu_batch_2<=0;
        working_tag<=0;working_sideband<=0;working_accumulators<=0;
        working_token_factors<=0;working_output_factors<=0;working_biases<=0;
        valid_out<=0;tag_out<=0;gelu_packed<=0;sideband_out<=0;end
      else begin
        valid_out<=0;
        if(valid_in&&ready_in)begin active<=1;issuing<=1;batch_counter<=0;
          working_tag<=tag_in;working_sideband<=sideband_in;
          working_accumulators<=accumulators_packed;
          working_token_factors<=token_factors_packed;
          working_output_factors<=output_factors_packed;
          working_biases<=biases_packed;end
        if(issuing)begin
          requant_batch<=batch_counter;
          if(batch_counter==BATCHES-1)issuing<=0;
          else batch_counter<=batch_counter+1'b1;
        end
        if(requant_valid)gelu_batch_0<=requant_batch;
        gelu_batch_1<=gelu_batch_0;gelu_batch_2<=gelu_batch_1;
        if(gelu_valid[0])begin
          gelu_packed[gelu_batch_2*PARALLEL_LANES*DATA_WIDTH
            +:PARALLEL_LANES*DATA_WIDTH]<=gelu_outputs;
          if(gelu_batch_2==BATCHES-1)begin active<=0;valid_out<=1;
            tag_out<=working_tag;sideband_out<=working_sideband;end
        end
      end
    end
    initial begin
      if(PARALLEL_LANES!=4 || LANES%PARALLEL_LANES!=0)
        $error("parallel MLP up postprocess requires complete four-lane batches");
    end
endmodule
