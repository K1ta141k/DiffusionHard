`timescale 1ns/1ps

// Eight-token INT8 MAC tile. Adjacent token lanes share each weight and are
// packed into one 27x18 multiply, so 8xN x32 products use 4xN x32 DSPs.
module int8_mac_tile_packed_m8_pipelined #(
    parameter integer N_LANES = 6,
    parameter integer ACC_WIDTH = 32,
    parameter integer TAG_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire clear_accumulators,
    input  wire last_k_tile,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [8*32*8-1:0] activations_packed,
    input  wire [N_LANES*32*8-1:0] weights_packed,
    output reg  valid_out,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output wire [8*N_LANES*ACC_WIDTH-1:0] accumulators_packed
);

    localparam integer M_LANES = 8;
    localparam integer M_PAIRS = M_LANES / 2;
    localparam integer OUTPUT_LANES = M_LANES * N_LANES;
    localparam integer PRODUCTS = OUTPUT_LANES * 32;

    wire [PRODUCTS*18-1:0] products_flat;
    wire [M_PAIRS*N_LANES*32-1:0] product_valid_unused;
    reg signed [16:0] sum_level_1 [0:OUTPUT_LANES-1][0:15];
    reg signed [17:0] sum_level_2 [0:OUTPUT_LANES-1][0:7];
    reg signed [18:0] sum_level_3 [0:OUTPUT_LANES-1][0:3];
    reg signed [19:0] sum_level_4 [0:OUTPUT_LANES-1][0:1];
    reg signed [20:0] dot_products [0:OUTPUT_LANES-1];
    reg signed [ACC_WIDTH-1:0] accumulators [0:OUTPUT_LANES-1];
    reg signed [ACC_WIDTH-1:0] results [0:OUTPUT_LANES-1];
    reg signed [8:0] weight_sum_level_1 [0:N_LANES-1][0:15];
    reg signed [9:0] weight_sum_level_2 [0:N_LANES-1][0:7];
    reg signed [10:0] weight_sum_level_3 [0:N_LANES-1][0:3];
    reg signed [11:0] weight_sum_level_4 [0:N_LANES-1][0:1];
    reg signed [12:0] weight_sum [0:N_LANES-1];
    reg signed [12:0] weight_sum_delay [0:N_LANES-1];
    reg [5:0] valid_pipeline;
    reg [5:0] clear_pipeline;
    reg [5:0] last_pipeline;
    reg [TAG_WIDTH-1:0] tag_pipeline [0:5];

    integer output_index;
    integer pair_index;
    integer pipeline_index;
    reg signed [ACC_WIDTH-1:0] updated_accumulator;
    reg signed [ACC_WIDTH-1:0] extended_weight_sum;

    genvar m_pair;
    genvar n_lane;
    genvar k_lane;
    generate
        for (m_pair = 0; m_pair < M_PAIRS; m_pair = m_pair + 1) begin : pairs
            for (n_lane = 0; n_lane < N_LANES; n_lane = n_lane + 1) begin : outputs
                for (k_lane = 0; k_lane < 32; k_lane = k_lane + 1) begin : products
                    wire signed [17:0] product_0;
                    wire signed [17:0] product_1;
                    int8_shared_weight_pair_offset_multiplier multiplier (
                        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                        .activation_0(activations_packed[
                            ((2*m_pair)*32+k_lane)*8 +: 8
                        ]),
                        .activation_1(activations_packed[
                            ((2*m_pair+1)*32+k_lane)*8 +: 8
                        ]),
                        .weight(weights_packed[(n_lane*32+k_lane)*8 +: 8]),
                        .valid_out(product_valid_unused[
                            (m_pair*N_LANES+n_lane)*32+k_lane
                        ]),
                        .offset_product_0(product_0),
                        .offset_product_1(product_1)
                    );
                    assign products_flat[
                        (((2*m_pair)*N_LANES+n_lane)*32+k_lane)*18 +: 18
                    ] = product_0;
                    assign products_flat[
                        (((2*m_pair+1)*N_LANES+n_lane)*32+k_lane)*18 +: 18
                    ] = product_1;
                end
            end
        end
        for (n_lane = 0; n_lane < OUTPUT_LANES; n_lane = n_lane + 1) begin : pack
            assign accumulators_packed[n_lane*ACC_WIDTH +: ACC_WIDTH] =
                results[n_lane];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_pipeline <= 6'b0;
            clear_pipeline <= 6'b0;
            last_pipeline <= 6'b0;
            valid_out <= 1'b0;
            tag_out <= {TAG_WIDTH{1'b0}};
            for (pipeline_index = 0; pipeline_index < 6;
                 pipeline_index = pipeline_index + 1)
                tag_pipeline[pipeline_index] <= {TAG_WIDTH{1'b0}};
            for (output_index = 0; output_index < N_LANES;
                 output_index = output_index + 1) begin
                weight_sum[output_index] <= 13'sd0;
                weight_sum_delay[output_index] <= 13'sd0;
                for (pair_index = 0; pair_index < 16;
                     pair_index = pair_index + 1)
                    weight_sum_level_1[output_index][pair_index] <= 9'sd0;
                for (pair_index = 0; pair_index < 8;
                     pair_index = pair_index + 1)
                    weight_sum_level_2[output_index][pair_index] <= 10'sd0;
                for (pair_index = 0; pair_index < 4;
                     pair_index = pair_index + 1)
                    weight_sum_level_3[output_index][pair_index] <= 11'sd0;
                for (pair_index = 0; pair_index < 2;
                     pair_index = pair_index + 1)
                    weight_sum_level_4[output_index][pair_index] <= 12'sd0;
            end
            for (output_index = 0; output_index < OUTPUT_LANES;
                 output_index = output_index + 1) begin
                accumulators[output_index] <= {ACC_WIDTH{1'b0}};
                results[output_index] <= {ACC_WIDTH{1'b0}};
                dot_products[output_index] <= 21'sd0;
                for (pair_index = 0; pair_index < 16;
                     pair_index = pair_index + 1)
                    sum_level_1[output_index][pair_index] <= 17'sd0;
                for (pair_index = 0; pair_index < 8;
                     pair_index = pair_index + 1)
                    sum_level_2[output_index][pair_index] <= 18'sd0;
                for (pair_index = 0; pair_index < 4;
                     pair_index = pair_index + 1)
                    sum_level_3[output_index][pair_index] <= 19'sd0;
                for (pair_index = 0; pair_index < 2;
                     pair_index = pair_index + 1)
                    sum_level_4[output_index][pair_index] <= 20'sd0;
            end
        end else begin
            valid_pipeline[0] <= valid_in;
            clear_pipeline[0] <= clear_accumulators;
            last_pipeline[0] <= last_k_tile;
            tag_pipeline[0] <= tag_in;
            for (pipeline_index = 1; pipeline_index < 6;
                 pipeline_index = pipeline_index + 1) begin
                valid_pipeline[pipeline_index] <= valid_pipeline[pipeline_index-1];
                clear_pipeline[pipeline_index] <= clear_pipeline[pipeline_index-1];
                last_pipeline[pipeline_index] <= last_pipeline[pipeline_index-1];
                tag_pipeline[pipeline_index] <= tag_pipeline[pipeline_index-1];
            end

            for (output_index = 0; output_index < OUTPUT_LANES;
                 output_index = output_index + 1) begin
                for (pair_index = 0; pair_index < 16;
                     pair_index = pair_index + 1)
                    sum_level_1[output_index][pair_index] <=
                        $signed(products_flat[
                            (output_index*32+2*pair_index)*18 +: 18
                        ]) + $signed(products_flat[
                            (output_index*32+2*pair_index+1)*18 +: 18
                        ]);
                for (pair_index = 0; pair_index < 8;
                     pair_index = pair_index + 1)
                    sum_level_2[output_index][pair_index] <=
                        $signed(sum_level_1[output_index][2*pair_index])
                        + $signed(sum_level_1[output_index][2*pair_index+1]);
                for (pair_index = 0; pair_index < 4;
                     pair_index = pair_index + 1)
                    sum_level_3[output_index][pair_index] <=
                        $signed(sum_level_2[output_index][2*pair_index])
                        + $signed(sum_level_2[output_index][2*pair_index+1]);
                for (pair_index = 0; pair_index < 2;
                     pair_index = pair_index + 1)
                    sum_level_4[output_index][pair_index] <=
                        $signed(sum_level_3[output_index][2*pair_index])
                        + $signed(sum_level_3[output_index][2*pair_index+1]);
                dot_products[output_index] <=
                    $signed(sum_level_4[output_index][0])
                    + $signed(sum_level_4[output_index][1]);
            end

            for (output_index = 0; output_index < N_LANES;
                 output_index = output_index + 1) begin
                for (pair_index = 0; pair_index < 16;
                     pair_index = pair_index + 1)
                    weight_sum_level_1[output_index][pair_index] <=
                        $signed(weights_packed[
                            (output_index*32+2*pair_index)*8 +: 8
                        ]) + $signed(weights_packed[
                            (output_index*32+2*pair_index+1)*8 +: 8
                        ]);
                for (pair_index = 0; pair_index < 8;
                     pair_index = pair_index + 1)
                    weight_sum_level_2[output_index][pair_index] <=
                        $signed(weight_sum_level_1[output_index][2*pair_index])
                        + $signed(weight_sum_level_1[output_index][2*pair_index+1]);
                for (pair_index = 0; pair_index < 4;
                     pair_index = pair_index + 1)
                    weight_sum_level_3[output_index][pair_index] <=
                        $signed(weight_sum_level_2[output_index][2*pair_index])
                        + $signed(weight_sum_level_2[output_index][2*pair_index+1]);
                for (pair_index = 0; pair_index < 2;
                     pair_index = pair_index + 1)
                    weight_sum_level_4[output_index][pair_index] <=
                        $signed(weight_sum_level_3[output_index][2*pair_index])
                        + $signed(weight_sum_level_3[output_index][2*pair_index+1]);
                weight_sum[output_index] <=
                    $signed(weight_sum_level_4[output_index][0])
                    + $signed(weight_sum_level_4[output_index][1]);
                weight_sum_delay[output_index] <= weight_sum[output_index];
            end

            valid_out <= 1'b0;
            if (valid_pipeline[5]) begin
                for (output_index = 0; output_index < OUTPUT_LANES;
                     output_index = output_index + 1) begin
                    extended_weight_sum = {{(ACC_WIDTH-13){
                        weight_sum_delay[output_index % N_LANES][12]
                    }}, weight_sum_delay[output_index % N_LANES]};
                    if (clear_pipeline[5])
                        updated_accumulator = $signed(dot_products[output_index])
                            - (extended_weight_sum <<< 7);
                    else
                        updated_accumulator = accumulators[output_index]
                            + $signed(dot_products[output_index])
                            - (extended_weight_sum <<< 7);
                    accumulators[output_index] <= updated_accumulator;
                    if (last_pipeline[5])
                        results[output_index] <= updated_accumulator;
                end
                if (last_pipeline[5]) begin
                    valid_out <= 1'b1;
                    tag_out <= tag_pipeline[5];
                end
            end
        end
    end

endmodule
