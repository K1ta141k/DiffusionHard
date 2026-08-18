`timescale 1ns/1ps

module int8_mac_tile_pipelined #(
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer TAG_WIDTH = 8
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire clear_accumulators,
    input  wire last_k_tile,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [M_LANES*32*DATA_WIDTH-1:0] activations_packed,
    input  wire [N_LANES*32*DATA_WIDTH-1:0] weights_packed,
    output reg  valid_out,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output wire [M_LANES*N_LANES*ACC_WIDTH-1:0] accumulators_packed
);

    localparam integer K_LANES = 32;
    localparam integer OUTPUT_LANES = M_LANES * N_LANES;
    localparam integer PRODUCT_WIDTH = 2 * DATA_WIDTH;

    reg signed [PRODUCT_WIDTH-1:0] products [0:OUTPUT_LANES-1][0:31];
    reg signed [PRODUCT_WIDTH:0] sum_level_1 [0:OUTPUT_LANES-1][0:15];
    reg signed [PRODUCT_WIDTH+1:0] sum_level_2 [0:OUTPUT_LANES-1][0:7];
    reg signed [PRODUCT_WIDTH+2:0] sum_level_3 [0:OUTPUT_LANES-1][0:3];
    reg signed [PRODUCT_WIDTH+3:0] sum_level_4 [0:OUTPUT_LANES-1][0:1];
    reg signed [PRODUCT_WIDTH+4:0] dot_products [0:OUTPUT_LANES-1];

    reg signed [ACC_WIDTH-1:0] accumulators [0:OUTPUT_LANES-1];
    reg signed [ACC_WIDTH-1:0] results [0:OUTPUT_LANES-1];

    reg [5:0] valid_pipeline;
    reg [5:0] clear_pipeline;
    reg [5:0] last_pipeline;
    reg [TAG_WIDTH-1:0] tag_pipeline [0:5];

    integer output_index;
    integer m_index;
    integer n_index;
    integer k_index;
    integer pair_index;
    integer pipeline_index;
    reg signed [DATA_WIDTH-1:0] activation_value;
    reg signed [DATA_WIDTH-1:0] weight_value;
    reg signed [ACC_WIDTH-1:0] updated_accumulator;

    genvar output_lane;
    generate
        for (output_lane = 0; output_lane < OUTPUT_LANES; output_lane = output_lane + 1) begin : pack_outputs
            assign accumulators_packed[
                output_lane*ACC_WIDTH +: ACC_WIDTH
            ] = results[output_lane];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_pipeline <= 6'b0;
            clear_pipeline <= 6'b0;
            last_pipeline <= 6'b0;
            valid_out <= 1'b0;
            tag_out <= {TAG_WIDTH{1'b0}};

            for (pipeline_index = 0; pipeline_index < 6; pipeline_index = pipeline_index + 1) begin
                tag_pipeline[pipeline_index] <= {TAG_WIDTH{1'b0}};
            end
            for (output_index = 0; output_index < OUTPUT_LANES; output_index = output_index + 1) begin
                accumulators[output_index] <= {ACC_WIDTH{1'b0}};
                results[output_index] <= {ACC_WIDTH{1'b0}};
                dot_products[output_index] <= {(PRODUCT_WIDTH+5){1'b0}};
                for (k_index = 0; k_index < K_LANES; k_index = k_index + 1) begin
                    products[output_index][k_index] <= {PRODUCT_WIDTH{1'b0}};
                end
                for (pair_index = 0; pair_index < 16; pair_index = pair_index + 1) begin
                    sum_level_1[output_index][pair_index] <= {(PRODUCT_WIDTH+1){1'b0}};
                end
                for (pair_index = 0; pair_index < 8; pair_index = pair_index + 1) begin
                    sum_level_2[output_index][pair_index] <= {(PRODUCT_WIDTH+2){1'b0}};
                end
                for (pair_index = 0; pair_index < 4; pair_index = pair_index + 1) begin
                    sum_level_3[output_index][pair_index] <= {(PRODUCT_WIDTH+3){1'b0}};
                end
                for (pair_index = 0; pair_index < 2; pair_index = pair_index + 1) begin
                    sum_level_4[output_index][pair_index] <= {(PRODUCT_WIDTH+4){1'b0}};
                end
            end
        end else begin
            valid_pipeline[0] <= valid_in;
            clear_pipeline[0] <= clear_accumulators;
            last_pipeline[0] <= last_k_tile;
            tag_pipeline[0] <= tag_in;
            for (pipeline_index = 1; pipeline_index < 6; pipeline_index = pipeline_index + 1) begin
                valid_pipeline[pipeline_index] <= valid_pipeline[pipeline_index-1];
                clear_pipeline[pipeline_index] <= clear_pipeline[pipeline_index-1];
                last_pipeline[pipeline_index] <= last_pipeline[pipeline_index-1];
                tag_pipeline[pipeline_index] <= tag_pipeline[pipeline_index-1];
            end

            if (valid_in) begin
                for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
                    for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                        output_index = m_index * N_LANES + n_index;
                        for (k_index = 0; k_index < K_LANES; k_index = k_index + 1) begin
                            activation_value = $signed(
                                activations_packed[(m_index*K_LANES+k_index)*DATA_WIDTH +: DATA_WIDTH]
                            );
                            weight_value = $signed(
                                weights_packed[(n_index*K_LANES+k_index)*DATA_WIDTH +: DATA_WIDTH]
                            );
                            products[output_index][k_index] <= activation_value * weight_value;
                        end
                    end
                end
            end

            for (output_index = 0; output_index < OUTPUT_LANES; output_index = output_index + 1) begin
                for (pair_index = 0; pair_index < 16; pair_index = pair_index + 1) begin
                    sum_level_1[output_index][pair_index] <=
                        $signed(products[output_index][2*pair_index])
                        + $signed(products[output_index][2*pair_index+1]);
                end
                for (pair_index = 0; pair_index < 8; pair_index = pair_index + 1) begin
                    sum_level_2[output_index][pair_index] <=
                        $signed(sum_level_1[output_index][2*pair_index])
                        + $signed(sum_level_1[output_index][2*pair_index+1]);
                end
                for (pair_index = 0; pair_index < 4; pair_index = pair_index + 1) begin
                    sum_level_3[output_index][pair_index] <=
                        $signed(sum_level_2[output_index][2*pair_index])
                        + $signed(sum_level_2[output_index][2*pair_index+1]);
                end
                for (pair_index = 0; pair_index < 2; pair_index = pair_index + 1) begin
                    sum_level_4[output_index][pair_index] <=
                        $signed(sum_level_3[output_index][2*pair_index])
                        + $signed(sum_level_3[output_index][2*pair_index+1]);
                end
                dot_products[output_index] <=
                    $signed(sum_level_4[output_index][0])
                    + $signed(sum_level_4[output_index][1]);
            end

            valid_out <= 1'b0;
            if (valid_pipeline[5]) begin
                for (output_index = 0; output_index < OUTPUT_LANES; output_index = output_index + 1) begin
                    if (clear_pipeline[5]) begin
                        updated_accumulator = $signed(dot_products[output_index]);
                    end else begin
                        updated_accumulator = accumulators[output_index]
                            + $signed(dot_products[output_index]);
                    end
                    accumulators[output_index] <= updated_accumulator;
                    if (last_pipeline[5]) begin
                        results[output_index] <= updated_accumulator;
                    end
                end
                if (last_pipeline[5]) begin
                    valid_out <= 1'b1;
                    tag_out <= tag_pipeline[5];
                end
            end
        end
    end

endmodule
