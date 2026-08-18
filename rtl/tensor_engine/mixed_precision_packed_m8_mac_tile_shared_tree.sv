`timescale 1ns/1ps

module mixed_precision_packed_m8_mac_tile_shared_tree #(
    parameter integer N_LANES = 6,
    parameter integer TAG_WIDTH = 16
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire narrow_int8_mode,
    input  wire clear_accumulators,
    input  wire last_k_tile,
    input  wire [TAG_WIDTH-1:0] tag_in,
    input  wire [4*32*18-1:0] attention_activations_packed,
    input  wire [N_LANES*32*18-1:0] attention_weights_packed,
    input  wire [8*32*8-1:0] mlp_activations_packed,
    input  wire [N_LANES*32*8-1:0] mlp_weights_packed,
    output reg  valid_out,
    output reg  narrow_int8_mode_out,
    output reg  [TAG_WIDTH-1:0] tag_out,
    output wire [4*N_LANES*48-1:0] attention_accumulators_packed,
    output wire [8*N_LANES*32-1:0] mlp_accumulators_packed
);

    localparam integer PHYSICAL_M_LANES = 4;
    localparam integer ATTENTION_OUTPUTS = 4 * N_LANES;
    localparam integer SHARED_OUTPUTS = 8 * N_LANES;
    wire [ATTENTION_OUTPUTS*32*36-1:0] attention_products_flat;
    wire [SHARED_OUTPUTS*32*18-1:0] mlp_products_flat;
    wire [PHYSICAL_M_LANES*N_LANES*32-1:0] valid_unused;
    wire [PHYSICAL_M_LANES*N_LANES*32-1:0] mode_unused;

    reg signed [40:0] shared_sum_1 [0:SHARED_OUTPUTS-1][0:15];
    reg signed [40:0] shared_sum_2 [0:SHARED_OUTPUTS-1][0:7];
    reg signed [40:0] shared_sum_3 [0:SHARED_OUTPUTS-1][0:3];
    reg signed [40:0] shared_sum_4 [0:SHARED_OUTPUTS-1][0:1];
    reg signed [40:0] shared_dots [0:SHARED_OUTPUTS-1];
    reg signed [47:0] shared_accumulators [0:SHARED_OUTPUTS-1];
    reg signed [47:0] shared_results [0:SHARED_OUTPUTS-1];

    reg signed [8:0] weight_sum_1 [0:N_LANES-1][0:15];
    reg signed [9:0] weight_sum_2 [0:N_LANES-1][0:7];
    reg signed [10:0] weight_sum_3 [0:N_LANES-1][0:3];
    reg signed [11:0] weight_sum_4 [0:N_LANES-1][0:1];
    reg signed [12:0] weight_sum [0:N_LANES-1];
    reg signed [12:0] weight_sum_delay [0:N_LANES-1];
    reg [5:0] valid_pipeline;
    reg [5:0] mode_pipeline;
    reg [5:0] clear_pipeline;
    reg [5:0] last_pipeline;
    reg [TAG_WIDTH-1:0] tag_pipeline [0:5];

    integer output_index;
    integer pair_index;
    integer pipeline_index;
    reg signed [47:0] updated_accumulator;
    reg signed [47:0] extended_weight_sum;

    genvar physical_m;
    genvar n_lane;
    genvar k_lane;
    generate
        for (physical_m = 0; physical_m < PHYSICAL_M_LANES;
             physical_m = physical_m + 1) begin : physical_rows
            for (n_lane = 0; n_lane < N_LANES; n_lane = n_lane + 1) begin : physical_columns
                for (k_lane = 0; k_lane < 32; k_lane = k_lane + 1) begin : products
                    wire signed [35:0] attention_product;
                    wire signed [17:0] mlp_product_0;
                    wire signed [17:0] mlp_product_1;
                    mixed_precision_token_pair_multiplier multiplier (
                        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                        .narrow_int8_mode(narrow_int8_mode),
                        .attention_activation(attention_activations_packed[
                            (physical_m*32+k_lane)*18 +: 18
                        ]),
                        .attention_weight(attention_weights_packed[
                            (n_lane*32+k_lane)*18 +: 18
                        ]),
                        .mlp_activation_0(mlp_activations_packed[
                            ((2*physical_m)*32+k_lane)*8 +: 8
                        ]),
                        .mlp_activation_1(mlp_activations_packed[
                            ((2*physical_m+1)*32+k_lane)*8 +: 8
                        ]),
                        .mlp_weight(mlp_weights_packed[
                            (n_lane*32+k_lane)*8 +: 8
                        ]),
                        .valid_out(valid_unused[
                            (physical_m*N_LANES+n_lane)*32+k_lane
                        ]),
                        .narrow_int8_mode_out(mode_unused[
                            (physical_m*N_LANES+n_lane)*32+k_lane
                        ]),
                        .attention_product(attention_product),
                        .mlp_offset_product_0(mlp_product_0),
                        .mlp_offset_product_1(mlp_product_1)
                    );
                    assign attention_products_flat[
                        ((physical_m*N_LANES+n_lane)*32+k_lane)*36 +: 36
                    ] = attention_product;
                    assign mlp_products_flat[
                        (((2*physical_m)*N_LANES+n_lane)*32+k_lane)*18 +: 18
                    ] = mlp_product_0;
                    assign mlp_products_flat[
                        (((2*physical_m+1)*N_LANES+n_lane)*32+k_lane)*18 +: 18
                    ] = mlp_product_1;
                end
            end
        end
        for (n_lane = 0; n_lane < ATTENTION_OUTPUTS; n_lane = n_lane + 1) begin : pack_attention
            assign attention_accumulators_packed[n_lane*48 +: 48] =
                shared_results[n_lane];
        end
        for (n_lane = 0; n_lane < SHARED_OUTPUTS; n_lane = n_lane + 1) begin : pack_mlp
            assign mlp_accumulators_packed[n_lane*32 +: 32] =
                shared_results[n_lane][31:0];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_pipeline <= 0;
            mode_pipeline <= 0;
            clear_pipeline <= 0;
            last_pipeline <= 0;
            valid_out <= 0;
            narrow_int8_mode_out <= 0;
            tag_out <= 0;
            for (pipeline_index = 0; pipeline_index < 6;
                 pipeline_index = pipeline_index + 1)
                tag_pipeline[pipeline_index] <= 0;
            for (output_index = 0; output_index < SHARED_OUTPUTS;
                 output_index = output_index + 1) begin
                shared_dots[output_index] <= 0;
                shared_accumulators[output_index] <= 0;
                shared_results[output_index] <= 0;
                for (pair_index = 0; pair_index < 16; pair_index = pair_index + 1)
                    shared_sum_1[output_index][pair_index] <= 0;
                for (pair_index = 0; pair_index < 8; pair_index = pair_index + 1)
                    shared_sum_2[output_index][pair_index] <= 0;
                for (pair_index = 0; pair_index < 4; pair_index = pair_index + 1)
                    shared_sum_3[output_index][pair_index] <= 0;
                for (pair_index = 0; pair_index < 2; pair_index = pair_index + 1)
                    shared_sum_4[output_index][pair_index] <= 0;
            end
            for (output_index = 0; output_index < N_LANES;
                 output_index = output_index + 1) begin
                weight_sum[output_index] <= 0;
                weight_sum_delay[output_index] <= 0;
                for (pair_index = 0; pair_index < 16; pair_index = pair_index + 1)
                    weight_sum_1[output_index][pair_index] <= 0;
                for (pair_index = 0; pair_index < 8; pair_index = pair_index + 1)
                    weight_sum_2[output_index][pair_index] <= 0;
                for (pair_index = 0; pair_index < 4; pair_index = pair_index + 1)
                    weight_sum_3[output_index][pair_index] <= 0;
                for (pair_index = 0; pair_index < 2; pair_index = pair_index + 1)
                    weight_sum_4[output_index][pair_index] <= 0;
            end
        end else begin
            valid_pipeline[0] <= valid_in;
            mode_pipeline[0] <= narrow_int8_mode;
            clear_pipeline[0] <= clear_accumulators;
            last_pipeline[0] <= last_k_tile;
            tag_pipeline[0] <= tag_in;
            for (pipeline_index = 1; pipeline_index < 6;
                 pipeline_index = pipeline_index + 1) begin
                valid_pipeline[pipeline_index] <= valid_pipeline[pipeline_index-1];
                mode_pipeline[pipeline_index] <= mode_pipeline[pipeline_index-1];
                clear_pipeline[pipeline_index] <= clear_pipeline[pipeline_index-1];
                last_pipeline[pipeline_index] <= last_pipeline[pipeline_index-1];
                tag_pipeline[pipeline_index] <= tag_pipeline[pipeline_index-1];
            end

            if (mode_pipeline[0]) begin
                for (output_index = 0; output_index < SHARED_OUTPUTS;
                     output_index = output_index + 1)
                    for (pair_index = 0; pair_index < 16;
                         pair_index = pair_index + 1)
                        shared_sum_1[output_index][pair_index] <=
                            $signed(mlp_products_flat[
                                (output_index*32+2*pair_index)*18 +: 18
                            ]) + $signed(mlp_products_flat[
                                (output_index*32+2*pair_index+1)*18 +: 18
                            ]);
            end else begin
                for (output_index = 0; output_index < SHARED_OUTPUTS;
                     output_index = output_index + 1)
                    for (pair_index = 0; pair_index < 16;
                         pair_index = pair_index + 1)
                        if (output_index < ATTENTION_OUTPUTS)
                            shared_sum_1[output_index][pair_index] <=
                                $signed(attention_products_flat[
                                    (output_index*32+2*pair_index)*36 +: 36
                                ]) + $signed(attention_products_flat[
                                    (output_index*32+2*pair_index+1)*36 +: 36
                                ]);
                        else
                            shared_sum_1[output_index][pair_index] <= 0;
            end
            for (output_index = 0; output_index < SHARED_OUTPUTS;
                 output_index = output_index + 1) begin
                for (pair_index = 0; pair_index < 8; pair_index = pair_index + 1)
                    shared_sum_2[output_index][pair_index] <=
                        $signed(shared_sum_1[output_index][2*pair_index])
                        + $signed(shared_sum_1[output_index][2*pair_index+1]);
                for (pair_index = 0; pair_index < 4; pair_index = pair_index + 1)
                    shared_sum_3[output_index][pair_index] <=
                        $signed(shared_sum_2[output_index][2*pair_index])
                        + $signed(shared_sum_2[output_index][2*pair_index+1]);
                for (pair_index = 0; pair_index < 2; pair_index = pair_index + 1)
                    shared_sum_4[output_index][pair_index] <=
                        $signed(shared_sum_3[output_index][2*pair_index])
                        + $signed(shared_sum_3[output_index][2*pair_index+1]);
                shared_dots[output_index] <=
                    $signed(shared_sum_4[output_index][0])
                    + $signed(shared_sum_4[output_index][1]);
            end

            if (narrow_int8_mode) begin
                for (output_index = 0; output_index < N_LANES;
                     output_index = output_index + 1) begin
                    for (pair_index = 0; pair_index < 16;
                         pair_index = pair_index + 1)
                        weight_sum_1[output_index][pair_index] <=
                            $signed(mlp_weights_packed[
                                (output_index*32+2*pair_index)*8 +: 8
                            ]) + $signed(mlp_weights_packed[
                                (output_index*32+2*pair_index+1)*8 +: 8
                            ]);
                    for (pair_index = 0; pair_index < 8; pair_index = pair_index + 1)
                        weight_sum_2[output_index][pair_index] <=
                            $signed(weight_sum_1[output_index][2*pair_index])
                            + $signed(weight_sum_1[output_index][2*pair_index+1]);
                    for (pair_index = 0; pair_index < 4; pair_index = pair_index + 1)
                        weight_sum_3[output_index][pair_index] <=
                            $signed(weight_sum_2[output_index][2*pair_index])
                            + $signed(weight_sum_2[output_index][2*pair_index+1]);
                    for (pair_index = 0; pair_index < 2; pair_index = pair_index + 1)
                        weight_sum_4[output_index][pair_index] <=
                            $signed(weight_sum_3[output_index][2*pair_index])
                            + $signed(weight_sum_3[output_index][2*pair_index+1]);
                    weight_sum[output_index] <=
                        $signed(weight_sum_4[output_index][0])
                        + $signed(weight_sum_4[output_index][1]);
                    weight_sum_delay[output_index] <= weight_sum[output_index];
                end
            end

            valid_out <= 0;
            if (valid_pipeline[5]) begin
                for (output_index = 0; output_index < SHARED_OUTPUTS;
                     output_index = output_index + 1) begin
                    extended_weight_sum = {{35{
                        weight_sum_delay[output_index % N_LANES][12]
                    }}, weight_sum_delay[output_index % N_LANES]};
                    if (mode_pipeline[5]) begin
                        if (clear_pipeline[5])
                            updated_accumulator = $signed(shared_dots[output_index])
                                - (extended_weight_sum <<< 7);
                        else
                            updated_accumulator = shared_accumulators[output_index]
                                + $signed(shared_dots[output_index])
                                - (extended_weight_sum <<< 7);
                    end else if (clear_pipeline[5]) begin
                        updated_accumulator = $signed(shared_dots[output_index]);
                    end else begin
                        updated_accumulator = shared_accumulators[output_index]
                            + $signed(shared_dots[output_index]);
                    end
                    shared_accumulators[output_index] <= updated_accumulator;
                    if (last_pipeline[5])
                        shared_results[output_index] <= updated_accumulator;
                end
                if (last_pipeline[5]) begin
                    valid_out <= 1;
                    narrow_int8_mode_out <= mode_pipeline[5];
                    tag_out <= tag_pipeline[5];
                end
            end
        end
    end

endmodule
