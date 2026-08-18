`timescale 1ns/1ps

module mixed_precision_packed_m8_mac_tile_pipelined #(
    parameter integer N_LANES = 6,
    parameter integer ATTENTION_ACC_WIDTH = 48,
    parameter integer MLP_ACC_WIDTH = 32,
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
    output wire [4*N_LANES*ATTENTION_ACC_WIDTH-1:0]
        attention_accumulators_packed,
    output wire [8*N_LANES*MLP_ACC_WIDTH-1:0]
        mlp_accumulators_packed
);

    localparam integer PHYSICAL_M_LANES = 4;
    localparam integer ATTENTION_OUTPUTS = 4 * N_LANES;
    localparam integer MLP_OUTPUTS = 8 * N_LANES;
    localparam integer ATTENTION_GROUPS = 4;
    localparam integer MLP_TREE_OUTPUTS = MLP_OUTPUTS;
    wire [ATTENTION_OUTPUTS*ATTENTION_GROUPS*48-1:0]
        attention_group_partials_flat;
    wire [ATTENTION_OUTPUTS*32*48-1:0] dsp_results_flat;
    wire [ATTENTION_OUTPUTS*32*48-1:0] dsp_cascade_flat;
    wire [MLP_OUTPUTS*32*18-1:0] mlp_products_flat;
    wire [PHYSICAL_M_LANES*32*27-1:0] selected_activations_flat;
    wire [N_LANES*32*18-1:0] selected_weights_flat;

    reg signed [47:0] attention_partial_sum [0:ATTENTION_OUTPUTS-1][0:1];
    reg signed [47:0] attention_dot_stage [0:ATTENTION_OUTPUTS-1];
    reg signed [47:0] attention_dot_delay_0 [0:ATTENTION_OUTPUTS-1];
    reg signed [47:0] attention_dot_delay_1 [0:ATTENTION_OUTPUTS-1];
    reg signed [47:0] attention_dot_delay_2 [0:ATTENTION_OUTPUTS-1];
`ifndef SYNTHESIS
    reg signed [36:0] behavioral_attention_sum_1
        [0:ATTENTION_OUTPUTS-1][0:15];
    reg signed [37:0] behavioral_attention_sum_2
        [0:ATTENTION_OUTPUTS-1][0:7];
    reg signed [38:0] behavioral_attention_sum_3
        [0:ATTENTION_OUTPUTS-1][0:3];
`endif
    reg signed [ATTENTION_ACC_WIDTH-1:0]
        attention_accumulators [0:ATTENTION_OUTPUTS-1];
    reg signed [ATTENTION_ACC_WIDTH-1:0]
        attention_results [0:ATTENTION_OUTPUTS-1];

    reg signed [16:0] mlp_sum_1 [0:MLP_TREE_OUTPUTS-1][0:15];
    reg signed [17:0] mlp_sum_2 [0:MLP_TREE_OUTPUTS-1][0:7];
    reg signed [18:0] mlp_sum_3 [0:MLP_TREE_OUTPUTS-1][0:3];
    reg signed [19:0] mlp_sum_4 [0:MLP_TREE_OUTPUTS-1][0:1];
    reg signed [20:0] mlp_dots [0:MLP_TREE_OUTPUTS-1];
    reg signed [MLP_ACC_WIDTH-1:0]
        mlp_accumulators [0:MLP_TREE_OUTPUTS-1];
    reg signed [MLP_ACC_WIDTH-1:0]
        mlp_results [0:MLP_TREE_OUTPUTS-1];
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
    reg signed [ATTENTION_ACC_WIDTH-1:0] updated_attention;
    reg signed [MLP_ACC_WIDTH-1:0] updated_mlp;
    reg signed [ATTENTION_ACC_WIDTH-1:0] extended_weight_sum;

    genvar physical_m;
    genvar n_lane;
    genvar k_lane;
    generate
        for (physical_m = 0; physical_m < PHYSICAL_M_LANES;
             physical_m = physical_m + 1) begin : shared_activations
            for (k_lane = 0; k_lane < 32; k_lane = k_lane + 1) begin : select
                wire [7:0] mlp_activation_0_offset;
                wire [7:0] mlp_activation_1_offset;
                assign mlp_activation_0_offset = mlp_activations_packed[
                    ((2*physical_m)*32+k_lane)*8 +: 8
                ] ^ 8'h80;
                assign mlp_activation_1_offset = mlp_activations_packed[
                    ((2*physical_m+1)*32+k_lane)*8 +: 8
                ] ^ 8'h80;
                assign selected_activations_flat[
                    (physical_m*32+k_lane)*27 +: 27
                ] = narrow_int8_mode
                    ? {1'b0, mlp_activation_1_offset, 10'b0,
                        mlp_activation_0_offset}
                    : {{9{attention_activations_packed[
                        (physical_m*32+k_lane)*18+17
                    ]}}, attention_activations_packed[
                        (physical_m*32+k_lane)*18 +: 18
                    ]};
            end
        end
        for (n_lane = 0; n_lane < N_LANES; n_lane = n_lane + 1) begin : shared_weights
            for (k_lane = 0; k_lane < 32; k_lane = k_lane + 1) begin : select
                assign selected_weights_flat[
                    (n_lane*32+k_lane)*18 +: 18
                ] = narrow_int8_mode
                    ? {{10{mlp_weights_packed[
                        (n_lane*32+k_lane)*8+7
                    ]}}, mlp_weights_packed[(n_lane*32+k_lane)*8 +: 8]}
                    : attention_weights_packed[
                        (n_lane*32+k_lane)*18 +: 18
                    ];
            end
        end
        for (physical_m = 0; physical_m < PHYSICAL_M_LANES;
             physical_m = physical_m + 1) begin : physical_rows
            for (n_lane = 0; n_lane < N_LANES; n_lane = n_lane + 1) begin : physical_columns
                for (k_lane = 0; k_lane < 32; k_lane = k_lane + 1) begin : products
                    wire signed [47:0] cascade_input;
                    wire signed [47:0] dsp_result;
                    wire signed [47:0] dsp_cascade;
                    wire signed [17:0] mlp_product_0;
                    wire signed [17:0] mlp_product_1;
                    wire signed [17:0] low_mlp_product;
                    wire signed [26:0] high_mlp_floor;
                    wire signed [26:0] high_mlp_product;
                    wire signed [35:0] behavioral_product;
                    if ((k_lane % 8) == 0) begin : group_start
                        assign cascade_input = 48'sd0;
                    end else begin : group_continue
                        assign cascade_input = dsp_cascade_flat[
                            ((physical_m*N_LANES+n_lane)*32+k_lane-1)*48
                                +: 48
                        ];
                    end
`ifdef SYNTHESIS
                    mixed_precision_dsp48e2_cascade_cell multiplier (
                        .clk(clk), .rst_n(rst_n),
                        .cascade_enable(!narrow_int8_mode && ((k_lane % 8) != 0)),
                        .selected_activation(selected_activations_flat[
                            (physical_m*32+k_lane)*27 +: 27
                        ]),
                        .selected_weight(selected_weights_flat[
                            (n_lane*32+k_lane)*18 +: 18
                        ]),
                        .cascade_in(cascade_input),
                        .result(dsp_result),
                        .cascade_out(dsp_cascade)
                    );
                    assign behavioral_product = 36'sd0;
`else
                    mixed_precision_selected_pair_multiplier multiplier (
                        .clk(clk), .rst_n(rst_n),
                        .selected_activation(selected_activations_flat[
                            (physical_m*32+k_lane)*27 +: 27
                        ]),
                        .selected_weight(selected_weights_flat[
                            (n_lane*32+k_lane)*18 +: 18
                        ]),
                        .attention_product(behavioral_product),
                        .mlp_offset_product_0(),
                        .mlp_offset_product_1()
                    );
                    assign dsp_result = {{12{behavioral_product[35]}},
                        behavioral_product};
                    assign dsp_cascade = 48'sd0;
`endif
                    assign dsp_results_flat[
                        ((physical_m*N_LANES+n_lane)*32+k_lane)*48 +: 48
                    ] = dsp_result;
                    assign dsp_cascade_flat[
                        ((physical_m*N_LANES+n_lane)*32+k_lane)*48 +: 48
                    ] = dsp_cascade;
                    assign low_mlp_product = $signed(dsp_result[17:0]);
                    assign high_mlp_floor = $signed(dsp_result) >>> 18;
                    assign high_mlp_product = high_mlp_floor
                        + (low_mlp_product < 0 ? 27'sd1 : 27'sd0);
                    assign mlp_product_0 = low_mlp_product;
                    assign mlp_product_1 = high_mlp_product[17:0];
                    assign mlp_products_flat[
                        (((2*physical_m)*N_LANES+n_lane)*32+k_lane)*18 +: 18
                    ] = mlp_product_0;
                    assign mlp_products_flat[
                        (((2*physical_m+1)*N_LANES+n_lane)*32+k_lane)*18 +: 18
                    ] = mlp_product_1;
                    if ((k_lane % 8) == 7) begin : group_result
`ifdef SYNTHESIS
                        assign attention_group_partials_flat[
                            ((physical_m*N_LANES+n_lane)*ATTENTION_GROUPS
                                +(k_lane/8))*48 +: 48
                        ] = dsp_result;
`else
                        assign attention_group_partials_flat[
                            ((physical_m*N_LANES+n_lane)*ATTENTION_GROUPS
                                +(k_lane/8))*48 +: 48
                        ] = 48'sd0;
`endif
                    end
                end
            end
        end
        for (n_lane = 0; n_lane < ATTENTION_OUTPUTS; n_lane = n_lane + 1) begin : pack_attention
            assign attention_accumulators_packed[
                n_lane*ATTENTION_ACC_WIDTH +: ATTENTION_ACC_WIDTH
            ] = attention_results[n_lane];
        end
        for (n_lane = 0; n_lane < MLP_OUTPUTS; n_lane = n_lane + 1) begin : pack_mlp
            assign mlp_accumulators_packed[
                n_lane*MLP_ACC_WIDTH +: MLP_ACC_WIDTH
            ] = mlp_results[n_lane];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_pipeline <= 6'b0;
            mode_pipeline <= 6'b0;
            clear_pipeline <= 6'b0;
            last_pipeline <= 6'b0;
            valid_out <= 1'b0;
            narrow_int8_mode_out <= 1'b0;
            tag_out <= {TAG_WIDTH{1'b0}};
            for (pipeline_index = 0; pipeline_index < 6;
                 pipeline_index = pipeline_index + 1)
                tag_pipeline[pipeline_index] <= {TAG_WIDTH{1'b0}};
            for (output_index = 0; output_index < ATTENTION_OUTPUTS;
                output_index = output_index + 1) begin
                attention_accumulators[output_index] <= 0;
                attention_results[output_index] <= 0;
                attention_dot_stage[output_index] <= 0;
                attention_dot_delay_0[output_index] <= 0;
                attention_dot_delay_1[output_index] <= 0;
                attention_dot_delay_2[output_index] <= 0;
                for (pair_index = 0; pair_index < 2; pair_index = pair_index + 1)
                    attention_partial_sum[output_index][pair_index] <= 0;
`ifndef SYNTHESIS
                for (pair_index = 0; pair_index < 16; pair_index = pair_index + 1)
                    behavioral_attention_sum_1[output_index][pair_index] <= 0;
                for (pair_index = 0; pair_index < 8; pair_index = pair_index + 1)
                    behavioral_attention_sum_2[output_index][pair_index] <= 0;
                for (pair_index = 0; pair_index < 4; pair_index = pair_index + 1)
                    behavioral_attention_sum_3[output_index][pair_index] <= 0;
`endif
            end
            for (output_index = 0; output_index < MLP_TREE_OUTPUTS;
                 output_index = output_index + 1) begin
                mlp_accumulators[output_index] <= 0;
                mlp_results[output_index] <= 0;
                mlp_dots[output_index] <= 0;
                for (pair_index = 0; pair_index < 16; pair_index = pair_index + 1)
                    mlp_sum_1[output_index][pair_index] <= 0;
                for (pair_index = 0; pair_index < 8; pair_index = pair_index + 1)
                    mlp_sum_2[output_index][pair_index] <= 0;
                for (pair_index = 0; pair_index < 4; pair_index = pair_index + 1)
                    mlp_sum_3[output_index][pair_index] <= 0;
                for (pair_index = 0; pair_index < 2; pair_index = pair_index + 1)
                    mlp_sum_4[output_index][pair_index] <= 0;
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

            for (output_index = 0; output_index < ATTENTION_OUTPUTS;
                 output_index = output_index + 1) begin
`ifdef SYNTHESIS
                for (pair_index = 0; pair_index < 2; pair_index = pair_index + 1)
                    attention_partial_sum[output_index][pair_index] <=
                        $signed(attention_group_partials_flat[
                            (output_index*ATTENTION_GROUPS+2*pair_index)*48
                                +: 48
                        ]) + $signed(attention_group_partials_flat[
                            (output_index*ATTENTION_GROUPS+2*pair_index+1)*48
                                +: 48
                        ]);
                attention_dot_stage[output_index] <=
                    $signed(attention_partial_sum[output_index][0])
                    + $signed(attention_partial_sum[output_index][1]);
                attention_dot_delay_0[output_index] <=
                    attention_dot_stage[output_index];
                attention_dot_delay_1[output_index] <=
                    attention_dot_delay_0[output_index];
                attention_dot_delay_2[output_index] <=
                    attention_dot_delay_1[output_index];
`else
                for (pair_index = 0; pair_index < 16; pair_index = pair_index + 1)
                    behavioral_attention_sum_1[output_index][pair_index] <=
                        $signed(dsp_results_flat[
                            (output_index*32+2*pair_index)*48 +: 36
                        ]) + $signed(dsp_results_flat[
                            (output_index*32+2*pair_index+1)*48 +: 36
                        ]);
                for (pair_index = 0; pair_index < 8; pair_index = pair_index + 1)
                    behavioral_attention_sum_2[output_index][pair_index] <=
                        $signed(behavioral_attention_sum_1[
                            output_index][2*pair_index
                        ]) + $signed(behavioral_attention_sum_1[
                            output_index][2*pair_index+1
                        ]);
                for (pair_index = 0; pair_index < 4; pair_index = pair_index + 1)
                    behavioral_attention_sum_3[output_index][pair_index] <=
                        $signed(behavioral_attention_sum_2[
                            output_index][2*pair_index
                        ]) + $signed(behavioral_attention_sum_2[
                            output_index][2*pair_index+1
                        ]);
                for (pair_index = 0; pair_index < 2; pair_index = pair_index + 1)
                    attention_partial_sum[output_index][pair_index] <=
                        $signed(behavioral_attention_sum_3[
                            output_index][2*pair_index
                        ]) + $signed(behavioral_attention_sum_3[
                            output_index][2*pair_index+1
                        ]);
                attention_dot_stage[output_index] <=
                    $signed(attention_partial_sum[output_index][0])
                    + $signed(attention_partial_sum[output_index][1]);
`endif
            end

            if (mode_pipeline[0]) begin
            for (output_index = 0; output_index < MLP_TREE_OUTPUTS;
                 output_index = output_index + 1) begin
                for (pair_index = 0; pair_index < 16; pair_index = pair_index + 1)
                    mlp_sum_1[output_index][pair_index] <=
                        $signed(mlp_products_flat[
                            (output_index*32+2*pair_index)*18 +: 18
                        ]) + $signed(mlp_products_flat[
                            (output_index*32+2*pair_index+1)*18 +: 18
                        ]);
                for (pair_index = 0; pair_index < 8; pair_index = pair_index + 1)
                    mlp_sum_2[output_index][pair_index] <=
                        $signed(mlp_sum_1[output_index][2*pair_index])
                        + $signed(mlp_sum_1[output_index][2*pair_index+1]);
                for (pair_index = 0; pair_index < 4; pair_index = pair_index + 1)
                    mlp_sum_3[output_index][pair_index] <=
                        $signed(mlp_sum_2[output_index][2*pair_index])
                        + $signed(mlp_sum_2[output_index][2*pair_index+1]);
                for (pair_index = 0; pair_index < 2; pair_index = pair_index + 1)
                    mlp_sum_4[output_index][pair_index] <=
                        $signed(mlp_sum_3[output_index][2*pair_index])
                        + $signed(mlp_sum_3[output_index][2*pair_index+1]);
                mlp_dots[output_index] <=
                    $signed(mlp_sum_4[output_index][0])
                    + $signed(mlp_sum_4[output_index][1]);
            end
            end

            if (narrow_int8_mode) begin
            for (output_index = 0; output_index < N_LANES;
                 output_index = output_index + 1) begin
                for (pair_index = 0; pair_index < 16; pair_index = pair_index + 1)
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

            valid_out <= 1'b0;
            if (valid_pipeline[5]) begin
                if (mode_pipeline[5]) begin
                    for (output_index = 0;
                         output_index < MLP_TREE_OUTPUTS;
                         output_index = output_index + 1) begin
                        extended_weight_sum = {{(ATTENTION_ACC_WIDTH-13){
                            weight_sum_delay[output_index % N_LANES][12]
                        }}, weight_sum_delay[output_index % N_LANES]};
                        if (clear_pipeline[5])
                            updated_mlp = $signed(mlp_dots[output_index])
                                - (extended_weight_sum <<< 7);
                        else
                            updated_mlp = mlp_accumulators[output_index]
                                + $signed(mlp_dots[output_index])
                                - (extended_weight_sum <<< 7);
                        mlp_accumulators[output_index] <= updated_mlp;
                        if (last_pipeline[5]) mlp_results[output_index] <= updated_mlp;
                    end
                end else begin
                    for (output_index = 0; output_index < ATTENTION_OUTPUTS;
                         output_index = output_index + 1) begin
                        if (clear_pipeline[5])
                            updated_attention =
`ifdef SYNTHESIS
                                $signed(attention_dot_delay_2[output_index]);
`else
                                $signed(attention_dot_stage[output_index]);
`endif
                        else
                            updated_attention = attention_accumulators[output_index]
`ifdef SYNTHESIS
                                + $signed(attention_dot_delay_2[output_index]);
`else
                                + $signed(attention_dot_stage[output_index]);
`endif
                        attention_accumulators[output_index] <= updated_attention;
                        if (last_pipeline[5])
                            attention_results[output_index] <= updated_attention;
                    end
                end
                if (last_pipeline[5]) begin
                    valid_out <= 1'b1;
                    narrow_int8_mode_out <= mode_pipeline[5];
                    tag_out <= tag_pipeline[5];
                end
            end
        end
    end

endmodule
