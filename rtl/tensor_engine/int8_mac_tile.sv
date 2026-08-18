`timescale 1ns/1ps

module int8_mac_tile #(
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 8,
    parameter integer K_LANES = 32,
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH = 32
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire clear_accumulators,
    input  wire last_k_tile,
    input  wire [M_LANES*K_LANES*DATA_WIDTH-1:0] activations_packed,
    input  wire [N_LANES*K_LANES*DATA_WIDTH-1:0] weights_packed,
    output reg  valid_out,
    output wire [M_LANES*N_LANES*ACC_WIDTH-1:0] accumulators_packed
);

    localparam integer OUTPUT_LANES = M_LANES * N_LANES;

    reg signed [ACC_WIDTH-1:0] accumulators [0:OUTPUT_LANES-1];
    reg signed [ACC_WIDTH-1:0] dot_products [0:OUTPUT_LANES-1];
    reg signed [DATA_WIDTH-1:0] activation_value;
    reg signed [DATA_WIDTH-1:0] weight_value;
    reg signed [2*DATA_WIDTH-1:0] product_value;
    integer m_index;
    integer n_index;
    integer k_index;
    integer output_index;

    always @* begin
        for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
            for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                output_index = m_index * N_LANES + n_index;
                dot_products[output_index] = {ACC_WIDTH{1'b0}};
                for (k_index = 0; k_index < K_LANES; k_index = k_index + 1) begin
                    activation_value = $signed(
                        activations_packed[
                            (m_index*K_LANES+k_index)*DATA_WIDTH +: DATA_WIDTH
                        ]
                    );
                    weight_value = $signed(
                        weights_packed[
                            (n_index*K_LANES+k_index)*DATA_WIDTH +: DATA_WIDTH
                        ]
                    );
                    product_value = activation_value * weight_value;
                    dot_products[output_index] =
                        dot_products[output_index] + product_value;
                end
            end
        end
    end

    genvar output_lane;
    generate
        for (output_lane = 0; output_lane < OUTPUT_LANES; output_lane = output_lane + 1) begin : pack_outputs
            assign accumulators_packed[
                output_lane*ACC_WIDTH +: ACC_WIDTH
            ] = accumulators[output_lane];
        end
    endgenerate

    integer accumulator_index;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for (
                accumulator_index = 0;
                accumulator_index < OUTPUT_LANES;
                accumulator_index = accumulator_index + 1
            ) begin
                accumulators[accumulator_index] <= {ACC_WIDTH{1'b0}};
            end
        end else begin
            valid_out <= 1'b0;
            if (valid_in) begin
                for (
                    accumulator_index = 0;
                    accumulator_index < OUTPUT_LANES;
                    accumulator_index = accumulator_index + 1
                ) begin
                    if (clear_accumulators) begin
                        accumulators[accumulator_index] <=
                            dot_products[accumulator_index];
                    end else begin
                        accumulators[accumulator_index] <=
                            accumulators[accumulator_index]
                            + dot_products[accumulator_index];
                    end
                end
                valid_out <= last_k_tile;
            end
        end
    end

endmodule
