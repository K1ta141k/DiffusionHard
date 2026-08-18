`timescale 1ns/1ps

module philox4x32_iterative (
    input  logic         clk,
    input  logic         rst_n,

    input  logic         input_valid,
    output logic         input_ready,
    input  logic [31:0]  input_c0,
    input  logic [31:0]  input_c1,
    input  logic [31:0]  input_c2,
    input  logic [31:0]  input_c3,
    input  logic [31:0]  input_k0,
    input  logic [31:0]  input_k1,

    output logic         output_valid,
    input  logic         output_ready,
    output logic [127:0] output_words,
    output logic         busy
);

    localparam logic [31:0] PHILOX_M0 = 32'hD2511F53;
    localparam logic [31:0] PHILOX_M1 = 32'hCD9E8D57;
    localparam logic [31:0] PHILOX_W0 = 32'h9E3779B9;
    localparam logic [31:0] PHILOX_W1 = 32'hBB67AE85;

    logic [31:0] c0_reg;
    logic [31:0] c1_reg;
    logic [31:0] c2_reg;
    logic [31:0] c3_reg;
    logic [31:0] k0_reg;
    logic [31:0] k1_reg;
    logic [3:0] round_reg;

    logic [63:0] state_product0;
    logic [63:0] state_product1;
    logic [31:0] state_next_c0;
    logic [31:0] state_next_c1;
    logic [31:0] state_next_c2;
    logic [31:0] state_next_c3;

    logic [63:0] input_product0;
    logic [63:0] input_product1;
    logic [31:0] input_next_c0;
    logic [31:0] input_next_c1;
    logic [31:0] input_next_c2;
    logic [31:0] input_next_c3;

    assign state_product0 = PHILOX_M0 * c0_reg;
    assign state_product1 = PHILOX_M1 * c2_reg;
    assign state_next_c0 = state_product1[63:32] ^ c1_reg ^ k0_reg;
    assign state_next_c1 = state_product1[31:0];
    assign state_next_c2 = state_product0[63:32] ^ c3_reg ^ k1_reg;
    assign state_next_c3 = state_product0[31:0];

    assign input_product0 = PHILOX_M0 * input_c0;
    assign input_product1 = PHILOX_M1 * input_c2;
    assign input_next_c0 = input_product1[63:32] ^ input_c1 ^ input_k0;
    assign input_next_c1 = input_product1[31:0];
    assign input_next_c2 = input_product0[63:32] ^ input_c3 ^ input_k1;
    assign input_next_c3 = input_product0[31:0];

    assign input_ready = !busy && (!output_valid || output_ready);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            c0_reg <= 0;
            c1_reg <= 0;
            c2_reg <= 0;
            c3_reg <= 0;
            k0_reg <= 0;
            k1_reg <= 0;
            round_reg <= 0;
            output_words <= 0;
            output_valid <= 1'b0;
            busy <= 1'b0;
        end else begin
            if (output_valid && output_ready) begin
                output_valid <= 1'b0;
            end

            if (input_valid && input_ready) begin
                c0_reg <= input_next_c0;
                c1_reg <= input_next_c1;
                c2_reg <= input_next_c2;
                c3_reg <= input_next_c3;
                k0_reg <= input_k0 + PHILOX_W0;
                k1_reg <= input_k1 + PHILOX_W1;
                round_reg <= 4'd1;
                busy <= 1'b1;
            end else if (busy) begin
                c0_reg <= state_next_c0;
                c1_reg <= state_next_c1;
                c2_reg <= state_next_c2;
                c3_reg <= state_next_c3;
                if (round_reg == 4'd9) begin
                    output_words <= {
                        state_next_c3,
                        state_next_c2,
                        state_next_c1,
                        state_next_c0
                    };
                    output_valid <= 1'b1;
                    busy <= 1'b0;
                end else begin
                    k0_reg <= k0_reg + PHILOX_W0;
                    k1_reg <= k1_reg + PHILOX_W1;
                    round_reg <= round_reg + 1'b1;
                end
            end
        end
    end

endmodule
