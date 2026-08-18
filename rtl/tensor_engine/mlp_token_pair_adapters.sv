`timescale 1ns/1ps

module mlp_token_pair_input_adapter (
    input  wire clk,
    input  wire rst_n,
    input  wire activation_valid_in,
    input  wire [3:0] activation_group_in,
    input  wire [4:0] activation_k_tile_in,
    input  wire [4*32*8-1:0] activation_data_in,
    input  wire token_factor_valid_in,
    input  wire [3:0] token_factor_group_in,
    input  wire [4*16-1:0] token_factors_in,
    output reg  activation_valid_out,
    output reg  [2:0] activation_group_out,
    output reg  [4:0] activation_k_tile_out,
    output reg  [8*32*8-1:0] activation_data_out,
    output reg  token_factor_valid_out,
    output reg  [2:0] token_factor_group_out,
    output reg  [8*16-1:0] token_factors_out
);

    reg [4*32*8-1:0] even_activation_tiles [0:23];
    reg [4*16-1:0] even_token_factors;

    always @(posedge clk) begin
        if (!rst_n) begin
            activation_valid_out <= 1'b0;
            activation_group_out <= 3'b0;
            activation_k_tile_out <= 5'b0;
            activation_data_out <= {8*32*8{1'b0}};
            token_factor_valid_out <= 1'b0;
            token_factor_group_out <= 3'b0;
            token_factors_out <= {8*16{1'b0}};
            even_token_factors <= {4*16{1'b0}};
        end else begin
            activation_valid_out <= 1'b0;
            token_factor_valid_out <= 1'b0;
            if (activation_valid_in) begin
                if (!activation_group_in[0]) begin
                    even_activation_tiles[activation_k_tile_in] <=
                        activation_data_in;
                end else begin
                    activation_valid_out <= 1'b1;
                    activation_group_out <= activation_group_in[3:1];
                    activation_k_tile_out <= activation_k_tile_in;
                    activation_data_out <= {
                        activation_data_in,
                        even_activation_tiles[activation_k_tile_in]
                    };
                end
            end
            if (token_factor_valid_in) begin
                if (!token_factor_group_in[0]) begin
                    even_token_factors <= token_factors_in;
                end else begin
                    token_factor_valid_out <= 1'b1;
                    token_factor_group_out <= token_factor_group_in[3:1];
                    token_factors_out <= {token_factors_in, even_token_factors};
                end
            end
        end
    end

endmodule

module mlp_token_pair_residual_adapter #(
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire [3:0] group_in,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile_in,
    input  wire [4*6*24-1:0] data_in,
    output reg  valid_out,
    output reg  [2:0] group_out,
    output reg  [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile_out,
    output reg  [8*6*24-1:0] data_out
);

    reg [4*6*24-1:0] even_data;
    reg [OUTPUT_TILE_TAG_WIDTH-1:0] even_output_tile;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            group_out <= 3'b0;
            output_tile_out <= {OUTPUT_TILE_TAG_WIDTH{1'b0}};
            data_out <= {8*6*24{1'b0}};
            even_data <= {4*6*24{1'b0}};
            even_output_tile <= {OUTPUT_TILE_TAG_WIDTH{1'b0}};
        end else begin
            valid_out <= 1'b0;
            if (valid_in) begin
                if (!group_in[0]) begin
                    even_data <= data_in;
                    even_output_tile <= output_tile_in;
                end else begin
`ifndef SYNTHESIS
                    if (output_tile_in != even_output_tile)
                        $error("paired MLP residual groups changed output tile");
`endif
                    valid_out <= 1'b1;
                    group_out <= group_in[3:1];
                    output_tile_out <= output_tile_in;
                    data_out <= {data_in, even_data};
                end
            end
        end
    end

endmodule

module mlp_token_pair_output_serializer #(
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire done_in,
    input  wire [2:0] group_in,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile_in,
    input  wire [8*6*24-1:0] data_in,
    output reg  valid_out,
    output reg  done_out,
    output reg  [3:0] group_out,
    output reg  [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile_out,
    output reg  [4*6*24-1:0] data_out
);

    reg active;
    reg pending_done;
    reg [2:0] pending_group;
    reg [OUTPUT_TILE_TAG_WIDTH-1:0] pending_output_tile;
    reg [4*6*24-1:0] pending_high_data;

    always @(posedge clk) begin
        if (!rst_n) begin
            active <= 1'b0;
            pending_done <= 1'b0;
            pending_group <= 3'b0;
            pending_output_tile <= {OUTPUT_TILE_TAG_WIDTH{1'b0}};
            pending_high_data <= {4*6*24{1'b0}};
            valid_out <= 1'b0;
            done_out <= 1'b0;
            group_out <= 4'b0;
            output_tile_out <= {OUTPUT_TILE_TAG_WIDTH{1'b0}};
            data_out <= {4*6*24{1'b0}};
        end else begin
            valid_out <= 1'b0;
            done_out <= 1'b0;
            if (active) begin
                valid_out <= 1'b1;
                done_out <= pending_done;
                group_out <= {pending_group, 1'b1};
                output_tile_out <= pending_output_tile;
                data_out <= pending_high_data;
                active <= 1'b0;
            end else if (valid_in) begin
                valid_out <= 1'b1;
                group_out <= {group_in, 1'b0};
                output_tile_out <= output_tile_in;
                data_out <= data_in[0 +: 4*6*24];
                pending_high_data <= data_in[4*6*24 +: 4*6*24];
                pending_done <= done_in;
                pending_group <= group_in;
                pending_output_tile <= output_tile_in;
                active <= 1'b1;
            end
`ifndef SYNTHESIS
            if (active && valid_in)
                $error("wide MLP output arrived before serializer drained");
`endif
        end
    end

endmodule
