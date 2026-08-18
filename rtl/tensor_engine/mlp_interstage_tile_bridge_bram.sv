`timescale 1ns/1ps

module mlp_interstage_tile_bridge_bram #(
    parameter integer TOKENS = 64,
    parameter integer M_LANES = 4,
    parameter integer N_LANES = 6,
    parameter integer INPUT_SIZE = 3072,
    parameter integer OUTPUT_TILE_TAG_WIDTH = 10,
    parameter integer GROUP_WIDTH = ((TOKENS / M_LANES) <= 1)
        ? 1 : $clog2(TOKENS / M_LANES),
    parameter integer K_TILE_WIDTH = ((INPUT_SIZE / 32) <= 1)
        ? 1 : $clog2(INPUT_SIZE / 32),
    parameter integer PAIR_WIDTH = (N_LANES <= 2)
        ? 1 : $clog2((N_LANES + 1) / 2)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    input  wire [OUTPUT_TILE_TAG_WIDTH-1:0] output_tile_in,
    input  wire [GROUP_WIDTH-1:0] group_in,
    input  wire [M_LANES*N_LANES*8-1:0] values_packed,
    output reg  activation_load_valid,
    output reg  [GROUP_WIDTH-1:0] activation_load_group,
    output reg  [K_TILE_WIDTH-1:0] activation_load_k_tile,
    output reg  [M_LANES*32*8-1:0] activation_load_data,
    output wire done
);

    localparam integer TOKEN_GROUPS = TOKENS / M_LANES;
    localparam integer WORD_WIDTH = M_LANES * 8;
    localparam integer BANK_DEPTH = TOKEN_GROUPS * 16;
    localparam integer BANK_ADDR_WIDTH = (BANK_DEPTH <= 1)
        ? 1 : $clog2(BANK_DEPTH);
    localparam integer TILE_BITS = M_LANES * 32 * 8;
    localparam [1:0] ENGINE_IDLE = 2'd0;
    localparam [1:0] ENGINE_READ = 2'd1;
    localparam [1:0] ENGINE_EMIT = 2'd2;
    localparam [1:0] ENGINE_SPILL = 2'd3;

    reg [WORD_WIDTH-1:0] even_bank [0:BANK_DEPTH-1];
    reg [WORD_WIDTH-1:0] odd_bank [0:BANK_DEPTH-1];
    reg [K_TILE_WIDTH-1:0] current_k_tiles [0:TOKEN_GROUPS-1];

    reg write_active;
    reg [PAIR_WIDTH-1:0] write_pair;
    reg [GROUP_WIDTH-1:0] write_group;
    reg [4:0] write_base_offset;
    reg [M_LANES*N_LANES*8-1:0] write_values;

    reg [1:0] engine_state;
    reg [4:0] read_issue_count;
    reg read_data_valid;
    reg [3:0] read_data_index;
    reg [WORD_WIDTH-1:0] read_even_data;
    reg [WORD_WIDTH-1:0] read_odd_data;
    reg [GROUP_WIDTH-1:0] completion_group;
    reg [K_TILE_WIDTH-1:0] completion_k_tile;
    reg [4:0] completion_base_offset;
    reg [M_LANES*N_LANES*8-1:0] completion_values;
    reg [TILE_BITS-1:0] tile_assembly;
    reg [1:0] spill_pair;
    reg [2:0] spill_channel_count;

    wire [WORD_WIDTH-1:0] input_channel_words [0:N_LANES-1];
    wire [M_LANES*N_LANES*8-1:0] selected_write_values =
        (engine_state == ENGINE_SPILL) ? completion_values : write_values;
    reg memory_write_enable;
    reg [BANK_ADDR_WIDTH-1:0] memory_write_address;
    reg [WORD_WIDTH-1:0] memory_write_even_data;
    reg [WORD_WIDTH-1:0] memory_write_odd_data;
    wire memory_read_enable = (engine_state == ENGINE_READ)
        && (read_issue_count < 16);
    wire [BANK_ADDR_WIDTH-1:0] memory_read_address =
        (completion_group << 4) + read_issue_count[3:0];

    reg [OUTPUT_TILE_TAG_WIDTH+$clog2(N_LANES)-1:0] base_channel;
    reg [K_TILE_WIDTH-1:0] base_k_tile;
    reg [4:0] base_offset;
    reg completes_tile;
    integer token_index;
    integer channel_index;
    integer group_index;
    integer pair_index;
    genvar generated_channel;
    genvar generated_token;

    generate
        for (generated_channel = 0; generated_channel < N_LANES;
             generated_channel = generated_channel + 1) begin : channel_words
            for (generated_token = 0; generated_token < M_LANES;
                 generated_token = generated_token + 1) begin : token_bytes
                assign input_channel_words[generated_channel][
                    generated_token*8 +: 8
                ] = selected_write_values[
                        (generated_token*N_LANES + generated_channel)*8 +: 8
                    ];
            end
        end
    endgenerate

    assign done = activation_load_valid
        && (activation_load_group == TOKEN_GROUPS-1)
        && (activation_load_k_tile == (INPUT_SIZE / 32)-1);

    always @* begin
        base_channel = (output_tile_in << 2) + (output_tile_in << 1);
        base_k_tile = base_channel >> 5;
        base_offset = base_channel[4:0];
        completes_tile = (base_offset + N_LANES) >= 32;
    end

    always @* begin
        memory_write_enable = 1'b0;
        memory_write_address = {BANK_ADDR_WIDTH{1'b0}};
        memory_write_even_data = {WORD_WIDTH{1'b0}};
        memory_write_odd_data = {WORD_WIDTH{1'b0}};
        if (engine_state == ENGINE_SPILL) begin
            memory_write_enable = 1'b1;
            memory_write_address = (completion_group << 4) + spill_pair;
            memory_write_even_data = input_channel_words[
                32 - completion_base_offset + spill_pair*2
            ];
            memory_write_odd_data = input_channel_words[
                32 - completion_base_offset + spill_pair*2 + 1
            ];
        end else if (write_active) begin
            memory_write_enable = 1'b1;
            memory_write_address = (write_group << 4)
                + (write_base_offset >> 1) + write_pair;
            memory_write_even_data = input_channel_words[write_pair*2];
            memory_write_odd_data = input_channel_words[write_pair*2 + 1];
        end
    end

    always @(posedge clk) begin
        if (memory_write_enable) begin
            even_bank[memory_write_address] <= memory_write_even_data;
            odd_bank[memory_write_address] <= memory_write_odd_data;
        end
        if (memory_read_enable) begin
            read_even_data <= even_bank[memory_read_address];
            read_odd_data <= odd_bank[memory_read_address];
        end
    end

    initial begin
        if ((M_LANES != 4 && M_LANES != 8) || N_LANES != 6) begin
            $error("BRAM bridge supports four or eight tokens by six channels");
        end
        if (TOKENS % M_LANES != 0 || INPUT_SIZE % 32 != 0
            || INPUT_SIZE % N_LANES != 0) begin
            $error("invalid fixed MLP dimensions");
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            write_active <= 1'b0;
            write_pair <= {PAIR_WIDTH{1'b0}};
            write_group <= {GROUP_WIDTH{1'b0}};
            write_base_offset <= 5'd0;
            write_values <= {M_LANES*N_LANES*8{1'b0}};
            engine_state <= ENGINE_IDLE;
            read_issue_count <= 5'd0;
            read_data_valid <= 1'b0;
            read_data_index <= 4'd0;
            completion_group <= {GROUP_WIDTH{1'b0}};
            completion_k_tile <= {K_TILE_WIDTH{1'b0}};
            completion_base_offset <= 5'd0;
            completion_values <= {M_LANES*N_LANES*8{1'b0}};
            tile_assembly <= {TILE_BITS{1'b0}};
            spill_pair <= 2'd0;
            spill_channel_count <= 3'd0;
            activation_load_valid <= 1'b0;
            activation_load_group <= {GROUP_WIDTH{1'b0}};
            activation_load_k_tile <= {K_TILE_WIDTH{1'b0}};
            activation_load_data <= {TILE_BITS{1'b0}};
            for (group_index = 0; group_index < TOKEN_GROUPS;
                 group_index = group_index + 1) begin
                current_k_tiles[group_index] <= {K_TILE_WIDTH{1'b0}};
            end
        end else begin
            activation_load_valid <= 1'b0;

            if (write_active) begin
                if (write_pair == (N_LANES/2)-1) begin
                    write_active <= 1'b0;
                    write_pair <= {PAIR_WIDTH{1'b0}};
                end else begin
                    write_pair <= write_pair + 1'b1;
                end
            end

            if (valid_in) begin
`ifndef SYNTHESIS
                if (write_active) $error("interstage write scheduler overflow");
                if (base_k_tile != current_k_tiles[group_in])
                    $error("interstage output tiles arrived out of order");
`endif
                if (completes_tile) begin
`ifndef SYNTHESIS
                    if (engine_state != ENGINE_IDLE)
                        $error("interstage completed-tile engine overflow");
`endif
                    completion_group <= group_in;
                    completion_k_tile <= base_k_tile;
                    completion_base_offset <= base_offset;
                    completion_values <= values_packed;
                    spill_channel_count <= base_offset + N_LANES - 32;
                    tile_assembly <= {TILE_BITS{1'b0}};
                    read_issue_count <= 5'd0;
                    read_data_valid <= 1'b0;
                    engine_state <= ENGINE_READ;
                    current_k_tiles[group_in] <= base_k_tile + 1'b1;
                end else begin
                    write_active <= 1'b1;
                    write_pair <= {PAIR_WIDTH{1'b0}};
                    write_group <= group_in;
                    write_base_offset <= base_offset;
                    write_values <= values_packed;
                end
            end

            if (engine_state == ENGINE_READ) begin
                if (read_issue_count < 16) begin
                    read_data_index <= read_issue_count[3:0];
                    read_data_valid <= 1'b1;
                    read_issue_count <= read_issue_count + 1'b1;
                end else begin
                    read_data_valid <= 1'b0;
                end
                if (read_data_valid) begin
                    for (pair_index = 0; pair_index < 16;
                         pair_index = pair_index + 1) begin
                        if (read_data_index == pair_index) begin
                            for (token_index = 0; token_index < M_LANES;
                                 token_index = token_index + 1) begin
                                tile_assembly[
                                    (token_index*32 + pair_index*2)*8 +: 8
                                ] <= read_even_data[token_index*8 +: 8];
                                tile_assembly[
                                    (token_index*32 + pair_index*2 + 1)*8 +: 8
                                ] <= read_odd_data[token_index*8 +: 8];
                            end
                        end
                    end
                    if (read_data_index == 15) begin
                        case (completion_base_offset)
                            5'd26: begin
                                for (token_index = 0; token_index < M_LANES;
                                     token_index = token_index + 1) begin
                                    for (channel_index = 0;
                                         channel_index < 6;
                                         channel_index = channel_index + 1)
                                        tile_assembly[
                                            (token_index*32 + 26
                                             + channel_index)*8 +: 8
                                        ] <= completion_values[
                                            (token_index*N_LANES
                                             + channel_index)*8 +: 8
                                        ];
                                end
                            end
                            5'd28: begin
                                for (token_index = 0; token_index < M_LANES;
                                     token_index = token_index + 1) begin
                                    for (channel_index = 0;
                                         channel_index < 4;
                                         channel_index = channel_index + 1)
                                        tile_assembly[
                                            (token_index*32 + 28
                                             + channel_index)*8 +: 8
                                        ] <= completion_values[
                                            (token_index*N_LANES
                                             + channel_index)*8 +: 8
                                        ];
                                end
                            end
                            5'd30: begin
                                for (token_index = 0; token_index < M_LANES;
                                     token_index = token_index + 1) begin
                                    for (channel_index = 0;
                                         channel_index < 2;
                                         channel_index = channel_index + 1)
                                        tile_assembly[
                                            (token_index*32 + 30
                                             + channel_index)*8 +: 8
                                        ] <= completion_values[
                                            (token_index*N_LANES
                                             + channel_index)*8 +: 8
                                        ];
                                end
                            end
                            default: begin end
                        endcase
                        engine_state <= ENGINE_EMIT;
                    end
                end
            end else if (engine_state == ENGINE_EMIT) begin
                activation_load_valid <= 1'b1;
                activation_load_group <= completion_group;
                activation_load_k_tile <= completion_k_tile;
                activation_load_data <= tile_assembly;
                spill_pair <= 2'd0;
                if (spill_channel_count == 0) begin
                    engine_state <= ENGINE_IDLE;
                end else begin
                    engine_state <= ENGINE_SPILL;
                end
            end else if (engine_state == ENGINE_SPILL) begin
                if ((spill_pair + 1)*2 >= spill_channel_count) begin
                    engine_state <= ENGINE_IDLE;
                    spill_pair <= 2'd0;
                end else begin
                    spill_pair <= spill_pair + 1'b1;
                end
            end
        end
    end

endmodule
