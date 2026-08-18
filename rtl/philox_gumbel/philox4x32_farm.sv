`timescale 1ns/1ps

module philox4x32_farm (
    input  logic         clk,
    input  logic         rst_n,

    input  logic         start_valid,
    output logic         start_ready,
    input  logic [6:0]   position_count,
    input  logic [15:0]  vocabulary_size,
    input  logic [31:0]  evaluation_id,
    input  logic [31:0]  stream_id,
    input  logic [31:0]  seed_low,
    input  logic [31:0]  seed_high,

    output logic         block_valid,
    input  logic         block_ready,
    output logic [5:0]   block_position,
    output logic [15:0]  block_token_base,
    output logic [2:0]   block_valid_words,
    output logic [127:0] block_random_words,

    output logic         busy,
    output logic         done,
    output logic [1:0]   status
);

    localparam logic [1:0] STATUS_OK = 2'd0;
    localparam logic [1:0] STATUS_INVALID_POSITION_COUNT = 2'd1;
    localparam logic [1:0] STATUS_INVALID_VOCABULARY_SIZE = 2'd2;

    logic [6:0] position_count_reg;
    logic [15:0] vocabulary_size_reg;
    logic [31:0] evaluation_id_reg;
    logic [31:0] stream_id_reg;
    logic [31:0] seed_low_reg;
    logic [31:0] seed_high_reg;
    logic [5:0] dispatch_position;
    logic [15:0] dispatch_token_base;
    logic [1:0] dispatch_pointer;
    logic all_dispatched;
    logic [4:0] blocks_in_flight;

    logic [3:0] core_input_valid;
    logic [3:0] core_input_ready;
    logic [3:0] core_output_valid;
    logic [3:0] core_output_ready;
    logic [3:0] core_busy;
    logic [127:0] core_output_words [0:3];
    logic [5:0] metadata_position [0:3];
    logic [15:0] metadata_token_base [0:3];
    logic [2:0] metadata_valid_words [0:3];
    logic dispatch_selected;
    logic [1:0] dispatch_index;
    logic output_selected;
    logic [1:0] output_index;
    logic dispatch_fire;
    logic output_fire;
    logic [16:0] dispatch_token_end;
    integer offset;
    integer candidate_index;

    assign start_ready = !busy;
    assign dispatch_token_end = {1'b0, dispatch_token_base} + 17'd4;

    always_comb begin
        core_input_valid = 4'b0000;
        dispatch_selected = 1'b0;
        dispatch_index = dispatch_pointer;
        for (offset = 0; offset < 4; offset = offset + 1) begin
            candidate_index = (dispatch_pointer + offset) & 3;
            if (!dispatch_selected && core_input_ready[candidate_index]) begin
                dispatch_selected = 1'b1;
                dispatch_index = candidate_index[1:0];
            end
        end
        if (busy && !all_dispatched && dispatch_selected) begin
            core_input_valid[dispatch_index] = 1'b1;
        end
    end
    assign dispatch_fire = |(core_input_valid & core_input_ready);

    always_comb begin
        output_selected = 1'b0;
        output_index = 0;
        for (offset = 0; offset < 4; offset = offset + 1) begin
            if (!output_selected && core_output_valid[offset]) begin
                output_selected = 1'b1;
                output_index = offset[1:0];
            end
        end
        block_valid = output_selected;
        block_position = metadata_position[output_index];
        block_token_base = metadata_token_base[output_index];
        block_valid_words = metadata_valid_words[output_index];
        block_random_words = core_output_words[output_index];
        core_output_ready = 4'b0000;
        if (output_selected) begin
            core_output_ready[output_index] = block_ready;
        end
    end
    assign output_fire = block_valid && block_ready;

    genvar core_index;
    generate
        for (core_index = 0; core_index < 4; core_index = core_index + 1) begin : cores
            philox4x32_iterative core (
                .clk,
                .rst_n,
                .input_valid(core_input_valid[core_index]),
                .input_ready(core_input_ready[core_index]),
                .input_c0({18'd0, dispatch_token_base[15:2]}),
                .input_c1({26'd0, dispatch_position}),
                .input_c2(evaluation_id_reg),
                .input_c3(stream_id_reg),
                .input_k0(seed_low_reg),
                .input_k1(seed_high_reg),
                .output_valid(core_output_valid[core_index]),
                .output_ready(core_output_ready[core_index]),
                .output_words(core_output_words[core_index]),
                .busy(core_busy[core_index])
            );
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            position_count_reg <= 0;
            vocabulary_size_reg <= 0;
            evaluation_id_reg <= 0;
            stream_id_reg <= 0;
            seed_low_reg <= 0;
            seed_high_reg <= 0;
            dispatch_position <= 0;
            dispatch_token_base <= 0;
            dispatch_pointer <= 0;
            all_dispatched <= 1'b0;
            blocks_in_flight <= 0;
            busy <= 1'b0;
            done <= 1'b0;
            status <= STATUS_OK;
            metadata_position[0] <= 0;
            metadata_position[1] <= 0;
            metadata_position[2] <= 0;
            metadata_position[3] <= 0;
            metadata_token_base[0] <= 0;
            metadata_token_base[1] <= 0;
            metadata_token_base[2] <= 0;
            metadata_token_base[3] <= 0;
            metadata_valid_words[0] <= 0;
            metadata_valid_words[1] <= 0;
            metadata_valid_words[2] <= 0;
            metadata_valid_words[3] <= 0;
        end else begin
            done <= 1'b0;
            if (!busy) begin
                if (start_valid && start_ready) begin
                    status <= STATUS_OK;
                    if (position_count == 0 || position_count > 64) begin
                        done <= 1'b1;
                        status <= STATUS_INVALID_POSITION_COUNT;
                    end else if (vocabulary_size == 0) begin
                        done <= 1'b1;
                        status <= STATUS_INVALID_VOCABULARY_SIZE;
                    end else begin
                        position_count_reg <= position_count;
                        vocabulary_size_reg <= vocabulary_size;
                        evaluation_id_reg <= evaluation_id;
                        stream_id_reg <= stream_id;
                        seed_low_reg <= seed_low;
                        seed_high_reg <= seed_high;
                        dispatch_position <= 0;
                        dispatch_token_base <= 0;
                        dispatch_pointer <= 0;
                        all_dispatched <= 1'b0;
                        blocks_in_flight <= 0;
                        busy <= 1'b1;
                    end
                end
            end else begin
                case ({dispatch_fire, output_fire})
                    2'b10: blocks_in_flight <= blocks_in_flight + 1'b1;
                    2'b01: blocks_in_flight <= blocks_in_flight - 1'b1;
                    default: blocks_in_flight <= blocks_in_flight;
                endcase

                if (dispatch_fire) begin
                    metadata_position[dispatch_index] <= dispatch_position;
                    metadata_token_base[dispatch_index] <= dispatch_token_base;
                    metadata_valid_words[dispatch_index] <=
                        (vocabulary_size_reg - dispatch_token_base >= 4)
                            ? 3'd4
                            : vocabulary_size_reg - dispatch_token_base;
                    dispatch_pointer <= dispatch_index + 1'b1;
                    if (
                        dispatch_token_end < {1'b0, vocabulary_size_reg}
                    ) begin
                        dispatch_token_base <= dispatch_token_base + 4;
                    end else if (dispatch_position + 1'b1 < position_count_reg) begin
                        dispatch_token_base <= 0;
                        dispatch_position <= dispatch_position + 1'b1;
                    end else begin
                        all_dispatched <= 1'b1;
                    end
                end

                if (
                    output_fire && all_dispatched && blocks_in_flight == 1
                ) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end

endmodule
