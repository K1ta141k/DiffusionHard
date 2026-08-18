`timescale 1ns/1ps

module philox_gumbel_farm_stream (
    input  logic               clk,
    input  logic               rst_n,

    input  logic               start_valid,
    output logic               start_ready,
    input  logic [6:0]         position_count,
    input  logic [15:0]        vocabulary_size,
    input  logic [31:0]        evaluation_id,
    input  logic [31:0]        stream_id,
    input  logic [31:0]        seed_low,
    input  logic [31:0]        seed_high,

    output logic               score_valid,
    input  logic               score_ready,
    output logic [1:0]         score_valid_mask,
    output logic [5:0]         score_position,
    output logic [15:0]        score_token_base,
    output logic signed [15:0] score0_q10,
    output logic signed [15:0] score1_q10,

    output logic               busy,
    output logic               done,
    output logic [1:0]         status
);

    localparam integer FIFO_DEPTH = 4;

    logic farm_start_ready;
    logic farm_block_valid;
    logic farm_block_ready;
    logic [5:0] farm_block_position;
    logic [15:0] farm_block_token_base;
    logic [2:0] farm_block_valid_words;
    logic [127:0] farm_block_random_words;
    logic farm_busy;
    logic farm_done;
    logic [1:0] farm_status;

    logic [127:0] fifo_words [0:FIFO_DEPTH-1];
    logic [5:0] fifo_position [0:FIFO_DEPTH-1];
    logic [15:0] fifo_token_base [0:FIFO_DEPTH-1];
    logic [2:0] fifo_valid_words [0:FIFO_DEPTH-1];
    logic [1:0] fifo_write_pointer;
    logic [1:0] fifo_read_pointer;
    logic [2:0] fifo_count;
    logic fifo_push;
    logic fifo_pop;

    logic current_valid;
    logic current_phase;
    logic [127:0] current_words;
    logic [5:0] current_position;
    logic [15:0] current_token_base;
    logic [2:0] current_valid_words;

    logic pair_input_valid;
    logic pair_input_ready;
    logic [31:0] pair_word0;
    logic [31:0] pair_word1;
    logic [1:0] pair_valid_mask;
    logic [15:0] pair_token_base;
    logic gumbel_output_valid;
    logic metadata_valid1;
    logic metadata_valid2;
    logic [1:0] metadata_mask1;
    logic [1:0] metadata_mask2;
    logic [5:0] metadata_position1;
    logic [5:0] metadata_position2;
    logic [15:0] metadata_token_base1;
    logic [15:0] metadata_token_base2;
    logic drain_pending;
    logic pipeline_empty;
    logic score_output_stage_ready;

    assign pipeline_empty = fifo_count == 0 && !current_valid &&
        !metadata_valid1 && !metadata_valid2 && !gumbel_output_valid;
    assign start_ready = farm_start_ready && pipeline_empty && !drain_pending;
    assign farm_block_ready = fifo_count < FIFO_DEPTH;
    assign fifo_push = farm_block_valid && farm_block_ready;
    assign fifo_pop = !current_valid && fifo_count != 0;
    assign busy = farm_busy || drain_pending || !pipeline_empty;
    assign status = farm_status;
    assign score_output_stage_ready = !score_valid || score_ready;

    philox4x32_farm farm (
        .clk,
        .rst_n,
        .start_valid(start_valid && start_ready),
        .start_ready(farm_start_ready),
        .position_count,
        .vocabulary_size,
        .evaluation_id,
        .stream_id,
        .seed_low,
        .seed_high,
        .block_valid(farm_block_valid),
        .block_ready(farm_block_ready),
        .block_position(farm_block_position),
        .block_token_base(farm_block_token_base),
        .block_valid_words(farm_block_valid_words),
        .block_random_words(farm_block_random_words),
        .busy(farm_busy),
        .done(farm_done),
        .status(farm_status)
    );

    always_comb begin
        pair_input_valid = current_valid;
        if (!current_phase) begin
            pair_word0 = current_words[31:0];
            pair_word1 = current_words[63:32];
            pair_valid_mask[0] = current_valid_words >= 1;
            pair_valid_mask[1] = current_valid_words >= 2;
            pair_token_base = current_token_base;
        end else begin
            pair_word0 = current_words[95:64];
            pair_word1 = current_words[127:96];
            pair_valid_mask[0] = current_valid_words >= 3;
            pair_valid_mask[1] = current_valid_words >= 4;
            pair_token_base = current_token_base + 2;
        end
    end

    gumbel_q10_dual gumbel (
        .clk,
        .rst_n,
        .input_valid(pair_input_valid),
        .input_ready(pair_input_ready),
        .input_word0(pair_word0),
        .input_word1(pair_word1),
        .output_valid(gumbel_output_valid),
        .output_ready(score_ready),
        .output_score0_q10(score0_q10),
        .output_score1_q10(score1_q10)
    );

    assign score_valid = gumbel_output_valid && metadata_valid2;
    assign score_valid_mask = metadata_mask2;
    assign score_position = metadata_position2;
    assign score_token_base = metadata_token_base2;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            fifo_write_pointer <= 0;
            fifo_read_pointer <= 0;
            fifo_count <= 0;
            current_valid <= 1'b0;
            current_phase <= 1'b0;
            current_words <= 0;
            current_position <= 0;
            current_token_base <= 0;
            current_valid_words <= 0;
            metadata_valid1 <= 1'b0;
            metadata_valid2 <= 1'b0;
            metadata_mask1 <= 0;
            metadata_mask2 <= 0;
            metadata_position1 <= 0;
            metadata_position2 <= 0;
            metadata_token_base1 <= 0;
            metadata_token_base2 <= 0;
            drain_pending <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;

            case ({fifo_push, fifo_pop})
                2'b10: fifo_count <= fifo_count + 1'b1;
                2'b01: fifo_count <= fifo_count - 1'b1;
                default: fifo_count <= fifo_count;
            endcase
            if (fifo_push) begin
                fifo_words[fifo_write_pointer] <= farm_block_random_words;
                fifo_position[fifo_write_pointer] <= farm_block_position;
                fifo_token_base[fifo_write_pointer] <= farm_block_token_base;
                fifo_valid_words[fifo_write_pointer] <= farm_block_valid_words;
                fifo_write_pointer <= fifo_write_pointer + 1'b1;
            end
            if (fifo_pop) begin
                current_words <= fifo_words[fifo_read_pointer];
                current_position <= fifo_position[fifo_read_pointer];
                current_token_base <= fifo_token_base[fifo_read_pointer];
                current_valid_words <= fifo_valid_words[fifo_read_pointer];
                current_valid <= 1'b1;
                current_phase <= 1'b0;
                fifo_read_pointer <= fifo_read_pointer + 1'b1;
            end else if (current_valid && pair_input_ready) begin
                if (!current_phase && current_valid_words > 2) begin
                    current_phase <= 1'b1;
                end else begin
                    current_valid <= 1'b0;
                    current_phase <= 1'b0;
                end
            end

            if (score_output_stage_ready) begin
                metadata_valid2 <= metadata_valid1;
                if (metadata_valid1) begin
                    metadata_mask2 <= metadata_mask1;
                    metadata_position2 <= metadata_position1;
                    metadata_token_base2 <= metadata_token_base1;
                end
            end
            if (pair_input_ready) begin
                metadata_valid1 <= pair_input_valid;
                if (pair_input_valid) begin
                    metadata_mask1 <= pair_valid_mask;
                    metadata_position1 <= current_position;
                    metadata_token_base1 <= pair_token_base;
                end
            end

            if (farm_done) begin
                drain_pending <= 1'b1;
            end
            if (drain_pending && pipeline_empty) begin
                drain_pending <= 1'b0;
                done <= 1'b1;
            end
        end
    end

endmodule
