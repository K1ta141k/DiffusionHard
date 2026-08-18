`timescale 1ns/1ps

module tb_int8_mac_tile_pipelined;
    localparam integer M_LANES = 2;
    localparam integer N_LANES = 2;
    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH = 32;
    localparam integer TAG_WIDTH = 4;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg valid_in = 1'b0;
    reg clear_accumulators = 1'b0;
    reg last_k_tile = 1'b0;
    reg [TAG_WIDTH-1:0] tag_in = '0;
    reg [M_LANES*32*DATA_WIDTH-1:0] activations_packed = '0;
    reg [N_LANES*32*DATA_WIDTH-1:0] weights_packed = '0;
    wire valid_out;
    wire [TAG_WIDTH-1:0] tag_out;
    wire [M_LANES*N_LANES*ACC_WIDTH-1:0] accumulators_packed;

    integer expected [0:1][0:M_LANES-1][0:N_LANES-1];
    integer output_count = 0;
    integer group_index;
    integer tile_index;
    integer m_index;
    integer n_index;
    integer k_index;
    integer activation_integer;
    integer weight_integer;
    reg signed [DATA_WIDTH-1:0] activation_value;
    reg signed [DATA_WIDTH-1:0] weight_value;
    reg signed [ACC_WIDTH-1:0] actual_value;

    int8_mac_tile_pipelined #(
        .M_LANES(M_LANES),
        .N_LANES(N_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .TAG_WIDTH(TAG_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .clear_accumulators(clear_accumulators),
        .last_k_tile(last_k_tile),
        .tag_in(tag_in),
        .activations_packed(activations_packed),
        .weights_packed(weights_packed),
        .valid_out(valid_out),
        .tag_out(tag_out),
        .accumulators_packed(accumulators_packed)
    );

    always #5 clk = ~clk;

    task drive_tile;
        input integer group_number;
        input integer tile_number;
        begin
            @(negedge clk);
            activations_packed = '0;
            weights_packed = '0;
            for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
                for (k_index = 0; k_index < 32; k_index = k_index + 1) begin
                    activation_integer = ((group_number+1)*7 + m_index*5 + tile_number*3 + k_index) % 19 - 9;
                    activation_value = activation_integer;
                    activations_packed[(m_index*32+k_index)*DATA_WIDTH +: DATA_WIDTH] = activation_value;
                end
            end
            for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                for (k_index = 0; k_index < 32; k_index = k_index + 1) begin
                    weight_integer = ((group_number+1)*3 + n_index*7 + tile_number*5 + k_index*2) % 17 - 8;
                    weight_value = weight_integer;
                    weights_packed[(n_index*32+k_index)*DATA_WIDTH +: DATA_WIDTH] = weight_value;
                end
            end
            for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
                for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                    for (k_index = 0; k_index < 32; k_index = k_index + 1) begin
                        activation_value = activations_packed[(m_index*32+k_index)*DATA_WIDTH +: DATA_WIDTH];
                        weight_value = weights_packed[(n_index*32+k_index)*DATA_WIDTH +: DATA_WIDTH];
                        expected[group_number][m_index][n_index] =
                            expected[group_number][m_index][n_index]
                            + $signed(activation_value) * $signed(weight_value);
                    end
                end
            end
            valid_in = 1'b1;
            clear_accumulators = (tile_number == 0);
            last_k_tile = (tile_number == 1);
            tag_in = group_number;
            @(posedge clk);
            #1;
        end
    endtask

    always @(posedge clk) begin
        #1;
        if (valid_out) begin
            if (tag_out !== output_count[TAG_WIDTH-1:0]) begin
                $fatal(1, "tag_out expected %0d got %0d", output_count, tag_out);
            end
            for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
                for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                    actual_value = $signed(
                        accumulators_packed[(m_index*N_LANES+n_index)*ACC_WIDTH +: ACC_WIDTH]
                    );
                    if (actual_value !== expected[output_count][m_index][n_index]) begin
                        $fatal(
                            1,
                            "group %0d accumulator[%0d,%0d] expected %0d got %0d",
                            output_count,
                            m_index,
                            n_index,
                            expected[output_count][m_index][n_index],
                            actual_value
                        );
                    end
                end
            end
            output_count = output_count + 1;
        end
    end

    initial begin
        for (group_index = 0; group_index < 2; group_index = group_index + 1) begin
            for (m_index = 0; m_index < M_LANES; m_index = m_index + 1) begin
                for (n_index = 0; n_index < N_LANES; n_index = n_index + 1) begin
                    expected[group_index][m_index][n_index] = 0;
                end
            end
        end

        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        drive_tile(0, 0);
        drive_tile(0, 1);
        drive_tile(1, 0);
        drive_tile(1, 1);

        @(negedge clk);
        valid_in = 1'b0;
        clear_accumulators = 1'b0;
        last_k_tile = 1'b0;

        repeat (12) @(posedge clk);
        #1;
        if (output_count !== 2) begin
            $fatal(1, "expected two completed groups, got %0d", output_count);
        end
        if (valid_out !== 1'b0) begin
            $fatal(1, "valid_out did not return low");
        end

        $display("tb_int8_mac_tile_pipelined: PASS");
        $finish;
    end
endmodule
