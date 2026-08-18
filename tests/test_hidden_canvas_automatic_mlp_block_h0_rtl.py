from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from diffusion_accel.fixed_mlp import (
    _integer_matmul,
    _load_tensors,
    _quantize_weight_per_output,
    hardware_gelu_q10,
    quantize_up_activation_fixed,
)
from diffusion_accel.fixed_norm import fixed_layer_norm_q12


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"


def _packed_hex(values: list[int], width: int) -> str:
    mask = (1 << width) - 1
    packed = sum(
        (value & mask) << (index * width)
        for index, value in enumerate(values)
    )
    return f"{packed:0{(len(values) * width + 3) // 4}x}"


def _mixed_packed_hex(fields: list[tuple[list[int], int]]) -> str:
    packed = 0
    offset = 0
    for values, width in fields:
        mask = (1 << width) - 1
        for value in values:
            packed |= (value & mask) << offset
            offset += width
    return f"{packed:0{(offset + 3) // 4}x}"


@pytest.mark.skipif(
    os.environ.get("DIFFUSION_ACCEL_RUN_REAL_MLP_RTL") != "1"
    and os.environ.get("DIFFUSION_ACCEL_RUN_FULL_MLP_RTL") != "1"
    and os.environ.get("DIFFUSION_ACCEL_RUN_WIDE_MLP_RTL") != "1"
    and os.environ.get("DIFFUSION_ACCEL_RUN_WIDE_BASELINE_MLP_RTL") != "1",
    reason="set a captured H0 MLP RTL environment gate",
)
def test_automatic_mlp_block_matches_captured_h0(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    artifact_path = PACKAGE / "mlp_interstage_int8.safetensors"
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    if not artifact_path.is_file() or not golden_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    goldens = _load_tensors(
        golden_path,
        ["folded.block_00.after_attention", "folded.block_00.norm2_unaffine"],
    )
    weights = _load_tensors(
        weights_path,
        ["block_00.mlp_up.weight"],
    )
    artifacts = _load_tensors(
        artifact_path,
        [
            "block_00.up_smoothing_reciprocal_q15",
            "block_00.up_output_factor",
            "block_00.up_bias_q10",
            "block_00.interstage_multiplier",
            "block_00.down_weight",
            "block_00.down_output_multiplier",
            "block_00.down_bias_q10",
        ],
    )

    full_shape = os.environ.get("DIFFUSION_ACCEL_RUN_FULL_MLP_RTL") == "1"
    wide_shape = os.environ.get("DIFFUSION_ACCEL_RUN_WIDE_MLP_RTL") == "1"
    wide_baseline = (
        os.environ.get("DIFFUSION_ACCEL_RUN_WIDE_BASELINE_MLP_RTL") == "1"
    )
    mlp_m_lanes = 8 if wide_shape else 4
    token_count = 64 if full_shape else 8 if wide_shape or wide_baseline else 4
    token_groups = token_count // 4
    mac_token_groups = token_count // mlp_m_lanes
    down_output_size = (
        768 if full_shape else 12 if wide_shape or wide_baseline else 6
    )
    down_output_tiles = down_output_size // 6
    group_width = max(1, (token_groups - 1).bit_length())
    client_tag_width = 1 + 10 + max(
        1, (mac_token_groups - 1).bit_length()
    )
    mac_module = (
        "int8_mac_tile_packed_m8_pipelined"
        if wide_shape
        else "int8_mac_tile_pipelined"
    )
    mac_parameters = (
        ".N_LANES(6),.TAG_WIDTH(CLIENT_TAG_WIDTH+1)"
        if wide_shape
        else ".M_LANES(4),.N_LANES(6),.TAG_WIDTH(CLIENT_TAG_WIDTH+1)"
    )

    residual = goldens[
        "folded.block_00.after_attention"
    ][0, :token_count]
    calibration_norm = goldens["folded.block_00.norm2_unaffine"][0]
    up_weight = weights["block_00.mlp_up.weight"]
    smoothing = (
        calibration_norm.abs().amax(dim=0).clamp_min(1e-8).pow(0.75)
        / up_weight.abs().amax(dim=0).clamp_min(1e-8).pow(0.25)
    ).clamp_min(1e-8)
    fixed_norm, _, _ = fixed_layer_norm_q12(residual)
    activation_q, token_factors, activation_tensors, _ = (
        quantize_up_activation_fixed(fixed_norm, smoothing)
    )
    reciprocals = artifacts["block_00.up_smoothing_reciprocal_q15"].to(
        torch.int64
    )
    assert torch.equal(
        activation_tensors["reciprocal"].to(torch.int64), reciprocals
    )

    up_weight_q, _ = _quantize_weight_per_output(
        up_weight * smoothing[None, :], 8
    )
    up_accumulators = _integer_matmul(
        activation_q, up_weight_q.t(), 8
    ).to(torch.int64)
    up_output_factors = artifacts["block_00.up_output_factor"].to(torch.int64)
    up_biases = artifacts["block_00.up_bias_q10"].to(torch.int64)
    up_multipliers = (
        token_factors.to(torch.int64)[:, None]
        * up_output_factors[None, :]
        + 128
    ) >> 8
    up_products = up_accumulators * up_multipliers
    up_q = torch.sign(up_products) * (
        (up_products.abs() + (1 << 19)) >> 20
    )
    up_q = (up_q + up_biases[None, :]).clamp(-(1 << 15), (1 << 15) - 1)
    _, gelu_q = hardware_gelu_q10(up_q)

    interstage_multipliers = artifacts[
        "block_00.interstage_multiplier"
    ].to(torch.int64)
    interstage_products = gelu_q.to(torch.int64) * interstage_multipliers[None, :]
    interstage_q = torch.sign(interstage_products) * (
        (interstage_products.abs() + (1 << 19)) >> 20
    )
    interstage_q = interstage_q.clamp(-127, 127).to(torch.int8)
    down_weight_q = artifacts["block_00.down_weight"].to(torch.int8)
    down_accumulators = _integer_matmul(
        interstage_q, down_weight_q[:down_output_size].t(), 8
    ).to(torch.int64)
    down_multipliers = artifacts[
        "block_00.down_output_multiplier"
    ][:down_output_size].to(torch.int64)
    down_biases = artifacts[
        "block_00.down_bias_q10"
    ][:down_output_size].to(torch.int64)
    down_products = down_accumulators * down_multipliers[None, :]
    down_q = torch.sign(down_products) * (
        (down_products.abs() + (1 << 19)) >> 20
    )
    down_q = (down_q + down_biases[None, :]).clamp(
        -(1 << 23), (1 << 23) - 1
    )
    residual_q = torch.round(residual.double() * 1024.0).to(torch.int64)
    residual_q = residual_q.clamp(-(1 << 23), (1 << 23) - 1)
    expected_q = (residual_q[:, :down_output_size] + down_q).clamp(
        -(1 << 23), (1 << 23) - 1
    )

    residual_hex = tmp_path / "automatic_mlp_h0_residual.hex"
    residual_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(residual_q[group * 4 + token, tile * 6 + lane])
                    for token in range(4)
                    for lane in range(6)
                ],
                24,
            )
            for group in range(token_groups)
            for tile in range(128)
        )
        + "\n",
        encoding="utf-8",
    )
    reciprocal_hex = tmp_path / "automatic_mlp_h0_reciprocal.hex"
    reciprocal_hex.write_text(
        "\n".join(f"{int(value):05x}" for value in reciprocals) + "\n",
        encoding="utf-8",
    )
    up_weight_hex = tmp_path / "automatic_mlp_h0_up_weight.hex"
    up_weight_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(up_weight_q[output_tile * 6 + lane, k_tile * 32 + k])
                    for lane in range(6)
                    for k in range(32)
                ],
                8,
            )
            for output_tile in range(512)
            for k_tile in range(24)
        )
        + "\n",
        encoding="utf-8",
    )
    up_metadata_hex = tmp_path / "automatic_mlp_h0_up_metadata.hex"
    up_metadata_hex.write_text(
        "\n".join(
            _mixed_packed_hex(
                [
                    (
                        [
                            int(up_output_factors[output_tile * 6 + lane])
                            for lane in range(6)
                        ],
                        18,
                    ),
                    (
                        [
                            int(up_biases[output_tile * 6 + lane])
                            for lane in range(6)
                        ],
                        32,
                    ),
                    (
                        [
                            int(interstage_multipliers[output_tile * 6 + lane])
                            for lane in range(6)
                        ],
                        24,
                    ),
                ]
            )
            for output_tile in range(512)
        )
        + "\n",
        encoding="utf-8",
    )
    down_weight_hex = tmp_path / "automatic_mlp_h0_down_weight.hex"
    down_weight_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(
                        down_weight_q[
                            output_tile * 6 + lane, k_tile * 32 + k
                        ]
                    )
                    for lane in range(6)
                    for k in range(32)
                ],
                8,
            )
            for output_tile in range(down_output_tiles)
            for k_tile in range(96)
        )
        + "\n",
        encoding="utf-8",
    )
    down_metadata_hex = tmp_path / "automatic_mlp_h0_down_metadata.hex"
    down_metadata_hex.write_text(
        "\n".join(
            _mixed_packed_hex(
                [
                    (
                        [
                            int(down_multipliers[output_tile * 6 + lane])
                            for token in range(4)
                            for lane in range(6)
                        ],
                        24,
                    ),
                    (
                        [
                            int(down_biases[output_tile * 6 + lane])
                            for token in range(4)
                            for lane in range(6)
                        ],
                        32,
                    ),
                ]
            )
            for output_tile in range(down_output_tiles)
        )
        + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "automatic_mlp_h0_expected.hex"
    expected_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(
                        expected_q[
                            group * 4 + token, output_tile * 6 + lane
                        ]
                    )
                    for token in range(4)
                    for lane in range(6)
                ],
                24,
            )
            for output_tile in range(down_output_tiles)
            for group in range(token_groups)
        )
        + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_hidden_canvas_automatic_mlp_block_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_hidden_canvas_automatic_mlp_block_h0;
  localparam integer CLIENT_TAG_WIDTH={client_tag_width};
  localparam integer TOKEN_GROUPS={token_groups};
  localparam integer DOWN_OUTPUT_TILES={down_output_tiles};
  reg clk=0,rst_n=0,block_start=0,canvas_data_valid=0;
  reg [575:0] canvas_data=0;
  wire block_start_ready,busy,done,canvas_read_valid;
  wire [3:0] canvas_group; wire [6:0] canvas_tile;
  wire [9:0] reciprocal_channel,requested_up_tile,requested_down_tile;
  wire requested_up_bank,requested_down_bank;
  wire up_weight_ready,up_metadata_ready,down_weight_ready,down_metadata_ready;
  wire output_valid; wire [9:0] output_tile;
  wire [{group_width - 1}:0] output_group;
  wire [575:0] outputs;
  wire array_valid,array_clear,array_last,array_response_valid;
  wire [CLIENT_TAG_WIDTH:0] array_tag,array_response_tag;
  wire [{mlp_m_lanes * 32 * 8 - 1}:0] array_activations;
  wire [1535:0] array_weights;
  wire [{mlp_m_lanes * 6 * 32 - 1}:0] array_accumulators;
  reg [575:0] residual_mem [0:{token_groups * 128 - 1}];
  reg [17:0] reciprocal_mem [0:767];
  reg [1535:0] up_weight_mem [0:12287]; reg [443:0] up_meta_mem [0:511];
  reg [1535:0] down_weight_mem [0:{down_output_tiles * 96 - 1}];
  reg [1343:0] down_meta_mem [0:{down_output_tiles - 1}];
  reg [575:0] expected_mem [0:{down_output_tiles * token_groups - 1}];
  reg [4:0] up_k=0; reg [6:0] down_k=0;
  integer canvas_reads=0,up_weights=0,up_meta=0,down_weights=0;
  integer down_meta=0,array_requests=0,outputs_seen=0,cycles=0;
  wire [17:0] reciprocal=reciprocal_mem[reciprocal_channel];
  wire [1535:0] up_weight=up_weight_mem[requested_up_tile*24+up_k];
  wire [443:0] up_meta_data=up_meta_mem[requested_up_tile];
  wire [1535:0] down_weight=
    down_weight_mem[requested_down_tile*96+down_k];
  wire [1343:0] down_meta_data=down_meta_mem[requested_down_tile];
  hidden_canvas_automatic_mlp_block #(
    .TOKENS({token_count}),.DOWN_INPUT_SIZE(3072),
    .DOWN_OUTPUT_SIZE({down_output_size}),
    .MLP_M_LANES({mlp_m_lanes}),
    .CLIENT_TAG_WIDTH(CLIENT_TAG_WIDTH)
  ) dut(
    .clk(clk),.rst_n(rst_n),.block_start(block_start),
    .block_start_ready(block_start_ready),
    .smoothing_reciprocal_q15(reciprocal),
    .smoothing_reciprocal_channel(reciprocal_channel),.busy(busy),.done(done),
    .canvas_read_valid(canvas_read_valid),.canvas_read_group(canvas_group),
    .canvas_read_output_tile(canvas_tile),.canvas_read_data_valid(canvas_data_valid),
    .canvas_read_q10_packed(canvas_data),
    .requested_up_output_tile(requested_up_tile),
    .requested_up_bank(requested_up_bank),.up_weight_stream_valid(1'b1),
    .up_weight_stream_ready(up_weight_ready),.up_weight_stream_data(up_weight),
    .up_metadata_stream_valid(1'b1),.up_metadata_stream_ready(up_metadata_ready),
    .up_metadata_stream_data(up_meta_data),
    .requested_down_output_tile(requested_down_tile),
    .requested_down_bank(requested_down_bank),.down_weight_stream_valid(1'b1),
    .down_weight_stream_ready(down_weight_ready),
    .down_weight_stream_data(down_weight),.down_metadata_stream_valid(1'b1),
    .down_metadata_stream_ready(down_metadata_ready),
    .down_metadata_stream_data(down_meta_data),
    .output_valid(output_valid),.output_tile(output_tile),
    .output_group(output_group),.outputs_packed(outputs),
    .array_request_valid(array_valid),.array_request_clear(array_clear),
    .array_request_last(array_last),.array_request_tag(array_tag),
    .array_request_activations(array_activations),
    .array_request_weights(array_weights),
    .array_response_valid(array_response_valid),
    .array_response_tag(array_response_tag),
    .array_response_accumulators(array_accumulators));
  {mac_module} #(
    {mac_parameters}
  ) mac(
    .clk(clk),.rst_n(rst_n),.valid_in(array_valid),
    .clear_accumulators(array_clear),.last_k_tile(array_last),.tag_in(array_tag),
    .activations_packed(array_activations),.weights_packed(array_weights),
    .valid_out(array_response_valid),.tag_out(array_response_tag),
    .accumulators_packed(array_accumulators));
  always #2 clk=~clk;
  always @(posedge clk) begin
    cycles=cycles+1;canvas_data_valid<=canvas_read_valid;
    if(canvas_read_valid) begin
      canvas_data<=residual_mem[canvas_group*128+canvas_tile];
      canvas_reads=canvas_reads+1;
    end
    if(up_weight_ready) begin
      up_weights=up_weights+1;up_k<=(up_k==23)?0:up_k+1;
    end
    if(up_metadata_ready) up_meta=up_meta+1;
    if(down_weight_ready) begin
      down_weights=down_weights+1;down_k<=(down_k==95)?0:down_k+1;
    end
    if(down_metadata_ready) down_meta=down_meta+1;
    if(array_valid) array_requests=array_requests+1;
    #1;
    if(output_valid) begin
      if(output_tile!==(outputs_seen/TOKEN_GROUPS) ||
         output_group!==(outputs_seen%TOKEN_GROUPS) ||
         outputs!==expected_mem[output_tile*TOKEN_GROUPS+output_group])
        $fatal(1,"captured H0 automatic MLP output mismatch");
      outputs_seen=outputs_seen+1;
    end
  end
  initial begin
    $readmemh("{residual_hex}",residual_mem);
    $readmemh("{reciprocal_hex}",reciprocal_mem);
    $readmemh("{up_weight_hex}",up_weight_mem);
    $readmemh("{up_metadata_hex}",up_meta_mem);
    $readmemh("{down_weight_hex}",down_weight_mem);
    $readmemh("{down_metadata_hex}",down_meta_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;block_start=1;
    wait(block_start_ready);@(posedge clk);@(negedge clk);block_start=0;
    wait(done);repeat(4) @(posedge clk);
    if(canvas_reads!={token_groups * (384 + down_output_tiles)} ||
       up_weights!=12288 || up_meta!=512 ||
       down_weights!={down_output_tiles * 96} ||
       down_meta!={down_output_tiles} ||
       array_requests!={512 * mac_token_groups * 24 + down_output_tiles * mac_token_groups * 96} ||
       outputs_seen!={down_output_tiles * token_groups} || busy)
      $fatal(1,"captured H0 automatic MLP count mismatch");
    $display("tb_hidden_canvas_automatic_mlp_block_h0: PASS cycles=%0d outputs=%0d requests=%0d",
      cycles,outputs_seen,array_requests);
    $finish;
  end
  initial begin repeat({2_000_000 if full_shape else 100_000}) @(posedge clk);
    $fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    sources = [
        "unsigned_divider_iterative.sv",
        "unsigned_sqrt_iterative.sv",
        "layer_norm_q12_group.sv",
        "mlp_up_activation_quantizer.sv",
        "layer_norm_mlp_up_activation_frontend.sv",
        "hidden_canvas_group_replay.sv",
        "hidden_canvas_mlp_frontend.sv",
        "mlp_token_pair_adapters.sv",
        "int8_mac_tile_pipelined.sv",
        "int8_shared_weight_pair_multiplier.sv",
        "int8_mac_tile_packed_m8_pipelined.sv",
        "mlp_tile_pingpong_controller.sv",
        "wide_synchronous_uram.sv",
        "fixed_requantize.sv",
        "fixed_requantize_vector_serial.sv",
        "gelu_q10_lut_scalar_bram.sv",
        "mlp_up_postprocess_serial.sv",
        "mlp_up_pingpong_pipeline.sv",
        "smoothquant_int8_vector_serial.sv",
        "mlp_interstage_tile_bridge_bram.sv",
        "mlp_interstage_pipeline.sv",
        "mlp_up_to_down_activation_pipeline.sv",
        "residual_add_saturating.sv",
        "mlp_down_pingpong_pipeline.sv",
        "mlp_shared_up_down_pipeline.sv",
        "hidden_canvas_mlp_shared_pipeline.sv",
        "hidden_canvas_mlp_wide_shared_pipeline.sv",
        "mlp_tile_load_sequencer.sv",
        "hidden_canvas_residual_load_sequencer.sv",
        "mlp_block_controller.sv",
        "hidden_canvas_automatic_mlp_block.sv",
    ]
    build = tmp_path / "tb_hidden_canvas_automatic_mlp_block_h0"
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s",
            "tb_hidden_canvas_automatic_mlp_block_h0", "-o", str(build),
            *(str(RTL / source) for source in sources), str(testbench),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        ["vvp", str(build)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800 if full_shape else 900,
    )
    if os.environ.get("DIFFUSION_ACCEL_SHOW_LONG_RTL_OUTPUT") == "1":
        print(run_result.stdout, end="")
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_hidden_canvas_automatic_mlp_block_h0: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) < (2_000_000 if full_shape else 100_000)
