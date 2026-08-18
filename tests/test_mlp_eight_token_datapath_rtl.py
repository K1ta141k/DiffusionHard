from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def _run_shape(tmp_path: Path, m_lanes: int) -> tuple[int, int]:
    token_groups = 8 // m_lanes
    client_tag_width = 12
    build = tmp_path / f"tb_mlp_m{m_lanes}"
    testbench = tmp_path / f"tb_mlp_m{m_lanes}.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_mlp_m{m_lanes};
  localparam integer M_LANES={m_lanes};
  localparam integer TOKEN_GROUPS={token_groups};
  localparam integer CLIENT_TAG_WIDTH={client_tag_width};
  reg clk=0,rst_n=0;
  reg up_activation_valid=0,up_weight_valid=0,up_metadata_valid=0;
  reg up_token_factor_valid=0,up_start=0;
  reg [0:0] up_activation_group=0,up_token_factor_group=0;
  reg [4:0] up_activation_k=0,up_weight_k=0;
  reg [9:0] up_output_tile=0;
  reg down_weight_valid=0,down_metadata_valid=0,down_residual_valid=0;
  reg down_start=0; reg [0:0] down_residual_group=0;
  reg [4:0] down_weight_k=0; reg [9:0] down_output_tile=0;
  wire up_weight_ready,up_metadata_ready,up_token_factor_ready;
  wire up_start_ready,up_busy,up_tile_done,up_all_done;
  wire down_weight_ready,down_metadata_ready,down_residual_ready;
  wire down_start_ready,down_busy,output_valid,down_done,output_bank;
  wire [9:0] output_tile; wire [0:0] output_group;
  wire [M_LANES*6*24-1:0] outputs;
  wire array_valid,array_clear,array_last,array_response_valid;
  wire [CLIENT_TAG_WIDTH:0] array_tag,array_response_tag;
  wire [M_LANES*32*8-1:0] array_activations;
  wire [6*32*8-1:0] array_weights;
  wire [M_LANES*6*32-1:0] array_accumulators;
  integer group_index,tile,k,requests=0,outputs_seen=0,busy_cycles=0;

  mlp_shared_up_down_pipeline #(
    .TOKENS(8),.UP_INPUT_SIZE(768),.DOWN_INPUT_SIZE(768),
    .DOWN_OUTPUT_SIZE(12),.M_LANES(M_LANES),.N_LANES(6),
    .UP_POSTPROCESS_PARALLEL4({1 if m_lanes == 8 else 0}),
    .CLIENT_TAG_WIDTH(CLIENT_TAG_WIDTH)
  ) dut(
    .clk(clk),.rst_n(rst_n),
    .up_activation_load_valid(up_activation_valid),
    .up_activation_load_group(up_activation_group),
    .up_activation_load_k_tile(up_activation_k),
    .up_activation_load_data({m_lanes * 32 * 8}'b0),
    .up_weight_load_valid(up_weight_valid),.up_weight_load_bank(0),
    .up_weight_load_k_tile(up_weight_k),
    .up_weight_load_data(1536'b0),.up_weight_load_ready(up_weight_ready),
    .up_metadata_load_valid(up_metadata_valid),.up_metadata_load_bank(0),
    .up_metadata_output_factors(108'b0),.up_metadata_biases(192'b0),
    .up_metadata_interstage_multipliers(144'b0),
    .up_metadata_load_ready(up_metadata_ready),
    .up_token_factor_load_valid(up_token_factor_valid),
    .up_token_factor_load_group(up_token_factor_group),
    .up_token_factor_load_factors({m_lanes * 16}'b0),
    .up_token_factor_load_ready(up_token_factor_ready),
    .up_start(up_start),.up_start_bank(0),
    .up_start_output_tile(up_output_tile),.up_start_ready(up_start_ready),
    .up_busy(up_busy),.up_tile_done(up_tile_done),
    .up_all_activations_done(up_all_done),
    .down_weight_load_valid(down_weight_valid),.down_weight_load_bank(0),
    .down_weight_load_k_tile(down_weight_k),
    .down_weight_load_data(1536'b0),.down_weight_load_ready(down_weight_ready),
    .down_metadata_load_valid(down_metadata_valid),
    .down_metadata_load_bank(0),
    .down_metadata_multipliers({m_lanes * 6 * 24}'b0),
    .down_metadata_biases({m_lanes * 6 * 32}'b0),
    .down_metadata_load_ready(down_metadata_ready),
    .down_residual_load_valid(down_residual_valid),
    .down_residual_load_group(down_residual_group),
    .down_residual_load_output_tile(down_output_tile),
    .down_residual_load_data({m_lanes * 6 * 24}'b0),
    .down_residual_load_ready(down_residual_ready),
    .down_start(down_start),.down_start_bank(0),
    .down_start_output_tile(down_output_tile),
    .down_start_ready(down_start_ready),.down_busy(down_busy),
    .output_valid(output_valid),.output_bank(output_bank),
    .output_tile(output_tile),.output_group(output_group),
    .outputs_packed(outputs),.down_done(down_done),
    .array_request_valid(array_valid),.array_request_clear(array_clear),
    .array_request_last(array_last),.array_request_tag(array_tag),
    .array_request_activations(array_activations),
    .array_request_weights(array_weights),
    .array_response_valid(array_response_valid),
    .array_response_tag(array_response_tag),
    .array_response_accumulators(array_accumulators));

  int8_mac_tile_pipelined #(
    .M_LANES(M_LANES),.N_LANES(6),.TAG_WIDTH(CLIENT_TAG_WIDTH+1)
  ) mac(
    .clk(clk),.rst_n(rst_n),.valid_in(array_valid),
    .clear_accumulators(array_clear),.last_k_tile(array_last),
    .tag_in(array_tag),.activations_packed(array_activations),
    .weights_packed(array_weights),.valid_out(array_response_valid),
    .tag_out(array_response_tag),.accumulators_packed(array_accumulators));

  always #2 clk=~clk;
  always @(posedge clk) begin
    if(array_valid) requests=requests+1;
    if(up_busy || down_busy) busy_cycles=busy_cycles+1;
    #1;
    if(output_valid) begin
      if(output_bank!==0 || output_tile!==(outputs_seen/TOKEN_GROUPS) ||
         output_group!==(outputs_seen%TOKEN_GROUPS) || outputs!==0)
        $fatal(1,"eight-token MLP output mismatch");
      outputs_seen=outputs_seen+1;
    end
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(group_index=0;group_index<TOKEN_GROUPS;group_index=group_index+1) begin
      for(k=0;k<24;k=k+1) begin
        @(negedge clk);up_activation_valid=1;
        up_activation_group=group_index;up_activation_k=k;
      end
      @(negedge clk);up_activation_valid=0;up_token_factor_valid=1;
      up_token_factor_group=group_index;
      @(posedge clk);#1;
      if(!up_token_factor_ready) $fatal(1,"token factor load blocked");
      @(negedge clk);up_token_factor_valid=0;
    end
    for(tile=0;tile<128;tile=tile+1) begin
      for(k=0;k<24;k=k+1) begin
        wait(up_weight_ready);@(negedge clk);up_weight_valid=1;up_weight_k=k;
      end
      @(negedge clk);up_weight_valid=0;up_metadata_valid=1;
      wait(up_metadata_ready);@(posedge clk);@(negedge clk);up_metadata_valid=0;
      up_output_tile=tile;up_start=1;wait(up_start_ready);
      @(posedge clk);@(negedge clk);up_start=0;wait(up_tile_done);
    end
    wait(up_all_done);
    for(tile=0;tile<2;tile=tile+1) begin
      down_output_tile=tile;
      for(k=0;k<24;k=k+1) begin
        wait(down_weight_ready);@(negedge clk);down_weight_valid=1;
        down_weight_k=k;
      end
      @(negedge clk);down_weight_valid=0;down_metadata_valid=1;
      wait(down_metadata_ready);@(posedge clk);@(negedge clk);
      down_metadata_valid=0;
      for(group_index=0;group_index<TOKEN_GROUPS;group_index=group_index+1) begin
        down_residual_group=group_index;down_residual_valid=1;
        wait(down_residual_ready);@(posedge clk);@(negedge clk);
        down_residual_valid=0;
      end
      down_start=1;wait(down_start_ready);
      @(posedge clk);@(negedge clk);down_start=0;wait(down_done);
      @(negedge clk);wait(!down_done);
    end
    wait(outputs_seen=={2 * token_groups});
    repeat(4) @(posedge clk);#1;
    if(requests!={128 * token_groups * 24 + 2 * token_groups * 24})
      $fatal(1,"request count %0d",requests);
    if(outputs_seen!={2 * token_groups})
      $fatal(1,"output count %0d",outputs_seen);
    $display("tb_mlp_m{m_lanes}: PASS busy_cycles=%0d requests=%0d",
      busy_cycles,requests);
    $finish;
  end
  initial begin repeat(40000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    sources = [
        "int8_mac_tile_pipelined.sv",
        "mlp_tile_pingpong_controller.sv",
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
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s", f"tb_mlp_m{m_lanes}",
            "-o", str(build), *(str(RTL / source) for source in sources),
            str(testbench),
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
        timeout=120,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    match = re.search(r"busy_cycles=(\d+) requests=(\d+)", run_result.stdout)
    assert match is not None, run_result.stdout
    return int(match.group(1)), int(match.group(2))


def test_eight_token_mlp_halves_the_mac_schedule(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    four_cycles, four_requests = _run_shape(tmp_path, 4)
    eight_cycles, eight_requests = _run_shape(tmp_path, 8)
    print(
        f"MLP_TOKEN_LANES m4_cycles={four_cycles} m8_cycles={eight_cycles} "
        f"m4_requests={four_requests} m8_requests={eight_requests}"
    )
    assert eight_requests * 2 == four_requests
    assert eight_cycles < four_cycles * 0.55
