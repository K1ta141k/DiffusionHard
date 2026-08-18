from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.block_image import read_block_image_record
from diffusion_accel.fixed_attention import (
    fixed_attention_projection_q10,
    fixed_attention_q12,
)
from diffusion_accel.fixed_mlp import _load_tensors


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"
MANIFEST = ROOT / "data/results/block-00-execution-image.json"


def _packed_hex(values: list[int], width: int) -> str:
    mask = (1 << width) - 1
    packed = sum((value & mask) << (index * width) for index, value in enumerate(values))
    return f"{packed:0{(len(values) * width + 3) // 4}x}"


def test_real_block_image_drives_projection_mac_and_matches_h0(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    image_path = PACKAGE / "block_00_execution.bin"
    if not MANIFEST.is_file() or not golden_path.is_file():
        pytest.skip("H0 hardware package is unavailable")
    goldens = _load_tensors(golden_path, ["folded.block_00.qkv"])
    weights = _load_tensors(
        weights_path,
        ["rotary.cos", "rotary.sin", "block_00.attention_out.weight"],
    )
    _, attention_q12, _ = fixed_attention_q12(
        goldens["folded.block_00.qkv"][0],
        weights["rotary.cos"].float(),
        weights["rotary.sin"].float(),
    )
    _, projection_q10, _ = fixed_attention_projection_q10(
        attention_q12, weights["block_00.attention_out.weight"]
    )
    canvas_hex = tmp_path / "projection_loader_compute_canvas.hex"
    canvas_hex.write_text(
        "\n".join(
            _packed_hex(
                [int(attention_q12[token, head * 64 + channel]) for channel in range(64)],
                18,
            )
            for head in range(12) for token in range(64)
        ) + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "projection_loader_compute_expected.hex"
    expected_hex.write_text(
        "\n".join(
            _packed_hex([int(projection_q10[token, lane]) for lane in range(6)], 24)
            for token in range(64)
        ) + "\n",
        encoding="utf-8",
    )
    beats: list[str] = []
    for input_tile in range(24):
        record = read_block_image_record(MANIFEST, "projection_weights", input_tile)
        beats.extend(
            f"{int.from_bytes(record[start:start + 64], 'little'):0128x}"
            for start in range(0, 192, 64)
        )
    beat_hex = tmp_path / "projection_loader_compute_beats.hex"
    beat_hex.write_text("\n".join(beats) + "\n", encoding="utf-8")
    metadata_offset = 3_686_400
    metadata_line = int.from_bytes(
        image_path.read_bytes()[metadata_offset:metadata_offset + 64], "little"
    )

    tb = tmp_path / "tb_mdlm_projection_loader_compute_h0.sv"
    tb.write_text(
        f"""`timescale 1ns/1ps
module tb_mdlm_projection_loader_compute_h0;
  localparam [63:0] BASE=64'ha0000000;
  reg clk=0,rst_n=0,command_valid=0,arready=0,rvalid=0,rlast=0;
  reg [511:0] rdata=0;reg projection_ready=1;
  wire command_ready,metadata_valid,metadata_ready,weight_valid,weight_ready;
  wire [6:0] metadata_tile,weight_tile;wire [143:0] multipliers;
  wire [4:0] weight_k;wire [1535:0] weight_data;
  wire [63:0] araddr,bytes_read,as,ds;wire [7:0] arlen;
  wire [2:0] arsize;wire [1:0] arburst;
  wire arvalid,rready,loader_busy,loader_done,loader_error;
  wire canvas_read_valid;wire [3:0] canvas_head,projection_group;
  wire [5:0] canvas_token;reg canvas_data_valid=0;reg [1151:0] canvas_data=0;
  wire projection_valid,compute_busy,compute_done;wire [6:0] projection_tile;
  wire [2:0] projection_channels;wire [575:0] projection_data;
  wire array_valid,array_clear,array_last;wire [7:0] array_tag;
  wire [2303:0] array_activations;wire [3455:0] array_weights;
  reg [511:0] beat_mem[0:71];reg [1151:0] canvas_mem[0:767];
  reg [143:0] expected_mem[0:63];
  integer cycle=0,source_beat=0,source_record=0,source_beats=0;
  integer groups=0,lane,token_lane,token;

  mdlm_projection_output_tile_loader loader(
    .clk(clk),.rst_n(rst_n),.block_base_address(BASE),
    .command_valid(command_valid),.command_ready(command_ready),
    .command_output_tile(0),.metadata_valid(metadata_valid),
    .metadata_ready(metadata_ready),.metadata_output_tile(metadata_tile),
    .metadata_multipliers_packed(multipliers),.weight_tile_valid(weight_valid),
    .weight_tile_ready(weight_ready),.weight_output_tile(weight_tile),
    .weight_input_tile(weight_k),.weight_int8_packed(weight_data),
    .m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),
    .m_axi_arburst(arburst),.m_axi_arvalid(arvalid),.m_axi_arready(arready),
    .m_axi_rdata(rdata),.m_axi_rresp(0),.m_axi_rlast(rlast),
    .m_axi_rvalid(rvalid),.m_axi_rready(rready),.busy(loader_busy),
    .done(loader_done),.protocol_error(loader_error),.bytes_read(bytes_read),
    .address_stall_cycles(as),.data_stall_cycles(ds));
  attention_projection_output_tile_scheduler compute(
    .clk(clk),.rst_n(rst_n),.start(metadata_valid),.start_ready(metadata_ready),
    .output_tile_in(metadata_tile),.multipliers_packed(multipliers),
    .weight_tile_valid(weight_valid),.weight_tile_ready(weight_ready),
    .weight_input_tile(weight_k),.weight_int8_packed(weight_data),
    .canvas_read_valid(canvas_read_valid),.canvas_read_head(canvas_head),
    .canvas_read_token(canvas_token),.canvas_read_data_valid(canvas_data_valid),
    .canvas_read_data_packed(canvas_data),.projection_tile_valid(projection_valid),
    .projection_tile_ready(projection_ready),.projection_group(projection_group),
    .projection_output_tile(projection_tile),
    .projection_valid_channels(projection_channels),
    .projection_q10_packed(projection_data),.array_request_valid(array_valid),
    .array_request_clear(array_clear),.array_request_last(array_last),
    .array_request_tag(array_tag),.array_request_activations(array_activations),
    .array_request_weights(array_weights),.array_response_valid(0),
    .array_response_tag(0),.array_response_accumulators(0),
    .busy(compute_busy),.done(compute_done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    cycle=cycle+1;arready<=arvalid && cycle[0];
    canvas_data_valid<=canvas_read_valid;
    if(canvas_read_valid) canvas_data<=canvas_mem[canvas_head*64+canvas_token];
    if(arvalid && arready) begin
      source_beat=0;source_beats=arlen+1;
      if(araddr==BASE+3686400) begin
        source_record=-1;rdata<=512'h{metadata_line:0128x};
      end else begin source_record=(araddr-BASE-3694592)/192;
        rdata<=beat_mem[source_record*3];end
      rlast<=source_beats==1;rvalid<=1;
    end else if(rvalid && rready) begin
      source_beat=source_beat+1;
      if(source_beat==source_beats) begin rvalid<=0;rlast<=0;end
      else begin rdata<=beat_mem[source_record*3+source_beat];
        rlast<=source_beat==source_beats-1;end
    end
    #1;
    if(projection_valid) begin
      if(projection_group!==groups || projection_tile!==0 || projection_channels!==6)
        $fatal(1,"projection tag mismatch");
      for(token_lane=0;token_lane<4;token_lane=token_lane+1)
        for(lane=0;lane<6;lane=lane+1) begin token=groups*4+token_lane;
          if($signed(projection_data[(token_lane*6+lane)*24 +: 24])!==
             $signed(expected_mem[token][lane*24 +: 24]))
            $fatal(1,"DDR-fed projection mismatch token=%0d lane=%0d",token,lane);
        end
      groups=groups+1;
    end
  end
  initial begin
    $readmemh("{beat_hex}",beat_mem);$readmemh("{canvas_hex}",canvas_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;command_valid=1;
    @(posedge clk);@(negedge clk);command_valid=0;wait(compute_done);
    repeat(2) @(posedge clk);#1;
    if(loader_error || loader_busy || compute_busy || groups!=16 || bytes_read!=4672)
      $fatal(1,"DDR-fed projection completion mismatch");
    $display("tb_mdlm_projection_loader_compute_h0: PASS groups=%0d bytes=%0d cycles=%0d",
      groups,bytes_read,cycle);$finish;
  end
  initial begin repeat(7000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_mdlm_projection_loader_compute_h0"
    sources = [
        "mdlm_block_parameter_address_generator.sv", "axi512_read_burst_master.sv",
        "mdlm_block_parameter_dma.sv", "fixed_aligned_record_adapter.sv",
        "fixed_weight_record_adapter.sv", "mdlm_projection_output_tile_loader.sv",
        "int8_mac_tile_pipelined.sv", "mixed_precision_mac_tile_pipelined.sv",
        "fixed_requantize.sv", "fixed_requantize_vector_serial.sv",
        "attention_projection_weight_tile_buffer.sv",
        "attention_projection_output_tile_scheduler.sv",
    ]
    compile_result = subprocess.run(
        ["iverilog", "-g2012", "-s", "tb_mdlm_projection_loader_compute_h0",
         "-o", str(build), *(str(RTL / source) for source in sources), str(tb)],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        ["vvp", str(build)], cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_mdlm_projection_loader_compute_h0: PASS" in run_result.stdout
