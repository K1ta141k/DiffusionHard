from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.block_image import read_block_image_record
from diffusion_accel.fixed_attention import fixed_qkv_projection_q12
from diffusion_accel.fixed_mlp import _load_tensors
from diffusion_accel.fixed_norm import fixed_layer_norm_q12


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"
MANIFEST = ROOT / "data/results/block-00-execution-image.json"


def _pack(values: list[int], width: int) -> int:
    mask = (1 << width) - 1
    return sum((value & mask) << (index * width) for index, value in enumerate(values))


def test_one_shared_dma_loads_real_qkv_metadata_and_weight_slice(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    image_path = PACKAGE / "block_00_execution.bin"
    if not MANIFEST.is_file() or not golden_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    goldens = _load_tensors(golden_path, ["folded.embedding"])
    weights = _load_tensors(
        weights_path, ["block_00.qkv.weight", "block_00.qkv.bias"]
    )
    _, normalized_q12, _ = fixed_layer_norm_q12(goldens["folded.embedding"][0])
    _, _, details = fixed_qkv_projection_q12(
        normalized_q12,
        weights["block_00.qkv.weight"],
        weights["block_00.qkv.bias"],
    )
    weight_q = details["tensors"]["weight_int16"]
    multipliers = [
        int(value) for value in details["tensors"]["requant_multiplier_q28"][:6]
    ]
    biases = [int(value) for value in details["tensors"]["bias_q12"][:6]]
    expected_metadata = _pack(multipliers, 24) | (_pack(biases, 18) << 144)
    metadata_line = int.from_bytes(image_path.read_bytes()[:64], "little")

    weight_beats: list[str] = []
    expected_tiles: list[str] = []
    for input_tile in range(24):
        record = read_block_image_record(MANIFEST, "qkv_weights", input_tile)
        weight_beats.extend(
            f"{int.from_bytes(record[start:start + 64], 'little'):0128x}"
            for start in range(0, 384, 64)
        )
        expected_tiles.append(
            f"{_pack([int(weight_q[row, input_tile * 32 + lane]) for row in range(6) for lane in range(32)], 16):0768x}"
        )
    beat_hex = tmp_path / "qkv_loader_weight_beats.hex"
    expected_hex = tmp_path / "qkv_loader_expected_tiles.hex"
    beat_hex.write_text("\n".join(weight_beats) + "\n", encoding="utf-8")
    expected_hex.write_text("\n".join(expected_tiles) + "\n", encoding="utf-8")

    testbench = tmp_path / "tb_mdlm_qkv_output_tile_loader_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_mdlm_qkv_output_tile_loader_h0;
  localparam [63:0] BASE=64'h70000000;
  reg clk=0,rst_n=0,command_valid=0;
  reg metadata_ready=0,weight_ready=0,arready=0,rvalid=0,rlast=0;
  reg [511:0] rdata=0;
  wire command_ready,metadata_valid,weight_valid;
  wire [3:0] metadata_head,metadata_channel,weight_head,weight_channel;
  wire [1:0] metadata_kind,weight_kind;
  wire [143:0] metadata_multipliers;
  wire [107:0] metadata_biases;
  wire [4:0] weight_k;
  wire [3071:0] weight_data;
  wire [63:0] araddr,bytes_read,address_stalls,data_stalls;
  wire [7:0] arlen;
  wire [2:0] arsize;
  wire [1:0] arburst;
  wire arvalid,rready,busy,done,protocol_error;
  reg [511:0] beat_mem[0:143];
  reg [3071:0] expected_mem[0:23];
  integer cycle=0,source_beat=0,source_record=0,source_beats=0;
  integer metadata_count=0,weight_count=0;

  mdlm_qkv_output_tile_loader dut(
    .clk(clk),.rst_n(rst_n),.block_base_address(BASE),
    .command_valid(command_valid),.command_ready(command_ready),
    .command_head(4'd0),.command_kind(2'd0),.command_channel_tile(4'd0),
    .metadata_valid(metadata_valid),.metadata_ready(metadata_ready),
    .metadata_head(metadata_head),.metadata_kind(metadata_kind),
    .metadata_channel_tile(metadata_channel),
    .metadata_multipliers_packed(metadata_multipliers),
    .metadata_biases_q12_packed(metadata_biases),
    .weight_tile_valid(weight_valid),.weight_tile_ready(weight_ready),
    .weight_head(weight_head),.weight_kind(weight_kind),
    .weight_channel_tile(weight_channel),.weight_input_tile(weight_k),
    .weight_int16_packed(weight_data),.m_axi_araddr(araddr),
    .m_axi_arlen(arlen),.m_axi_arsize(arsize),.m_axi_arburst(arburst),
    .m_axi_arvalid(arvalid),.m_axi_arready(arready),.m_axi_rdata(rdata),
    .m_axi_rresp(2'b00),.m_axi_rlast(rlast),.m_axi_rvalid(rvalid),
    .m_axi_rready(rready),.busy(busy),.done(done),
    .protocol_error(protocol_error),.bytes_read(bytes_read),
    .address_stall_cycles(address_stalls),.data_stall_cycles(data_stalls));

  always #2 clk=~clk;
  always @(posedge clk) begin
    cycle=cycle+1;arready<=arvalid && cycle[0];
    metadata_ready<=cycle[1] || cycle[3];weight_ready<=cycle[0] || cycle[2];
    if(arvalid && arready) begin
      source_beat=0;source_beats=arlen+1;
      if(araddr==BASE) begin
        if(arlen!==0) $fatal(1,"metadata burst mismatch");
        source_record=-1;rdata<=512'h{metadata_line:0128x};
      end else begin
        source_record=(araddr-BASE-28672)/384;
        if(source_record<0 || source_record>23 || arlen!==5)
          $fatal(1,"weight burst mismatch");
        rdata<=beat_mem[source_record*6];
      end
      rlast<=source_beats==1;rvalid<=1;
    end else if(rvalid && rready) begin
      source_beat=source_beat+1;
      if(source_beat==source_beats) begin rvalid<=0;rlast<=0;end
      else begin
        rdata<=beat_mem[source_record*6+source_beat];
        rlast<=source_beat==source_beats-1;
      end
    end
    #1;
    if(metadata_valid && metadata_ready) begin
      if(metadata_head!==0 || metadata_kind!==0 || metadata_channel!==0 ||
         {{metadata_biases,metadata_multipliers}}!==252'h{expected_metadata:063x})
        $fatal(1,"shared-DMA metadata mismatch");
      metadata_count=metadata_count+1;
    end
    if(weight_valid && weight_ready) begin
      if(weight_head!==0 || weight_kind!==0 || weight_channel!==0 ||
         weight_k!==weight_count[4:0] || weight_data!==expected_mem[weight_count])
        $fatal(1,"shared-DMA weight mismatch %0d",weight_count);
      weight_count=weight_count+1;
    end
  end

  initial begin
    $readmemh("{beat_hex}",beat_mem);$readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;command_valid=1;
    @(posedge clk);@(negedge clk);command_valid=0;wait(done);
    repeat(2) @(posedge clk);#1;
    if(protocol_error || metadata_count!=1 || weight_count!=24 ||
       bytes_read!=9280 || busy)
      $fatal(1,"shared-DMA QKV loader completion mismatch");
    $display("tb_mdlm_qkv_output_tile_loader_h0: PASS metadata=%0d tiles=%0d bytes=%0d",
      metadata_count,weight_count,bytes_read);
    $finish;
  end
  initial begin repeat(2200) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_mdlm_qkv_output_tile_loader_h0"
    sources = [
        "mdlm_block_parameter_address_generator.sv",
        "axi512_read_burst_master.sv",
        "mdlm_block_parameter_dma.sv",
        "fixed_aligned_record_adapter.sv",
        "fixed_weight_record_adapter.sv",
        "mdlm_qkv_output_tile_loader.sv",
    ]
    compile_result = subprocess.run(
        ["iverilog", "-g2012", "-Wall", "-s", "tb_mdlm_qkv_output_tile_loader_h0",
         "-o", str(build), *(str(RTL / source) for source in sources),
         str(testbench)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        ["vvp", str(build)], cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_mdlm_qkv_output_tile_loader_h0: PASS" in run_result.stdout
