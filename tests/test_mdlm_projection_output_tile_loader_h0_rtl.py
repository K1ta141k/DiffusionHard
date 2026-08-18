from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.block_image import read_block_image_record
from diffusion_accel.fixed_attention import fixed_attention_projection_q10, fixed_attention_q12
from diffusion_accel.fixed_mlp import _load_tensors


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"
MANIFEST = ROOT / "data/results/block-00-execution-image.json"


def _pack(values: list[int], width: int) -> int:
    mask = (1 << width) - 1
    return sum((value & mask) << (index * width) for index, value in enumerate(values))


def test_one_shared_dma_loads_real_projection_metadata_and_weights(
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
    _, _, details = fixed_attention_projection_q10(
        attention_q12, weights["block_00.attention_out.weight"]
    )
    weight_q = details["tensors"]["weight_int8"]
    multipliers = details["tensors"]["requant_multiplier_q24"][:6]
    expected_metadata = _pack([int(value) for value in multipliers], 24)
    section_offset = 3_686_400
    metadata_line = int.from_bytes(
        image_path.read_bytes()[section_offset:section_offset + 64], "little"
    )
    beats: list[str] = []
    expected: list[str] = []
    for input_tile in range(24):
        record = read_block_image_record(MANIFEST, "projection_weights", input_tile)
        beats.extend(
            f"{int.from_bytes(record[start:start + 64], 'little'):0128x}"
            for start in range(0, 192, 64)
        )
        expected.append(
            f"{_pack([int(weight_q[row, input_tile * 32 + lane]) for row in range(6) for lane in range(32)], 8):0384x}"
        )
    beat_hex = tmp_path / "projection_loader_beats.hex"
    expected_hex = tmp_path / "projection_loader_expected.hex"
    beat_hex.write_text("\n".join(beats) + "\n", encoding="utf-8")
    expected_hex.write_text("\n".join(expected) + "\n", encoding="utf-8")
    tb = tmp_path / "tb_mdlm_projection_output_tile_loader_h0.sv"
    tb.write_text(
        f"""`timescale 1ns/1ps
module tb_mdlm_projection_output_tile_loader_h0;
  localparam [63:0] BASE=64'h80000000;
  reg clk=0,rst_n=0,command_valid=0,metadata_ready=0,weight_ready=0;
  reg arready=0,rvalid=0,rlast=0;reg [511:0] rdata=0;
  wire command_ready,metadata_valid,weight_valid;
  wire [6:0] metadata_tile,weight_tile;wire [4:0] weight_k;
  wire [143:0] metadata_data;wire [1535:0] weight_data;
  wire [63:0] araddr,bytes_read,address_stalls,data_stalls;wire [7:0] arlen;
  wire [2:0] arsize;wire [1:0] arburst;
  wire arvalid,rready,busy,done,protocol_error;
  reg [511:0] beat_mem[0:71];reg [1535:0] expected_mem[0:23];
  integer cycle=0,source_beat=0,source_record=0,source_beats=0;
  integer metadata_count=0,weight_count=0;
  mdlm_projection_output_tile_loader dut(
    .clk(clk),.rst_n(rst_n),.block_base_address(BASE),
    .command_valid(command_valid),.command_ready(command_ready),
    .command_output_tile(7'd0),.metadata_valid(metadata_valid),
    .metadata_ready(metadata_ready),.metadata_output_tile(metadata_tile),
    .metadata_multipliers_packed(metadata_data),.weight_tile_valid(weight_valid),
    .weight_tile_ready(weight_ready),.weight_output_tile(weight_tile),
    .weight_input_tile(weight_k),.weight_int8_packed(weight_data),
    .m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),
    .m_axi_arburst(arburst),.m_axi_arvalid(arvalid),.m_axi_arready(arready),
    .m_axi_rdata(rdata),.m_axi_rresp(2'b00),.m_axi_rlast(rlast),
    .m_axi_rvalid(rvalid),.m_axi_rready(rready),.busy(busy),.done(done),
    .protocol_error(protocol_error),.bytes_read(bytes_read),
    .address_stall_cycles(address_stalls),.data_stall_cycles(data_stalls));
  always #2 clk=~clk;
  always @(posedge clk) begin
    cycle=cycle+1;arready<=arvalid && cycle[0];
    metadata_ready<=cycle[1] || cycle[3];weight_ready<=cycle[0] || cycle[2];
    if(arvalid && arready) begin
      source_beat=0;source_beats=arlen+1;
      if(araddr==BASE+3686400) begin
        source_record=-1;rdata<=512'h{metadata_line:0128x};
        if(arlen!==0) $fatal(1,"projection metadata burst mismatch");
      end else begin
        source_record=(araddr-BASE-3694592)/192;
        if(source_record<0 || source_record>23 || arlen!==2)
          $fatal(1,"projection weight burst mismatch");
        rdata<=beat_mem[source_record*3];
      end
      rlast<=source_beats==1;rvalid<=1;
    end else if(rvalid && rready) begin
      source_beat=source_beat+1;
      if(source_beat==source_beats) begin rvalid<=0;rlast<=0;end
      else begin rdata<=beat_mem[source_record*3+source_beat];
        rlast<=source_beat==source_beats-1;end
    end
    #1;
    if(metadata_valid && metadata_ready) begin
      if(metadata_tile!==0 || metadata_data!==144'h{expected_metadata:036x})
        $fatal(1,"projection metadata mismatch");
      metadata_count=metadata_count+1;
    end
    if(weight_valid && weight_ready) begin
      if(weight_tile!==0 || weight_k!==weight_count[4:0] ||
         weight_data!==expected_mem[weight_count])
        $fatal(1,"projection weight mismatch %0d",weight_count);
      weight_count=weight_count+1;
    end
  end
  initial begin
    $readmemh("{beat_hex}",beat_mem);$readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;command_valid=1;
    @(posedge clk);@(negedge clk);command_valid=0;wait(done);
    repeat(2) @(posedge clk);#1;
    if(protocol_error || metadata_count!=1 || weight_count!=24 ||
       bytes_read!=4672 || busy) $fatal(1,"projection loader completion mismatch");
    $display("tb_mdlm_projection_output_tile_loader_h0: PASS metadata=%0d tiles=%0d bytes=%0d",
      metadata_count,weight_count,bytes_read);$finish;
  end
  initial begin repeat(1300) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_mdlm_projection_output_tile_loader_h0"
    sources = [
        "mdlm_block_parameter_address_generator.sv", "axi512_read_burst_master.sv",
        "mdlm_block_parameter_dma.sv", "fixed_aligned_record_adapter.sv",
        "fixed_weight_record_adapter.sv", "mdlm_projection_output_tile_loader.sv",
    ]
    compile_result = subprocess.run(
        ["iverilog", "-g2012", "-Wall", "-s",
         "tb_mdlm_projection_output_tile_loader_h0", "-o", str(build),
         *(str(RTL / source) for source in sources), str(tb)],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        ["vvp", str(build)], cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_mdlm_projection_output_tile_loader_h0: PASS" in run_result.stdout
