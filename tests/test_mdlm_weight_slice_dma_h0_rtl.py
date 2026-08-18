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


def _packed_hex(values: list[int], width: int) -> str:
    mask = (1 << width) - 1
    packed = sum(
        (value & mask) << (index * width)
        for index, value in enumerate(values)
    )
    return f"{packed:0{(len(values) * width + 3) // 4}x}"


def test_block_image_qkv_slice_reaches_rtl_loader_bit_exactly(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
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

    beats: list[str] = []
    expected_tiles: list[str] = []
    for input_tile in range(24):
        record = read_block_image_record(MANIFEST, "qkv_weights", input_tile)
        assert len(record) == 384
        beats.extend(
            f"{int.from_bytes(record[start:start + 64], 'little'):0128x}"
            for start in range(0, 384, 64)
        )
        expected_tiles.append(
            _packed_hex(
                [
                    int(weight_q[row, input_tile * 32 + lane])
                    for row in range(6)
                    for lane in range(32)
                ],
                16,
            )
        )

    beat_hex = tmp_path / "qkv_block_image_beats.hex"
    expected_hex = tmp_path / "qkv_expected_tiles.hex"
    beat_hex.write_text("\n".join(beats) + "\n", encoding="utf-8")
    expected_hex.write_text("\n".join(expected_tiles) + "\n", encoding="utf-8")
    testbench = tmp_path / "tb_mdlm_weight_slice_dma_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_mdlm_weight_slice_dma_h0;
  localparam [63:0] BASE=64'h40000000;
  reg clk=0,rst_n=0,command_valid=0;
  reg arready=0,rvalid=0,rlast=0,weight_ready=0;
  reg [511:0] rdata=0;
  wire command_ready,weight_valid,weight_bank;
  wire [4:0] weight_k;
  wire [3071:0] weight_data;
  wire [63:0] araddr,bytes_read,address_stalls,data_stalls;
  wire [7:0] arlen;
  wire [2:0] arsize;
  wire [1:0] arburst;
  wire arvalid,rready,busy,done,protocol_error;
  reg [511:0] beat_mem[0:143];
  reg [3071:0] expected_mem[0:23];
  integer cycle=0,record=0,source_beat=0,accepted=0;

  mdlm_weight_slice_dma #(.SECTION_ID(1),.DATA_WIDTH(16)) dut(
    .clk(clk),.rst_n(rst_n),.block_base_address(BASE),
    .command_valid(command_valid),.command_ready(command_ready),
    .command_bank(1'b0),.command_output_tile(9'd0),
    .weight_load_valid(weight_valid),.weight_load_ready(weight_ready),
    .weight_load_bank(weight_bank),.weight_load_k_tile(weight_k),
    .weight_load_data(weight_data),.m_axi_araddr(araddr),.m_axi_arlen(arlen),
    .m_axi_arsize(arsize),.m_axi_arburst(arburst),.m_axi_arvalid(arvalid),
    .m_axi_arready(arready),.m_axi_rdata(rdata),.m_axi_rresp(2'b00),
    .m_axi_rlast(rlast),.m_axi_rvalid(rvalid),.m_axi_rready(rready),
    .busy(busy),.done(done),.protocol_error(protocol_error),
    .bytes_read(bytes_read),.address_stall_cycles(address_stalls),
    .data_stall_cycles(data_stalls));

  always #2 clk=~clk;
  always @(posedge clk) begin
    cycle=cycle+1;arready<=arvalid && cycle[0];
    weight_ready<=cycle[0] || cycle[2];
    if(arvalid && arready) begin
      record=(araddr-BASE-28672)/384;
      if(record<0 || record>23 || arlen!==5) $fatal(1,"bad image address");
      source_beat=0;rdata<=beat_mem[record*6];rlast<=0;rvalid<=1;
    end else if(rvalid && rready) begin
      source_beat=source_beat+1;
      if(source_beat==6) begin rvalid<=0;rlast<=0;end
      else begin
        rdata<=beat_mem[record*6+source_beat];rlast<=source_beat==5;
      end
    end
    #1;
    if(weight_valid && weight_ready) begin
      if(weight_bank || weight_k!==accepted[4:0] ||
         weight_data!==expected_mem[accepted])
        $fatal(1,"real QKV tile mismatch %0d",accepted);
      accepted=accepted+1;
    end
  end

  initial begin
    $readmemh("{beat_hex}",beat_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;command_valid=1;
    @(posedge clk);@(negedge clk);command_valid=0;wait(done);
    repeat(2) @(posedge clk);#1;
    if(protocol_error || accepted!=24 || bytes_read!=9216 || busy)
      $fatal(1,"real QKV slice completion mismatch");
    $display("tb_mdlm_weight_slice_dma_h0: PASS tiles=%0d bytes=%0d",accepted,bytes_read);
    $finish;
  end
  initial begin repeat(2000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )

    build = tmp_path / "tb_mdlm_weight_slice_dma_h0"
    sources = [
        "mdlm_block_parameter_address_generator.sv",
        "axi512_read_burst_master.sv",
        "mdlm_block_parameter_dma.sv",
        "fixed_weight_record_adapter.sv",
        "mdlm_weight_slice_dma.sv",
    ]
    compile_result = subprocess.run(
        ["iverilog", "-g2012", "-Wall", "-s", "tb_mdlm_weight_slice_dma_h0",
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
    assert "tb_mdlm_weight_slice_dma_h0: PASS" in run_result.stdout
