from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.fixed_attention import fixed_qkv_projection_q12
from diffusion_accel.fixed_mlp import _load_tensors
from diffusion_accel.fixed_norm import fixed_layer_norm_q12


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"


def _pack(values: list[int], width: int) -> int:
    mask = (1 << width) - 1
    return sum((value & mask) << (index * width) for index, value in enumerate(values))


def test_block_image_qkv_metadata_reaches_rtl_loader_bit_exactly(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    image_path = PACKAGE / "block_00_execution.bin"
    if not image_path.is_file() or not golden_path.is_file():
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
    multipliers = [
        int(value) for value in details["tensors"]["requant_multiplier_q28"][:6]
    ]
    biases = [int(value) for value in details["tensors"]["bias_q12"][:6]]
    expected = _pack(multipliers, 24) | (_pack(biases, 18) << (6 * 24))
    line = int.from_bytes(image_path.read_bytes()[:64], "little")

    testbench = tmp_path / "tb_mdlm_parameter_record_dma_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_mdlm_parameter_record_dma_h0;
  localparam [63:0] BASE=64'h50000000;
  reg clk=0,rst_n=0,request_valid=0,record_ready=0;
  reg arready=0,rvalid=0;
  wire request_ready,record_valid;
  wire [251:0] record_data;
  wire [15:0] record_tag;
  wire [63:0] araddr,bytes_read,address_stalls,data_stalls;
  wire [7:0] arlen;
  wire [2:0] arsize;
  wire [1:0] arburst;
  wire arvalid,rready,busy,done,invalid_request,protocol_error;
  integer accepted=0;

  mdlm_parameter_record_dma #(.SECTION_ID(0),.RECORD_WIDTH(252),
    .EXPECTED_PAYLOAD_BYTES(32)) dut(
    .clk(clk),.rst_n(rst_n),.block_base_address(BASE),
    .request_valid(request_valid),.request_ready(request_ready),
    .request_record_index(14'd0),.request_tag(16'hbeef),
    .record_valid(record_valid),.record_ready(record_ready),
    .record_data(record_data),.record_tag(record_tag),.m_axi_araddr(araddr),
    .m_axi_arlen(arlen),.m_axi_arsize(arsize),.m_axi_arburst(arburst),
    .m_axi_arvalid(arvalid),.m_axi_arready(arready),
    .m_axi_rdata(512'h{line:0128x}),.m_axi_rresp(2'b00),.m_axi_rlast(1'b1),
    .m_axi_rvalid(rvalid),.m_axi_rready(rready),.busy(busy),.done(done),
    .invalid_request(invalid_request),.protocol_error(protocol_error),
    .bytes_read(bytes_read),.address_stall_cycles(address_stalls),
    .data_stall_cycles(data_stalls));

  always #2 clk=~clk;
  always @(posedge clk) begin
    arready<=arvalid;
    if(arvalid && arready) begin
      if(araddr!==BASE || arlen!==0 || arsize!==6 || arburst!==1)
        $fatal(1,"QKV metadata address mismatch");
      rvalid<=1;
    end else if(rvalid && rready) rvalid<=0;
    #1;
    if(record_valid) begin
      if(record_tag!==16'hbeef || record_data!==252'h{expected:063x})
        $fatal(1,"QKV metadata mismatch");
      accepted=accepted+1;record_ready<=1;
    end else record_ready<=0;
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;request_valid=1;
    @(posedge clk);@(negedge clk);request_valid=0;wait(done);
    repeat(2) @(posedge clk);#1;
    if(protocol_error || invalid_request || accepted!=1 || bytes_read!=64 || busy)
      $fatal(1,"QKV metadata completion mismatch");
    $display("tb_mdlm_parameter_record_dma_h0: PASS bytes=%0d",bytes_read);
    $finish;
  end
  initial begin repeat(100) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_mdlm_parameter_record_dma_h0"
    sources = [
        "mdlm_block_parameter_address_generator.sv",
        "axi512_read_burst_master.sv",
        "mdlm_block_parameter_dma.sv",
        "fixed_aligned_record_adapter.sv",
        "compact_table_record_adapter.sv",
        "mdlm_parameter_record_dma.sv",
    ]
    compile_result = subprocess.run(
        ["iverilog", "-g2012", "-Wall", "-s", "tb_mdlm_parameter_record_dma_h0",
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
    assert "tb_mdlm_parameter_record_dma_h0: PASS" in run_result.stdout


def test_block_image_compact_reciprocal_reaches_rtl_bit_exactly(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    image_path = PACKAGE / "block_00_execution.bin"
    artifact_path = PACKAGE / "mlp_interstage_int8.safetensors"
    if not image_path.is_file() or not artifact_path.is_file():
        pytest.skip("H0 hardware package is unavailable")
    record_index = 17
    reciprocal = int(
        _load_tensors(
            artifact_path, ["block_00.up_smoothing_reciprocal_q15"]
        )["block_00.up_smoothing_reciprocal_q15"][record_index]
    )
    line_offset = 4_284_416 + (record_index // 16) * 64
    line = int.from_bytes(
        image_path.read_bytes()[line_offset:line_offset + 64], "little"
    )
    testbench = tmp_path / "tb_mdlm_compact_record_dma_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_mdlm_compact_record_dma_h0;
  localparam [63:0] BASE=64'h60000000;
  reg clk=0,rst_n=0,request_valid=0,record_ready=0,arready=0,rvalid=0;
  wire request_ready,record_valid;
  wire [17:0] record_data;
  wire [15:0] record_tag;
  wire [63:0] araddr,bytes_read,address_stalls,data_stalls;
  wire [7:0] arlen;
  wire [2:0] arsize;
  wire [1:0] arburst;
  wire arvalid,rready,busy,done,invalid_request,protocol_error;
  integer accepted=0;

  mdlm_parameter_record_dma #(.SECTION_ID(5),.RECORD_WIDTH(18),
    .EXPECTED_PAYLOAD_BYTES(3),.COMPACT_RECORD(1)) dut(
    .clk(clk),.rst_n(rst_n),.block_base_address(BASE),
    .request_valid(request_valid),.request_ready(request_ready),
    .request_record_index(14'd{record_index}),.request_tag(16'h1234),
    .record_valid(record_valid),.record_ready(record_ready),
    .record_data(record_data),.record_tag(record_tag),.m_axi_araddr(araddr),
    .m_axi_arlen(arlen),.m_axi_arsize(arsize),.m_axi_arburst(arburst),
    .m_axi_arvalid(arvalid),.m_axi_arready(arready),
    .m_axi_rdata(512'h{line:0128x}),.m_axi_rresp(2'b00),.m_axi_rlast(1'b1),
    .m_axi_rvalid(rvalid),.m_axi_rready(rready),.busy(busy),.done(done),
    .invalid_request(invalid_request),.protocol_error(protocol_error),
    .bytes_read(bytes_read),.address_stall_cycles(address_stalls),
    .data_stall_cycles(data_stalls));

  always #2 clk=~clk;
  always @(posedge clk) begin
    arready<=arvalid;
    if(arvalid && arready) begin
      if(araddr!==BASE+4284416+64 || arlen!==0)
        $fatal(1,"compact reciprocal address mismatch");
      rvalid<=1;
    end else if(rvalid && rready) rvalid<=0;
    #1;
    if(record_valid) begin
      if(record_tag!==16'h1234 || record_data!==18'h{reciprocal & 0x3FFFF:05x})
        $fatal(1,"compact reciprocal mismatch");
      accepted=accepted+1;record_ready<=1;
    end else record_ready<=0;
  end

  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;request_valid=1;
    @(posedge clk);@(negedge clk);request_valid=0;wait(done);
    repeat(2) @(posedge clk);#1;
    if(protocol_error || invalid_request || accepted!=1 || bytes_read!=64 || busy)
      $fatal(1,"compact reciprocal completion mismatch");
    $display("tb_mdlm_compact_record_dma_h0: PASS value=%0d",record_data);
    $finish;
  end
  initial begin repeat(100) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_mdlm_compact_record_dma_h0"
    sources = [
        "mdlm_block_parameter_address_generator.sv",
        "axi512_read_burst_master.sv",
        "mdlm_block_parameter_dma.sv",
        "fixed_aligned_record_adapter.sv",
        "compact_table_record_adapter.sv",
        "mdlm_parameter_record_dma.sv",
    ]
    compile_result = subprocess.run(
        ["iverilog", "-g2012", "-Wall", "-s", "tb_mdlm_compact_record_dma_h0",
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
    assert "tb_mdlm_compact_record_dma_h0: PASS" in run_result.stdout
