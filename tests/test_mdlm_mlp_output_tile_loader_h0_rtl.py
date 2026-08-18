from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.block_image import (
    SMOOTHQUANT_ALPHAS,
    read_block_image_record,
)
from diffusion_accel.fixed_mlp import _load_tensors, _quantize_weight_per_output


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"
MANIFEST = ROOT / "data/results/block-00-execution-image.json"


def _pack_fields(fields: list[tuple[list[int], int]]) -> int:
    packed = 0
    offset = 0
    for values, width in fields:
        mask = (1 << width) - 1
        for value in values:
            packed |= (value & mask) << offset
            offset += width
    return packed


@pytest.mark.parametrize("phase", ["up", "down"])
def test_shared_dma_loads_real_mlp_metadata_and_weight_slice(
    tmp_path: Path,
    phase: str,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    artifact_path = PACKAGE / "mlp_interstage_int8.safetensors"
    image_path = PACKAGE / "block_00_execution.bin"
    if not MANIFEST.is_file() or not artifact_path.is_file():
        pytest.skip("H0 hardware package is unavailable")
    artifacts = _load_tensors(
        artifact_path,
        [
            "block_00.up_output_factor",
            "block_00.up_bias_q10",
            "block_00.interstage_multiplier",
            "block_00.down_weight",
            "block_00.down_output_multiplier",
            "block_00.down_bias_q10",
        ],
    )
    if phase == "up":
        norm2 = _load_tensors(
            golden_path, ["folded.block_00.norm2_unaffine"]
        )["folded.block_00.norm2_unaffine"][0]
        up_weight = _load_tensors(
            weights_path, ["block_00.mlp_up.weight"]
        )["block_00.mlp_up.weight"]
        alpha = SMOOTHQUANT_ALPHAS[0]
        smoothing = (
            norm2.abs().amax(dim=0).clamp_min(1e-8).pow(alpha)
            / up_weight.abs().amax(dim=0).clamp_min(1e-8).pow(1.0 - alpha)
        ).clamp_min(1e-8)
        weight_q, _ = _quantize_weight_per_output(
            up_weight * smoothing[None, :], 8
        )
        metadata = _pack_fields(
            [
                ([int(v) for v in artifacts["block_00.up_output_factor"][:6]], 18),
                ([int(v) for v in artifacts["block_00.up_bias_q10"][:6]], 32),
                ([int(v) for v in artifacts["block_00.interstage_multiplier"][:6]], 24),
            ]
        )
        config = dict(
            meta_section=6, weight_section=7, input_size=768,
            meta_width=444, meta_payload=56, k_tiles=24, k_width=5,
            meta_offset=4_288_512, weight_offset=4_321_280,
            meta_beats=1, section="mlp_up_weights", expected_bytes=4_672,
        )
    else:
        weight_q = artifacts["block_00.down_weight"]
        multipliers = [
            int(artifacts["block_00.down_output_multiplier"][lane])
            for _token in range(4) for lane in range(6)
        ]
        biases = [
            int(artifacts["block_00.down_bias_q10"][lane])
            for _token in range(4) for lane in range(6)
        ]
        metadata = _pack_fields([(multipliers, 24), (biases, 32)])
        config = dict(
            meta_section=8, weight_section=9, input_size=3072,
            meta_width=1344, meta_payload=168, k_tiles=96, k_width=7,
            meta_offset=6_680_576, weight_offset=6_705_152,
            meta_beats=3, section="mlp_down_weights", expected_bytes=18_624,
        )

    image = image_path.read_bytes()
    meta_raw = image[
        config["meta_offset"]:config["meta_offset"] + config["meta_beats"] * 64
    ]
    meta_hex = tmp_path / f"{phase}_loader_metadata_beats.hex"
    meta_hex.write_text(
        "\n".join(
            f"{int.from_bytes(meta_raw[start:start + 64], 'little'):0128x}"
            for start in range(0, len(meta_raw), 64)
        ) + "\n",
        encoding="utf-8",
    )
    weight_beats: list[str] = []
    expected_tiles: list[str] = []
    for input_tile in range(config["k_tiles"]):
        record = read_block_image_record(MANIFEST, config["section"], input_tile)
        weight_beats.extend(
            f"{int.from_bytes(record[start:start + 64], 'little'):0128x}"
            for start in range(0, 192, 64)
        )
        expected_tiles.append(
            f"{_pack_fields([([int(weight_q[row, input_tile * 32 + lane]) for row in range(6) for lane in range(32)], 8)]):0384x}"
        )
    weight_hex = tmp_path / f"{phase}_loader_weight_beats.hex"
    expected_hex = tmp_path / f"{phase}_loader_expected.hex"
    weight_hex.write_text("\n".join(weight_beats) + "\n", encoding="utf-8")
    expected_hex.write_text("\n".join(expected_tiles) + "\n", encoding="utf-8")

    module = f"tb_mdlm_mlp_{phase}_output_tile_loader_h0"
    tb = tmp_path / f"{module}.sv"
    tb.write_text(
        f"""`timescale 1ns/1ps
module {module};
  localparam [63:0] BASE=64'hb0000000;
  reg clk=0,rst_n=0,command_valid=0,meta_ready=0,weight_ready=0;
  reg arready=0,rvalid=0,rlast=0;reg [511:0] rdata=0;
  wire command_ready,meta_valid,weight_valid;
  wire [{config['meta_width']-1}:0] meta_data;
  wire [1535:0] weight_data;wire [{config['k_width']-1}:0] weight_k;
  wire [63:0] araddr,bytes_read,as,ds;wire [7:0] arlen;
  wire [2:0] arsize;wire [1:0] arburst;
  wire arvalid,rready,busy,done,protocol_error;
  reg [511:0] meta_mem[0:{config['meta_beats']-1}];
  reg [511:0] weight_mem[0:{config['k_tiles']*3-1}];
  reg [1535:0] expected_mem[0:{config['k_tiles']-1}];
  integer cycle=0,source_beat=0,source_record=0,source_beats=0,is_meta=0;
  integer metadata_count=0,weight_count=0;
  mdlm_mlp_output_tile_loader #(.METADATA_SECTION_ID({config['meta_section']}),
    .WEIGHT_SECTION_ID({config['weight_section']}),.INPUT_SIZE({config['input_size']}),
    .METADATA_WIDTH({config['meta_width']}),
    .METADATA_PAYLOAD_BYTES({config['meta_payload']})) dut(
    .clk(clk),.rst_n(rst_n),.block_base_address(BASE),
    .command_valid(command_valid),.command_ready(command_ready),
    .command_output_tile(0),.metadata_stream_valid(meta_valid),
    .metadata_stream_ready(meta_ready),.metadata_stream_data(meta_data),
    .weight_stream_valid(weight_valid),.weight_stream_ready(weight_ready),
    .weight_stream_data(weight_data),.weight_stream_k_tile(weight_k),
    .m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),
    .m_axi_arburst(arburst),.m_axi_arvalid(arvalid),.m_axi_arready(arready),
    .m_axi_rdata(rdata),.m_axi_rresp(0),.m_axi_rlast(rlast),
    .m_axi_rvalid(rvalid),.m_axi_rready(rready),.busy(busy),.done(done),
    .protocol_error(protocol_error),.bytes_read(bytes_read),
    .address_stall_cycles(as),.data_stall_cycles(ds));
  always #2 clk=~clk;
  always @(posedge clk) begin
    cycle=cycle+1;arready<=arvalid && cycle[0];
    meta_ready<=cycle[1] || cycle[3];weight_ready<=cycle[0] || cycle[2];
    if(arvalid && arready) begin
      source_beat=0;source_beats=arlen+1;
      if(araddr==BASE+{config['meta_offset']}) begin
        is_meta=1;source_record=0;rdata<=meta_mem[0];
        if(arlen!=={config['meta_beats']-1}) $fatal(1,"MLP metadata burst mismatch");
      end else begin
        is_meta=0;source_record=(araddr-BASE-{config['weight_offset']})/192;
        if(source_record<0 || source_record>={config['k_tiles']} || arlen!==2)
          $fatal(1,"MLP weight burst mismatch");
        rdata<=weight_mem[source_record*3];
      end
      rlast<=source_beats==1;rvalid<=1;
    end else if(rvalid && rready) begin
      source_beat=source_beat+1;
      if(source_beat==source_beats) begin rvalid<=0;rlast<=0;end
      else begin
        rdata<=is_meta ? meta_mem[source_beat]
          : weight_mem[source_record*3+source_beat];
        rlast<=source_beat==source_beats-1;
      end
    end
    #1;
    if(meta_valid && meta_ready) begin
      if(meta_data!=={config['meta_width']}'h{metadata:0{(config['meta_width']+3)//4}x})
        $fatal(1,"MLP metadata mismatch");
      metadata_count=metadata_count+1;
    end
    if(weight_valid && weight_ready) begin
      if(weight_k!==weight_count[{config['k_width']-1}:0] ||
         weight_data!==expected_mem[weight_count])
        $fatal(1,"MLP weight mismatch %0d",weight_count);
      weight_count=weight_count+1;
    end
  end
  initial begin
    $readmemh("{meta_hex}",meta_mem);$readmemh("{weight_hex}",weight_mem);
    $readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;command_valid=1;
    @(posedge clk);@(negedge clk);command_valid=0;wait(done);
    repeat(2) @(posedge clk);#1;
    if(protocol_error || metadata_count!=1 || weight_count!={config['k_tiles']} ||
       bytes_read!={config['expected_bytes']} || busy)
      $fatal(1,"MLP loader completion mismatch");
    $display("{module}: PASS metadata=%0d tiles=%0d bytes=%0d",
      metadata_count,weight_count,bytes_read);$finish;
  end
  initial begin repeat({config['k_tiles']*20+200}) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / module
    sources = [
        "mdlm_block_parameter_address_generator.sv", "axi512_read_burst_master.sv",
        "mdlm_block_parameter_dma.sv", "fixed_aligned_record_adapter.sv",
        "fixed_weight_record_adapter.sv", "mdlm_mlp_output_tile_loader.sv",
    ]
    compile_result = subprocess.run(
        ["iverilog", "-g2012", "-s", module, "-o", str(build),
         *(str(RTL / source) for source in sources), str(tb)],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        ["vvp", str(build)], cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert f"{module}: PASS" in run_result.stdout
