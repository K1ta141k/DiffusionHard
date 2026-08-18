from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_block_parameter_load_fabric_elaborates_one_physical_axi_port(
    tmp_path: Path,
) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    tb = tmp_path / "tb_mdlm_block_parameter_load_fabric.sv"
    tb.write_text(
        """`timescale 1ns/1ps
module tb_mdlm_block_parameter_load_fabric;
reg clk=0,rst_n=0;
wire [3:0] busy;wire error;wire [63:0] addr,transactions;
wire [7:0] len;wire [2:0] size;wire [1:0] burst;wire valid,rready;
mdlm_block_parameter_load_fabric dut(
 .clk(clk),.rst_n(rst_n),.block_base_address(0),
 .qkv_command_valid(0),.qkv_command_head(0),.qkv_command_kind(0),
 .qkv_command_channel_tile(0),.qkv_metadata_ready(0),.qkv_weight_ready(0),
 .projection_command_valid(0),.projection_command_output_tile(0),
 .projection_metadata_ready(0),.projection_weight_ready(0),
 .up_command_valid(0),.up_command_output_tile(0),.up_metadata_ready(0),
 .up_weight_ready(0),.down_command_valid(0),.down_command_output_tile(0),
 .down_metadata_ready(0),.down_weight_ready(0),.m_axi_araddr(addr),
 .m_axi_arlen(len),.m_axi_arsize(size),.m_axi_arburst(burst),
 .m_axi_arvalid(valid),.m_axi_arready(0),.m_axi_rdata(0),.m_axi_rresp(0),
 .m_axi_rlast(0),.m_axi_rvalid(0),.m_axi_rready(rready),
 .client_busy(busy),.protocol_error(error),.read_transactions(transactions));
always #2 clk=~clk;
initial begin
 repeat(3) @(posedge clk);rst_n=1;repeat(3) @(posedge clk);#1;
 if(busy||error||transactions) $fatal(1,"bad fabric reset");
 $display("tb_mdlm_block_parameter_load_fabric: PASS");$finish;
end
endmodule
""",
        encoding="utf-8",
    )
    sources = [
        "mdlm_block_parameter_address_generator.sv",
        "axi512_read_burst_master.sv",
        "mdlm_block_parameter_dma.sv",
        "fixed_aligned_record_adapter.sv",
        "fixed_weight_record_adapter.sv",
        "mdlm_qkv_output_tile_loader.sv",
        "mdlm_projection_output_tile_loader.sv",
        "mdlm_mlp_output_tile_loader.sv",
        "axi512_read_arbiter_4.sv",
        "mdlm_block_parameter_load_fabric.sv",
    ]
    build = tmp_path / "tb_mdlm_block_parameter_load_fabric"
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s",
            "tb_mdlm_block_parameter_load_fabric", "-o", str(build),
            *(str(RTL / source) for source in sources), str(tb),
        ],
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
    assert "tb_mdlm_block_parameter_load_fabric: PASS" in run_result.stdout
