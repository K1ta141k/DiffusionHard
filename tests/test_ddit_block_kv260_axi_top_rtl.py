from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]
RTL=ROOT/"rtl/tensor_engine"


def test_kv260_axi_control_and_memory_top_elaborates(tmp_path:Path)->None:
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog is required")
    build=tmp_path/"ddit_block_kv260_axi_top"
    result=subprocess.run(
        ["iverilog","-g2012","-Wall","-s","ddit_block_kv260_axi_top",
         "-o",str(build),*(str(path) for path in sorted(RTL.glob("*.sv")))],
        cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr


def test_kv260_axi_lite_register_contract(tmp_path:Path)->None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    tb=tmp_path/"tb_ddit_block_kv260_axi_top.sv"
    tb.write_text("""`timescale 1ns/1ps
module tb_ddit_block_kv260_axi_top;reg clk=0,rst_n=0;
reg [7:0] awaddr=0;reg awvalid=0;wire awready;reg [31:0] wdata=0;
reg [3:0] wstrb=0;reg wvalid=0;wire wready;wire [1:0] bresp;
wire bvalid;reg bready=0;reg [7:0] araddr=0;reg arvalid=0;wire arready;
wire [31:0] rdata;wire [1:0] rresp;wire rvalid;reg rready=0;
ddit_block_kv260_axi_top dut(.ap_clk(clk),.ap_rst_n(rst_n),
.s_axi_control_awaddr(awaddr),.s_axi_control_awvalid(awvalid),
.s_axi_control_awready(awready),.s_axi_control_wdata(wdata),
.s_axi_control_wstrb(wstrb),.s_axi_control_wvalid(wvalid),
.s_axi_control_wready(wready),.s_axi_control_bresp(bresp),
.s_axi_control_bvalid(bvalid),.s_axi_control_bready(bready),
.s_axi_control_araddr(araddr),.s_axi_control_arvalid(arvalid),
.s_axi_control_arready(arready),.s_axi_control_rdata(rdata),
.s_axi_control_rresp(rresp),.s_axi_control_rvalid(rvalid),
.s_axi_control_rready(rready),.m_axi_gmem_arready(0),.m_axi_gmem_rdata(0),
.m_axi_gmem_rresp(0),.m_axi_gmem_rlast(0),.m_axi_gmem_rvalid(0),
.m_axi_gmem_awready(0),.m_axi_gmem_wready(0),.m_axi_gmem_bresp(0),
.m_axi_gmem_bvalid(0));
always #2 clk=~clk;
task write_reg;input [7:0] address;input [31:0] value;begin
@(negedge clk);awaddr=address;awvalid=1;wait(awready);@(posedge clk);
@(negedge clk);awvalid=0;repeat(2)@(posedge clk);@(negedge clk);
wdata=value;wstrb=4'hf;wvalid=1;wait(wready);@(posedge clk);
@(negedge clk);wvalid=0;bready=1;wait(bvalid);#1;
if(bresp!==0)$fatal(1,"AXI-Lite write response mismatch");@(posedge clk);
@(negedge clk);bready=0;end endtask
task read_reg;input [7:0] address;input [31:0] expected;begin
@(negedge clk);araddr=address;arvalid=1;rready=1;wait(arready);@(posedge clk);#1;
if(!rvalid||rresp!==0||rdata!==expected)
$fatal(1,"AXI-Lite read mismatch address=%h data=%h expected=%h",address,rdata,expected);
@(negedge clk);arvalid=0;rready=0;end endtask
initial begin repeat(3)@(posedge clk);@(negedge clk);rst_n=1;
write_reg(8'h10,32'h89abcdef);write_reg(8'h14,32'h01234567);
write_reg(8'h18,32'h76543210);write_reg(8'h1c,32'hfedcba98);
write_reg(8'h20,32'h13579bdf);write_reg(8'h24,32'h2468ace0);
read_reg(8'h00,32'h00000001);read_reg(8'h10,32'h89abcdef);
read_reg(8'h14,32'h01234567);read_reg(8'h18,32'h76543210);
read_reg(8'h1c,32'hfedcba98);read_reg(8'h20,32'h13579bdf);
read_reg(8'h24,32'h2468ace0);
$display("tb_ddit_block_kv260_axi_top: PASS");$finish;end
initial begin repeat(300)@(posedge clk);$fatal(1,"timeout");end endmodule
""",encoding="utf-8")
    build=tmp_path/"tb_ddit_block_kv260_axi_top"
    comp=subprocess.run(
        ["iverilog","-g2012","-Wall","-s","tb_ddit_block_kv260_axi_top",
         "-o",str(build),*(str(path) for path in sorted(RTL.glob("*.sv"))),
         str(tb)],cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert comp.returncode==0,comp.stderr
    run=subprocess.run(["vvp",str(build)],cwd=ROOT,capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert "tb_ddit_block_kv260_axi_top: PASS" in run.stdout
