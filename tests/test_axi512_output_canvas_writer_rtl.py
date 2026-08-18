from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]
RTL=ROOT/"rtl/tensor_engine"


def test_output_canvas_writer_emits_two_beat_sparse_records(tmp_path:Path)->None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    tb=tmp_path/"tb_axi512_output_canvas_writer.sv"
    tb.write_text("""`timescale 1ns/1ps
module tb_axi512_output_canvas_writer;reg clk=0,rst_n=0,input_valid=0;
reg [9:0] tile=0;reg [3:0] group=0;reg [575:0] data=0;
reg awready=0,wready=0,bvalid=0;wire input_ready,busy,done,error;
wire [63:0] awaddr,bytes,tx;wire [7:0] awlen;wire [2:0] awsize;
wire [1:0] awburst;wire awvalid;wire [511:0] wdata;wire [63:0] wstrb;
wire wlast,wvalid,bready;integer cycles=0,beats=0;
axi512_output_canvas_writer dut(.clk(clk),.rst_n(rst_n),
.output_base_address(64'h8000),.input_valid(input_valid),.input_ready(input_ready),
.input_tile(tile),.input_group(group),.input_q10_packed(data),
.m_axi_awaddr(awaddr),.m_axi_awlen(awlen),.m_axi_awsize(awsize),
.m_axi_awburst(awburst),.m_axi_awvalid(awvalid),.m_axi_awready(awready),
.m_axi_wdata(wdata),.m_axi_wstrb(wstrb),.m_axi_wlast(wlast),
.m_axi_wvalid(wvalid),.m_axi_wready(wready),.m_axi_bresp(0),
.m_axi_bvalid(bvalid),.m_axi_bready(bready),.busy(busy),.done(done),
.protocol_error(error),.bytes_written(bytes),.write_transactions(tx));
always #2 clk=~clk;always @(posedge clk)begin cycles=cycles+1;
awready<=awvalid&&cycles[0];wready<=wvalid&&!cycles[0];
if(awvalid&&awready)begin if(awaddr!==64'h8000+(3*16+5)*128||awlen!==1||
awsize!==6||awburst!==1)$fatal(1,"output address mismatch");end
if(wvalid&&wready)begin if(beats==0)begin
if(wdata!==data[511:0]||wstrb!==64'hffffffffffffffff||wlast)
$fatal(1,"output first beat mismatch");beats=1;end else begin
if(wdata[63:0]!==data[575:512]||wdata[511:64]!==0||
wstrb!==64'hff||!wlast)$fatal(1,"output second beat mismatch");
beats=2;bvalid<=1;end end
if(bvalid&&bready)bvalid<=0;end
initial begin data={{2{32'habcdef01}},{16{32'h12345678}}};
repeat(3)@(posedge clk);@(negedge clk);rst_n=1;tile=3;group=5;input_valid=1;
wait(input_ready);@(posedge clk);@(negedge clk);input_valid=0;wait(done);
repeat(2)@(posedge clk);#1;if(error||busy||beats!=2||bytes!=72||tx!=1)
$fatal(1,"output completion mismatch");
$display("tb_axi512_output_canvas_writer: PASS cycles=%0d bytes=%0d",cycles,bytes);
$finish;end initial begin repeat(100)@(posedge clk);$fatal(1,"timeout");end endmodule
""",encoding="utf-8")
    build=tmp_path/"tb_axi512_output_canvas_writer"
    comp=subprocess.run(
        ["iverilog","-g2012","-Wall","-s","tb_axi512_output_canvas_writer",
         "-o",str(build),str(RTL/"axi512_output_canvas_writer.sv"),str(tb)],
        cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert comp.returncode==0,comp.stderr
    run=subprocess.run(["vvp",str(build)],cwd=ROOT,capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert "tb_axi512_output_canvas_writer: PASS" in run.stdout
