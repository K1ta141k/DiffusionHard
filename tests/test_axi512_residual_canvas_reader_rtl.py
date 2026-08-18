from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]
RTL=ROOT/"rtl/tensor_engine"


def test_residual_canvas_reader_unpacks_two_beat_records(tmp_path:Path)->None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    tb=tmp_path/"tb_axi512_residual_canvas_reader.sv"
    tb.write_text("""`timescale 1ns/1ps
module tb_axi512_residual_canvas_reader;
reg clk=0,rst_n=0,start=0,arready=0,rvalid=0,rlast=0;
reg [511:0] rdata=0;wire start_ready,load_valid,busy,done,error;
wire [3:0] group;wire [6:0] tile;wire [575:0] load_data;
wire [63:0] araddr,bytes,tx;wire [7:0] arlen;wire [2:0] arsize;
wire [1:0] arburst;wire arvalid,rready;integer cycles=0,beat=0,seen=0,lane;
reg [575:0] expected;
axi512_residual_canvas_reader #(.RECORDS(4)) dut(.clk(clk),.rst_n(rst_n),
.start(start),.start_ready(start_ready),.input_base_address(64'h1000),
.residual_load_valid(load_valid),.residual_load_group(group),
.residual_load_output_tile(tile),.residual_load_q10_packed(load_data),
.m_axi_araddr(araddr),.m_axi_arlen(arlen),.m_axi_arsize(arsize),
.m_axi_arburst(arburst),.m_axi_arvalid(arvalid),.m_axi_arready(arready),
.m_axi_rdata(rdata),.m_axi_rresp(0),.m_axi_rlast(rlast),
.m_axi_rvalid(rvalid),.m_axi_rready(rready),.busy(busy),.done(done),
.protocol_error(error),.bytes_read(bytes),.read_transactions(tx));
always #2 clk=~clk;always @(posedge clk)begin cycles=cycles+1;arready<=arvalid;
if(arvalid&&arready)begin
if(araddr!==64'h1000+seen*128||arlen!==1||arsize!==6||arburst!==1)
$fatal(1,"residual AXI command mismatch");beat=0;rvalid<=1;rlast<=0;
rdata<={16{32'h10000000+seen}};end else if(rvalid&&rready)begin
if(beat==0)begin beat=1;rlast<=1;rdata<={16{32'h20000000+seen}};end
else begin rvalid<=0;rlast<=0;end end
#1;if(load_valid)begin expected={{2{32'h20000000+seen}},
{16{32'h10000000+seen}}};
if(group!==0||tile!==seen||load_data!==expected)$fatal(1,"residual data mismatch");
seen=seen+1;end end
initial begin repeat(3)@(posedge clk);@(negedge clk);rst_n=1;start=1;
wait(start_ready);@(posedge clk);@(negedge clk);start=0;wait(done);
repeat(3)@(posedge clk);#1;if(error||busy||seen!=4||tx!=4||bytes!=512)
$fatal(1,"residual completion mismatch");
$display("tb_axi512_residual_canvas_reader: PASS cycles=%0d records=%0d bytes=%0d",
cycles,seen,bytes);$finish;end
initial begin repeat(200)@(posedge clk);$fatal(1,"timeout");end endmodule
""",encoding="utf-8")
    build=tmp_path/"tb_axi512_residual_canvas_reader"
    comp=subprocess.run(
        ["iverilog","-g2012","-Wall","-s","tb_axi512_residual_canvas_reader",
         "-o",str(build),str(RTL/"axi512_read_burst_master.sv"),
         str(RTL/"axi512_residual_canvas_reader.sv"),str(tb)],
        cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert comp.returncode==0,comp.stderr
    run=subprocess.run(["vvp",str(build)],cwd=ROOT,capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert "tb_axi512_residual_canvas_reader: PASS" in run.stdout
