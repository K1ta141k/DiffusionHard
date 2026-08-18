from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from diffusion_accel.fixed_mlp import _load_tensors
from diffusion_accel.fixed_norm import fixed_layer_norm_q12


ROOT=Path(__file__).resolve().parents[1]
RTL=ROOT/"rtl/tensor_engine"
PACKAGE=ROOT/"data/hardware/mdlm-owt-169m-h0"


def _packed_hex(values:list[int],width:int)->str:
    mask=(1<<width)-1
    packed=sum((value&mask)<<(index*width) for index,value in enumerate(values))
    return f"{packed:0{(len(values)*width+3)//4}x}"


def test_residual_canvas_precomputes_all_norm1_tiles_exactly(tmp_path:Path)->None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path=PACKAGE/"golden_tensors.safetensors"
    if not golden_path.is_file():
        pytest.skip("H0 hardware package is unavailable")
    embedding=_load_tensors(golden_path,["folded.embedding"])["folded.embedding"][0]
    residual_q10=(embedding.double()*1024.0).round().to(dtype=torch.int64)
    _,normalized_q12,_=fixed_layer_norm_q12(embedding)
    residual_hex=tmp_path/"norm1_residual.hex"
    expected_hex=tmp_path/"norm1_expected.hex"
    residual_hex.write_text("\n".join(
        _packed_hex([
            int(residual_q10[group*4+token,tile*6+lane])
            for token in range(4) for lane in range(6)
        ],24) for group in range(16) for tile in range(128)
    )+"\n",encoding="utf-8")
    expected_hex.write_text("\n".join(
        _packed_hex([
            int(normalized_q12[group*4+token,tile*32+lane])
            for token in range(4) for lane in range(32)
        ],18) for group in range(16) for tile in range(24)
    )+"\n",encoding="utf-8")
    tb=tmp_path/"tb_hidden_canvas_norm1_precompute_h0.sv"
    tb.write_text(f"""`timescale 1ns/1ps
module tb_hidden_canvas_norm1_precompute_h0;
reg clk=0,rst_n=0,start=0,load_valid=0,read_valid=0;
reg [3:0] load_group=0,read_group=0;reg [6:0] load_tile=0;
reg [4:0] read_tile=0;reg [575:0] load_data=0;
wire start_ready,canvas_read_valid,canvas_data_valid,norm_data_valid,busy,done;
wire [3:0] canvas_group;wire [6:0] canvas_tile;wire [575:0] canvas_data;
wire [2303:0] norm_data;reg [575:0] residual_mem[0:2047];
reg [2303:0] expected_mem[0:383];integer index,cycles=0;
attention_residual_canvas_uram canvas(.clk(clk),.load_valid(load_valid),
.load_group(load_group),.load_output_tile(load_tile),.load_data_packed(load_data),
.read_valid(canvas_read_valid),.read_group(canvas_group),
.read_output_tile(canvas_tile),.read_data_valid(canvas_data_valid),
.read_data_packed(canvas_data));
hidden_canvas_norm1_precompute dut(.clk(clk),.rst_n(rst_n),.start(start),
.start_ready(start_ready),.canvas_read_valid(canvas_read_valid),
.canvas_read_group(canvas_group),.canvas_read_output_tile(canvas_tile),
.canvas_read_data_valid(canvas_data_valid),.canvas_read_q10_packed(canvas_data),
.normalized_read_valid(read_valid),.normalized_read_group(read_group),
.normalized_read_input_tile(read_tile),.normalized_read_data_valid(norm_data_valid),
.normalized_q12_packed(norm_data),.busy(busy),.done(done));
always #2 clk=~clk;always @(posedge clk)cycles=cycles+1;
initial begin $readmemh("{residual_hex}",residual_mem);
$readmemh("{expected_hex}",expected_mem);repeat(3)@(posedge clk);
@(negedge clk);rst_n=1;
for(index=0;index<2048;index=index+1)begin @(negedge clk);load_valid=1;
load_group=index/128;load_tile=index%128;load_data=residual_mem[index];end
@(negedge clk);load_valid=0;start=1;wait(start_ready);@(posedge clk);
@(negedge clk);start=0;wait(done);@(posedge clk);#1;
for(index=0;index<384;index=index+1)begin @(negedge clk);read_valid=1;
read_group=index/24;read_tile=index%24;@(posedge clk);#1;
if(!norm_data_valid||norm_data!==expected_mem[index])
$fatal(1,"norm1 tile mismatch index=%0d group=%0d tile=%0d",index,read_group,read_tile);
@(negedge clk);read_valid=0;end
if(busy)$fatal(1,"norm1 remained busy");
$display("tb_hidden_canvas_norm1_precompute_h0: PASS cycles=%0d tiles=384",cycles);
$finish;end
initial begin repeat(100000)@(posedge clk);$fatal(1,"timeout");end endmodule
""",encoding="utf-8")
    build=tmp_path/"tb_hidden_canvas_norm1_precompute_h0"
    sources=[
        "attention_residual_canvas_uram.sv","hidden_canvas_group_replay.sv",
        "unsigned_divider_iterative.sv","unsigned_sqrt_iterative.sv",
        "layer_norm_q12_group.sv","hidden_canvas_norm1_precompute.sv",
    ]
    comp=subprocess.run(
        ["iverilog","-g2012","-Wall","-s","tb_hidden_canvas_norm1_precompute_h0",
         "-o",str(build),*(str(RTL/source) for source in sources),str(tb)],
        cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert comp.returncode==0,comp.stderr
    run=subprocess.run(
        ["vvp",str(build)],cwd=ROOT,check=False,capture_output=True,text=True,
        timeout=120,
    )
    assert run.returncode==0,run.stdout+run.stderr
    assert "tb_hidden_canvas_norm1_precompute_h0: PASS" in run.stdout
