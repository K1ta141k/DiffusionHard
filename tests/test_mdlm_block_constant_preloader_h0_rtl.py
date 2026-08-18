from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
import pytest
from diffusion_accel.fixed_mlp import _load_tensors

ROOT=Path(__file__).resolve().parents[1]
RTL=ROOT/"rtl/tensor_engine"
PACKAGE=ROOT/"data/hardware/mdlm-owt-169m-h0"

def test_real_rotary_and_reciprocal_tables_preload_from_one_axi_port(tmp_path:Path)->None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    image_path=PACKAGE/"block_00_execution.bin"
    weight_path=PACKAGE/"folded_fp16_weights.safetensors"
    artifact_path=PACKAGE/"mlp_interstage_int8.safetensors"
    if not image_path.is_file(): pytest.skip("H0 hardware package is unavailable")
    weights=_load_tensors(weight_path,["rotary.cos","rotary.sin"])
    artifacts=_load_tensors(artifact_path,["block_00.up_smoothing_reciprocal_q15"])
    import torch
    cosine=torch.round(weights["rotary.cos"].float()*32768).clamp(-32768,32767).to(torch.int16)
    sine=torch.round(weights["rotary.sin"].float()*32768).clamp(-32768,32767).to(torch.int16)
    rotary=[((int(sine[t,p])&0xffff)<<16)|(int(cosine[t,p])&0xffff)
            for t in range(64) for p in range(32)]
    reciprocal=[int(v)&0x3ffff for v in artifacts["block_00.up_smoothing_reciprocal_q15"]]
    image=image_path.read_bytes();lines=[]
    for offset,count in [(3_678_208,128),(4_284_416,48)]:
        lines.extend(image[offset+i*64:offset+(i+1)*64] for i in range(count))
    line_hex=tmp_path/"constant_lines.hex";rot_hex=tmp_path/"rotary_expected.hex"
    rec_hex=tmp_path/"reciprocal_expected.hex"
    line_hex.write_text("\n".join(f"{int.from_bytes(x,'little'):0128x}" for x in lines)+"\n")
    rot_hex.write_text("\n".join(f"{x:08x}" for x in rotary)+"\n")
    rec_hex.write_text("\n".join(f"{x:05x}" for x in reciprocal)+"\n")
    tb=tmp_path/"tb_constant_preloader.sv"
    tb.write_text(f"""`timescale 1ns/1ps
module tb_constant_preloader;reg clk=0,rst_n=0,start=0,arready=0,rvalid=0;
reg [9:0] lookup=0;wire ready,rv;wire [5:0] token;wire [4:0] pair;
wire [15:0] cos,sin;wire [17:0] reciprocal;wire [63:0] araddr,bytes,tx;
wire [7:0] arlen;wire [2:0] arsize;wire [1:0] arburst;wire arvalid,rready;
wire busy,done,error;reg [511:0] line_mem[0:175];reg [31:0] rot_mem[0:2047];
reg [17:0] rec_mem[0:767];integer cycle=0,line=0,seen=0,i;
mdlm_block_constant_preloader dut(.clk(clk),.rst_n(rst_n),.start(start),
.start_ready(ready),.block_base_address(0),.rotary_load_valid(rv),
.rotary_load_token(token),.rotary_load_pair(pair),.rotary_load_cosine_q15(cos),
.rotary_load_sine_q15(sin),.reciprocal_lookup_channel(lookup),
.reciprocal_lookup_q15(reciprocal),.m_axi_araddr(araddr),.m_axi_arlen(arlen),
.m_axi_arsize(arsize),.m_axi_arburst(arburst),.m_axi_arvalid(arvalid),
.m_axi_arready(arready),.m_axi_rdata(line_mem[line]),.m_axi_rresp(0),
.m_axi_rlast(1),.m_axi_rvalid(rvalid),.m_axi_rready(rready),.busy(busy),
.done(done),.protocol_error(error),.bytes_read(bytes),.read_transactions(tx));
always #2 clk=~clk;always @(posedge clk) begin cycle=cycle+1;arready<=arvalid&&cycle[0];
if(arvalid&&arready) begin
 if(araddr>=4284416) line=128+(araddr-4284416)/64;
 else line=(araddr-3678208)/64;rvalid<=1;end else if(rvalid&&rready) rvalid<=0;
#1;if(rv) begin if({{sin,cos}}!==rot_mem[seen]||token!==seen/32||pair!==seen%32)
$fatal(1,"rotary mismatch %0d",seen);seen=seen+1;end end
initial begin $readmemh("{line_hex}",line_mem);$readmemh("{rot_hex}",rot_mem);
$readmemh("{rec_hex}",rec_mem);repeat(3) @(posedge clk);@(negedge clk);rst_n=1;start=1;
@(posedge clk);@(negedge clk);start=0;wait(done);#1;
for(i=0;i<768;i=i+1) begin lookup=i;#1;if(reciprocal!==rec_mem[i])
$fatal(1,"reciprocal mismatch %0d",i);end
if(error||seen!=2048||bytes!=11264||tx!=176||busy)$fatal(1,"preload completion mismatch");
$display("tb_constant_preloader: PASS rotary=%0d reciprocal=768 tx=%0d bytes=%0d",seen,tx,bytes);$finish;end
initial begin repeat(10000) @(posedge clk);$fatal(1,"timeout");end endmodule
""")
    build=tmp_path/"tb_constant_preloader"
    sources=["mdlm_block_parameter_address_generator.sv","axi512_read_burst_master.sv",
      "mdlm_block_parameter_dma.sv","mdlm_block_constant_preloader.sv"]
    comp=subprocess.run(["iverilog","-g2012","-Wall","-s","tb_constant_preloader",
      "-o",str(build),*(str(RTL/s) for s in sources),str(tb)],cwd=ROOT,capture_output=True,text=True)
    assert comp.returncode==0,comp.stderr
    run=subprocess.run(["vvp",str(build)],cwd=ROOT,capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert "tb_constant_preloader: PASS" in run.stdout
