from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from diffusion_accel.attention_int8 import dynamic_qk_fixed_pv_attention_q12
from diffusion_accel.fixed_attention import (
    fixed_attention_projection_q10,
    fixed_attention_q12,
    fixed_qkv_projection_q12,
)
from diffusion_accel.fixed_mlp import (
    _integer_matmul,
    _load_tensors,
    _quantize_weight_per_output,
    hardware_gelu_q10,
    quantize_up_activation_fixed,
)
from diffusion_accel.fixed_norm import fixed_layer_norm_q12


ROOT=Path(__file__).resolve().parents[1]
RTL=ROOT/"rtl/tensor_engine"
PACKAGE=ROOT/"data/hardware/mdlm-owt-169m-h0"


def _packed_hex(values:list[int],width:int)->str:
    mask=(1<<width)-1
    packed=sum((value&mask)<<(index*width) for index,value in enumerate(values))
    return f"{packed:0{(len(values)*width+3)//4}x}"


def _build_fixed_block_reference(*,packed_attention:bool=False)->tuple[
    torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor
]:
    goldens=_load_tensors(
        PACKAGE/"golden_tensors.safetensors",
        ["folded.embedding","folded.block_00.norm2_unaffine"],
    )
    weights=_load_tensors(
        PACKAGE/"folded_fp16_weights.safetensors",
        [
            "block_00.qkv.weight","block_00.qkv.bias",
            "rotary.cos","rotary.sin","block_00.attention_out.weight",
            "block_00.mlp_up.weight",
        ],
    )
    artifacts=_load_tensors(
        PACKAGE/"mlp_interstage_int8.safetensors",
        [
            "block_00.up_smoothing_reciprocal_q15",
            "block_00.up_output_factor","block_00.up_bias_q10",
            "block_00.interstage_multiplier","block_00.down_weight",
            "block_00.down_output_multiplier","block_00.down_bias_q10",
        ],
    )
    embedding=goldens["folded.embedding"][0]
    residual_q10=torch.round(embedding.double()*1024.0).to(torch.int64)
    residual_q10=residual_q10.clamp(-(1<<23),(1<<23)-1)
    _,norm1_q12,_=fixed_layer_norm_q12(embedding)
    qkv,_,_=fixed_qkv_projection_q12(
        norm1_q12,weights["block_00.qkv.weight"],weights["block_00.qkv.bias"]
    )
    if packed_attention:
        attention_q12,_=dynamic_qk_fixed_pv_attention_q12(
            qkv,weights["rotary.cos"].float(),weights["rotary.sin"].float()
        )
    else:
        _,attention_q12,_=fixed_attention_q12(
            qkv,weights["rotary.cos"].float(),weights["rotary.sin"].float()
        )
    _,projection_q10,_=fixed_attention_projection_q10(
        attention_q12,weights["block_00.attention_out.weight"]
    )
    after_attention_q10=(residual_q10+projection_q10).clamp(
        -(1<<23),(1<<23)-1
    )

    after_attention=after_attention_q10.float()/1024.0
    fixed_norm,_,_=fixed_layer_norm_q12(after_attention)
    calibration_norm=goldens["folded.block_00.norm2_unaffine"][0]
    up_weight=weights["block_00.mlp_up.weight"]
    smoothing=(
        calibration_norm.abs().amax(dim=0).clamp_min(1e-8).pow(0.75)
        /up_weight.abs().amax(dim=0).clamp_min(1e-8).pow(0.25)
    ).clamp_min(1e-8)
    activation_q,token_factors,activation_tensors,_=(
        quantize_up_activation_fixed(fixed_norm,smoothing)
    )
    assert torch.equal(
        activation_tensors["reciprocal"].to(torch.int64),
        artifacts["block_00.up_smoothing_reciprocal_q15"].to(torch.int64),
    )
    up_weight_q,_=_quantize_weight_per_output(up_weight*smoothing[None,:],8)
    up_accumulators=_integer_matmul(activation_q,up_weight_q.t(),8).to(torch.int64)
    up_multipliers=(
        token_factors.to(torch.int64)[:,None]
        *artifacts["block_00.up_output_factor"].to(torch.int64)[None,:]
        +128
    )>>8
    up_products=up_accumulators*up_multipliers
    up_q=torch.sign(up_products)*((up_products.abs()+(1<<19))>>20)
    up_q=(
        up_q+artifacts["block_00.up_bias_q10"].to(torch.int64)[None,:]
    ).clamp(-(1<<15),(1<<15)-1)
    _,gelu_q=hardware_gelu_q10(up_q)
    interstage_products=(
        gelu_q.to(torch.int64)
        *artifacts["block_00.interstage_multiplier"].to(torch.int64)[None,:]
    )
    interstage_q=torch.sign(interstage_products)*(
        (interstage_products.abs()+(1<<19))>>20
    )
    interstage_q=interstage_q.clamp(-127,127).to(torch.int8)
    down_accumulators=_integer_matmul(
        interstage_q,artifacts["block_00.down_weight"].to(torch.int8).t(),8
    ).to(torch.int64)
    down_products=(
        down_accumulators
        *artifacts["block_00.down_output_multiplier"].to(torch.int64)[None,:]
    )
    down_q=torch.sign(down_products)*((down_products.abs()+(1<<19))>>20)
    down_q=(
        down_q+artifacts["block_00.down_bias_q10"].to(torch.int64)[None,:]
    ).clamp(-(1<<23),(1<<23)-1)
    block_output_q10=(after_attention_q10+down_q).clamp(-(1<<23),(1<<23)-1)
    return residual_q10,norm1_q12,after_attention_q10,block_output_q10


@pytest.mark.skipif(
    os.environ.get("DIFFUSION_ACCEL_RUN_FULL_BLOCK_RTL")!="1",
    reason="set DIFFUSION_ACCEL_RUN_FULL_BLOCK_RTL=1 for full real block-0 RTL",
)
@pytest.mark.parametrize(
    "packed_attention",[False,True],ids=["fixed18","packed-m8"]
)
def test_full_block_zero_uses_real_execution_image_and_matches_fixed_reference(
    tmp_path:Path,
    packed_attention:bool,
)->None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    image_path=PACKAGE/"block_00_execution.bin"
    if not image_path.is_file():
        pytest.skip("H0 hardware package is unavailable")
    residual_q10,norm1_q12,after_attention_q10,expected_q10=(
        _build_fixed_block_reference(packed_attention=packed_attention)
    )

    image=image_path.read_bytes()
    assert len(image)==9_064_448 and len(image)%64==0
    image_hex=tmp_path/"block_00_execution.hex"
    image_hex.write_text(
        "\n".join(
            f"{int.from_bytes(image[offset:offset+64],'little'):0128x}"
            for offset in range(0,len(image),64)
        )+"\n",encoding="utf-8",
    )
    residual_hex=tmp_path/"block_00_residual_q10.hex"
    residual_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(residual_q10[group*4+token,tile*6+lane])
                    for token in range(4) for lane in range(6)
                ],24,
            )
            for group in range(16) for tile in range(128)
        )+"\n",encoding="utf-8",
    )
    attention_hex=tmp_path/"block_00_after_attention_q10.hex"
    attention_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(after_attention_q10[group*4+token,tile*6+lane])
                    for token in range(4) for lane in range(6)
                ],24,
            )
            for tile in range(128) for group in range(16)
        )+"\n",encoding="utf-8",
    )
    expected_hex=tmp_path/"block_00_expected_q10.hex"
    expected_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(expected_q10[group*4+token,tile*6+lane])
                    for token in range(4) for lane in range(6)
                ],24,
            )
            for tile in range(128) for group in range(16)
        )+"\n",encoding="utf-8",
    )

    packed_parameter=",.PACKED_ATTENTION(1)" if packed_attention else ""
    attention_hierarchy=(
        "dut.dense.compute.packed_attention_path.attention"
        if packed_attention
        else "dut.dense.compute.fixed_attention_path.attention"
    )
    testbench=tmp_path/"tb_ddit_block_with_image_fabric_h0.sv"
    testbench.write_text(f"""`timescale 1ns/1ps
module tb_ddit_block_with_image_fabric_h0;
  reg clk=0,rst_n=0,preload_start=0,block_start=0,residual_load=0;
  reg [3:0] residual_group=0;reg [6:0] residual_tile=0;
  reg [575:0] residual_data=0;reg normalized_data_valid=0;
  reg [2303:0] normalized_data=0;reg arready=0,rvalid=0,rlast=0;
  reg [511:0] rdata=0;wire preload_ready,preload_done,constants_loaded;
  wire block_ready,busy,done,normalized_read_valid,output_valid;
  wire [3:0] normalized_group,output_group;wire [4:0] normalized_tile;
  wire [9:0] output_tile;wire [575:0] outputs;
  wire [63:0] araddr,total_tx,constant_tx,constant_bytes,dense_tx;
  wire [7:0] arlen;wire [2:0] arsize;wire [1:0] arburst;
  wire arvalid,rready,error,attention_busy,mlp_busy;wire [255:0] dense_bytes;
  reg [511:0] image_mem[0:141631];reg [575:0] residual_mem[0:2047];
  reg [575:0] attention_mem[0:2047];
  reg [575:0] expected_mem[0:2047];
  integer cycles=0,index,source_beat=0,source_beats=0,source_line=0;
  integer attention_seen=0,outputs_seen=0;reg saw_attention=0,saw_mlp=0;
  integer preload_cycles=0,norm1_cycles=0,attention_cycles=0,mlp_cycles=0;
  integer qkv_cycles=0,projection_cycles=0;
  integer qkv_staging_cycles=0,head_attention_cycles=0;
  integer mlp_frontend_cycles=0,mlp_up_load_cycles=0,mlp_up_cycles=0;
  integer mlp_down_load_cycles=0,mlp_down_cycles=0;
  ddit_block_with_image_fabric #(
    .INTERNAL_NORM1(1){packed_parameter}
  ) dut(.clk(clk),.rst_n(rst_n),
    .block_base_address(0),.preload_start(preload_start),
    .preload_start_ready(preload_ready),.preload_done(preload_done),
    .constants_loaded(constants_loaded),.block_start(block_start),
    .block_start_ready(block_ready),.busy(busy),.done(done),
    .residual_load_valid(residual_load),.residual_load_group(residual_group),
    .residual_load_output_tile(residual_tile),
    .residual_load_q10_packed(residual_data),
    .normalized_read_valid(normalized_read_valid),
    .normalized_read_group(normalized_group),
    .normalized_read_input_tile(normalized_tile),
    .normalized_read_data_valid(normalized_data_valid),
    .normalized_q12_packed(normalized_data),.output_valid(output_valid),
    .output_tile(output_tile),.output_group(output_group),
    .outputs_packed(outputs),.m_axi_araddr(araddr),.m_axi_arlen(arlen),
    .m_axi_arsize(arsize),.m_axi_arburst(arburst),.m_axi_arvalid(arvalid),
    .m_axi_arready(arready),.m_axi_rdata(rdata),.m_axi_rresp(0),
    .m_axi_rlast(rlast),.m_axi_rvalid(rvalid),.m_axi_rready(rready),
    .protocol_error(error),.read_transactions(total_tx),
    .constant_read_transactions(constant_tx),
    .constant_bytes_read(constant_bytes),.dense_client_bytes_read(dense_bytes),
    .dense_read_transactions(dense_tx),.attention_busy(attention_busy),
    .mlp_busy(mlp_busy));
  always #2 clk=~clk;
  always @(posedge clk) begin
    cycles=cycles+1;normalized_data_valid<=0;normalized_data<=0;
    arready<=arvalid&&cycles[0];
    if(arvalid&&arready) begin
      if(arsize!==6||arburst!==1)$fatal(1,"full block AXI attribute mismatch");
      source_line=araddr>>6;source_beat=0;source_beats=arlen+1;
      rdata<=image_mem[araddr>>6];rvalid<=1;rlast<=source_beats==1;
    end else if(rvalid&&rready) begin
      source_beat=source_beat+1;
      if(source_beat==source_beats) begin rvalid<=0;rlast<=0;end
      else begin source_line=source_line+1;rdata<=image_mem[source_line];
        rlast<=source_beat==source_beats-1;end
    end
    if(attention_busy)saw_attention=1;if(mlp_busy)saw_mlp=1;
    if(dut.preloader_busy)preload_cycles=preload_cycles+1;
    if(dut.dense.compute.state==1)norm1_cycles=norm1_cycles+1;
    if(dut.dense.compute.state==2)attention_cycles=attention_cycles+1;
    if(dut.dense.compute.state==3)mlp_cycles=mlp_cycles+1;
    if({attention_hierarchy}.producer_busy)qkv_cycles=qkv_cycles+1;
    if({attention_hierarchy}.projection_busy)
      projection_cycles=projection_cycles+1;
    if({attention_hierarchy}.producer.head_staging_busy)
      qkv_staging_cycles=qkv_staging_cycles+1;
    if({attention_hierarchy}.producer.head_attention_busy)
      head_attention_cycles=head_attention_cycles+1;
    if(dut.dense.compute.mlp.controller.state==1 ||
       dut.dense.compute.mlp.controller.state==2)
      mlp_frontend_cycles=mlp_frontend_cycles+1;
    if(dut.dense.compute.mlp.controller.state==3)
      mlp_up_load_cycles=mlp_up_load_cycles+1;
    if(dut.dense.compute.mlp.controller.state==4 ||
       dut.dense.compute.mlp.controller.state==5 ||
       dut.dense.compute.mlp.controller.state==6)
      mlp_up_cycles=mlp_up_cycles+1;
    if(dut.dense.compute.mlp.controller.state==7)
      mlp_down_load_cycles=mlp_down_load_cycles+1;
    if(dut.dense.compute.mlp.controller.state==8 ||
       dut.dense.compute.mlp.controller.state==9)
      mlp_down_cycles=mlp_down_cycles+1;
    #1;
    if(dut.dense.attention_tile_valid) begin
      if(dut.dense.attention_tile_output!==attention_seen/16 ||
         dut.dense.attention_tile_group!==attention_seen%16 ||
         dut.dense.attention_tile_data!==attention_mem[attention_seen])
        $fatal(1,"full block attention mismatch index=%0d tile=%0d group=%0d",
          attention_seen,dut.dense.attention_tile_output,
          dut.dense.attention_tile_group);
      attention_seen=attention_seen+1;
    end
    if(output_valid) begin
      if(output_tile!==outputs_seen/16 || output_group!==outputs_seen%16 ||
         outputs!==expected_mem[outputs_seen])
        $fatal(1,"full block output mismatch index=%0d tile=%0d group=%0d",
          outputs_seen,output_tile,output_group);
      outputs_seen=outputs_seen+1;
    end
  end
  initial begin
    $readmemh("{image_hex}",image_mem);$readmemh("{residual_hex}",residual_mem);
    $readmemh("{attention_hex}",attention_mem);$readmemh("{expected_hex}",expected_mem);
    repeat(3)@(posedge clk);@(negedge clk);rst_n=1;preload_start=1;
    wait(preload_ready);@(posedge clk);@(negedge clk);preload_start=0;
    wait(preload_done);@(posedge clk);#1;
    if(!constants_loaded||!block_ready)$fatal(1,"full block preload gate failed");
    for(index=0;index<2048;index=index+1)begin
      @(negedge clk);residual_load=1;residual_group=index/128;
      residual_tile=index%128;residual_data=residual_mem[index];end
    @(negedge clk);residual_load=0;block_start=1;
    wait(block_ready);@(posedge clk);@(negedge clk);block_start=0;
    wait(done);repeat(4)@(posedge clk);#1;
    if(error||!saw_attention||!saw_mlp||busy||attention_seen!=2048||
       outputs_seen!=2048||constant_tx!=176||constant_bytes!=11264||
       dense_tx!=38316||total_tx!=38492||dense_bytes[63:0]!=3674880||
       dense_bytes[127:64]!=598016||dense_bytes[191:128]!=2392064||
       dense_bytes[255:192]!=2383872)
      $fatal(1,"full block completion mismatch tx=%0d/%0d/%0d bytes=%0d/%0d/%0d/%0d outputs=%0d/%0d",
        total_tx,constant_tx,dense_tx,dense_bytes[63:0],dense_bytes[127:64],
        dense_bytes[191:128],dense_bytes[255:192],attention_seen,outputs_seen);
    $display("tb_ddit_block_with_image_fabric_h0: PASS cycles=%0d transactions=%0d bytes=%0d attention=%0d outputs=%0d",
      cycles,total_tx,constant_bytes+dense_bytes[63:0]+dense_bytes[127:64]+
      dense_bytes[191:128]+dense_bytes[255:192],attention_seen,outputs_seen);
    $display("tb_ddit_block_with_image_fabric_h0: PROFILE preload=%0d norm1=%0d attention=%0d qkv=%0d projection=%0d mlp=%0d",
      preload_cycles,norm1_cycles,attention_cycles,qkv_cycles,
      projection_cycles,mlp_cycles);
    $display("tb_ddit_block_with_image_fabric_h0: MLP_PROFILE frontend=%0d up_load=%0d up=%0d down_load=%0d down=%0d",
      mlp_frontend_cycles,mlp_up_load_cycles,mlp_up_cycles,
      mlp_down_load_cycles,mlp_down_cycles);
    $display("tb_ddit_block_with_image_fabric_h0: ATTN_PROFILE qkv_staging=%0d head_attention=%0d projection=%0d",
      qkv_staging_cycles,head_attention_cycles,projection_cycles);
    $finish;
  end
  initial begin repeat(2000000)@(posedge clk);
    $display("timeout cycles=%0d state=%0d attention=%0d mlp=%0d tx=%0d outputs=%0d",
      cycles,dut.dense.compute.state,{attention_hierarchy}.state,
      dut.dense.compute.mlp.controller.state,total_tx,outputs_seen);
    $fatal(1,"timeout");end
endmodule
""",encoding="utf-8")
    build=tmp_path/"tb_ddit_block_with_image_fabric_h0"
    compile_result=subprocess.run(
        ["iverilog","-g2012","-Wall","-s","tb_ddit_block_with_image_fabric_h0",
         "-o",str(build),*(str(path) for path in sorted(RTL.glob("*.sv"))),
         str(testbench)],cwd=ROOT,check=False,capture_output=True,text=True,
    )
    assert compile_result.returncode==0,compile_result.stderr
    run_result=subprocess.run(
        ["vvp",str(build)],cwd=ROOT,check=False,capture_output=True,text=True,
        timeout=10800,
    )
    if os.environ.get("DIFFUSION_ACCEL_SHOW_LONG_RTL_OUTPUT")=="1":
        print(run_result.stdout,end="")
    assert run_result.returncode==0,run_result.stdout+run_result.stderr
    assert "tb_ddit_block_with_image_fabric_h0: PASS" in run_result.stdout
    match=re.search(
        r"cycles=(\d+) transactions=(\d+) bytes=(\d+) attention=(\d+) outputs=(\d+)",
        run_result.stdout,
    )
    assert match is not None
    assert int(match.group(2))==38_492
    assert int(match.group(3))==9_060_096
    assert int(match.group(4))==2_048
    assert int(match.group(5))==2_048
    profile=re.search(
        r"PROFILE preload=(\d+) norm1=(\d+) attention=(\d+) qkv=(\d+) "
        r"projection=(\d+) mlp=(\d+)",run_result.stdout,
    )
    assert profile is not None
    assert sum(int(value) for value in profile.groups()[3:5])<=int(profile.group(3))
    mlp_profile=re.search(
        r"MLP_PROFILE frontend=(\d+) up_load=(\d+) up=(\d+) "
        r"down_load=(\d+) down=(\d+)",run_result.stdout,
    )
    assert mlp_profile is not None
    assert sum(int(value) for value in mlp_profile.groups())<=int(profile.group(6))
    attention_profile=re.search(
        r"ATTN_PROFILE qkv_staging=(\d+) head_attention=(\d+) "
        r"projection=(\d+)",run_result.stdout,
    )
    assert attention_profile is not None
    assert sum(int(value) for value in attention_profile.groups()[:2])<=int(profile.group(4))
