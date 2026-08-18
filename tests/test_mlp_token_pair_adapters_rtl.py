from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_mlp_token_pair_adapters_preserve_lane_order(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    testbench = tmp_path / "tb_mlp_token_pair_adapters.sv"
    testbench.write_text(
        """`timescale 1ns/1ps
module tb_mlp_token_pair_adapters;
  reg clk=0,rst_n=0,av=0,tv=0,rv=0,ov=0,od=0;
  reg [3:0] ag=0,tg=0,rg=0;reg [4:0] ak=0;reg [9:0] rt=0,ot=0;
  reg [1023:0] ad=0;reg [63:0] tf=0;reg [575:0] rd=0;
  reg [2:0] og=0;reg [1151:0] odat=0;
  wire avo,tvo,rvo,svo,sdo;wire [2:0] ago,tgo,rgo;
  wire [4:0] ako;wire [2047:0] ado;wire [127:0] tfo;
  wire [9:0] rto,sto;wire [1151:0] rdo;wire [3:0] sgo;wire [575:0] sdata;
  integer serial_seen=0;
  mlp_token_pair_input_adapter input_adapter(.clk(clk),.rst_n(rst_n),
    .activation_valid_in(av),.activation_group_in(ag),.activation_k_tile_in(ak),
    .activation_data_in(ad),.token_factor_valid_in(tv),
    .token_factor_group_in(tg),.token_factors_in(tf),
    .activation_valid_out(avo),.activation_group_out(ago),
    .activation_k_tile_out(ako),.activation_data_out(ado),
    .token_factor_valid_out(tvo),.token_factor_group_out(tgo),
    .token_factors_out(tfo));
  mlp_token_pair_residual_adapter residual_adapter(.clk(clk),.rst_n(rst_n),
    .valid_in(rv),.group_in(rg),.output_tile_in(rt),.data_in(rd),
    .valid_out(rvo),.group_out(rgo),.output_tile_out(rto),.data_out(rdo));
  mlp_token_pair_output_serializer serializer(.clk(clk),.rst_n(rst_n),
    .valid_in(ov),.done_in(od),.group_in(og),.output_tile_in(ot),.data_in(odat),
    .valid_out(svo),.done_out(sdo),.group_out(sgo),.output_tile_out(sto),
    .data_out(sdata));
  always #2 clk=~clk;
  always @(posedge clk) begin #1;
    if(avo && (ago!=2 || ako!=7 || ado!={1024'h22,1024'h11}))
      $fatal(1,"activation pair mismatch");
    if(tvo && (tgo!=2 || tfo!={64'h44,64'h33}))
      $fatal(1,"factor pair mismatch");
    if(rvo && (rgo!=2 || rto!=9 || rdo!={576'h66,576'h55}))
      $fatal(1,"residual pair mismatch");
    if(svo) begin
      if(serial_seen==0 && (sgo!=4 || sto!=9 || sdata!=576'h55 || sdo))
        $fatal(1,"serializer low mismatch");
      if(serial_seen==1 && (sgo!=5 || sto!=9 || sdata!=576'h66 || !sdo))
        $fatal(1,"serializer high mismatch");
      serial_seen=serial_seen+1;
    end
  end
  initial begin
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    av=1;ag=4;ak=7;ad=1024'h11;tv=1;tg=4;tf=64'h33;
    rv=1;rg=4;rt=9;rd=576'h55;
    @(negedge clk);ag=5;ad=1024'h22;tg=5;tf=64'h44;
    rg=5;rd=576'h66;
    @(negedge clk);av=0;tv=0;rv=0;
    wait(avo && tvo && rvo);@(negedge clk);
    ov=1;od=1;og=rgo;ot=rto;odat=rdo;
    @(negedge clk);ov=0;od=0;
    wait(serial_seen==2);repeat(3) @(posedge clk);
    $display("tb_mlp_token_pair_adapters: PASS");$finish;
  end
  initial begin repeat(100) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_mlp_token_pair_adapters"
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-Wall", "-s",
            "tb_mlp_token_pair_adapters", "-o", str(build),
            str(RTL / "mlp_token_pair_adapters.sv"), str(testbench),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        ["vvp", str(build)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "PASS" in run_result.stdout
