from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.fixed_attention import fixed_attention_q12, fixed_rotary_q12
from diffusion_accel.fixed_mlp import _load_tensors


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"


def _packed_hex(values: list[int], width: int) -> str:
    mask = (1 << width) - 1
    packed = sum(
        (value & mask) << (index * width)
        for index, value in enumerate(values)
    )
    return f"{packed:0{(len(values) * width + 3) // 4}x}"


def test_attention_head_pipeline_matches_all_h0_outputs(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    if not golden_path.is_file() or not weights_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    name = "folded.block_00.qkv"
    qkv = _load_tensors(golden_path, [name])[name][0]
    tables = _load_tensors(weights_path, ["rotary.cos", "rotary.sin"])
    cosine = tables["rotary.cos"].float()
    sine = tables["rotary.sin"].float()
    _, attention_q, _ = fixed_attention_q12(qkv, cosine, sine)
    _, _, query_q, key_q, rotary_details = fixed_rotary_q12(
        qkv, cosine, sine
    )
    value_q = rotary_details["tensors"]["qkv_q12"][:, 2]
    attention_head = attention_q.view(64, 12, 64)[:, 0]

    qkv_hex = tmp_path / "full_head0_qkv.hex"
    qkv_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(query_q[token, 0, channel]),
                    int(key_q[token, 0, channel]),
                    int(value_q[token, 0, channel]),
                ],
                18,
            )
            for token in range(64)
            for channel in range(64)
        )
        + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "full_head0_expected.hex"
    expected_hex.write_text(
        "\n".join(
            f"{int(attention_head[token, channel]) & 0x3FFFF:05x}"
            for token in range(64)
            for channel in range(64)
        )
        + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_attention_head_pipeline_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_attention_head_pipeline_h0;
  reg clk=0,rst_n=0,load_valid=0,start=0,attention_tile_ready=1;
  reg [5:0] load_token=0,load_channel=0;
  reg signed [17:0] load_q=0,load_k=0,load_v=0;
  wire start_ready,attention_tile_valid,busy,done;
  wire [3:0] attention_group,attention_output_tile;
  wire [2:0] attention_valid_channels;
  wire [431:0] attention_packed;
  reg [53:0] qkv_mem [0:4095];
  reg [17:0] expected_mem [0:4095];
  integer index,tile_count=0,query,channel_lane,channel,token,cycle_count=0;
  attention_head_pipeline dut(
    .clk(clk),.rst_n(rst_n),.load_valid(load_valid),.load_token(load_token),
    .load_channel(load_channel),.load_query_q12(load_q),.load_key_q12(load_k),
    .load_value_q12(load_v),.start(start),.start_ready(start_ready),
    .attention_tile_valid(attention_tile_valid),
    .attention_tile_ready(attention_tile_ready),.attention_group(attention_group),
    .attention_output_tile(attention_output_tile),
    .attention_valid_channels(attention_valid_channels),
    .attention_q12_packed(attention_packed),.busy(busy),.done(done));
  always #2 clk=~clk;
  always @(posedge clk) begin
    if(busy) cycle_count=cycle_count+1;
    #1;
    if(attention_tile_valid) begin
      if(attention_group!==(tile_count/11) ||
         attention_output_tile!==(tile_count%11))
        $fatal(1,"full-head tile tag mismatch at %0d",tile_count);
      if(attention_valid_channels!==(((tile_count%11)==10)?4:6))
        $fatal(1,"full-head valid channel mismatch");
      for(query=0;query<4;query=query+1)
        for(channel_lane=0;channel_lane<attention_valid_channels;
            channel_lane=channel_lane+1) begin
          token=attention_group*4+query;
          channel=attention_output_tile*6+channel_lane;
          if($signed(attention_packed[(query*6+channel_lane)*18 +: 18])!==
             $signed(expected_mem[token*64+channel]))
            $fatal(1,"full-head mismatch token %0d channel %0d",token,channel);
        end
      tile_count=tile_count+1;
    end
  end
  initial begin
    $readmemh("{qkv_hex}",qkv_mem);$readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(index=0;index<4096;index=index+1) begin
      @(negedge clk);load_valid=1;load_token=index/64;load_channel=index%64;
      load_q=qkv_mem[index][17:0];load_k=qkv_mem[index][35:18];
      load_v=qkv_mem[index][53:36];
    end
    @(negedge clk);load_valid=0;start=1;@(negedge clk);start=0;
    wait(done);repeat(2) @(posedge clk);
    if(tile_count!=176) $fatal(1,"missing full-head attention tiles");
    if(busy) $fatal(1,"attention head remained busy");
    $display("tb_attention_head_pipeline_h0: PASS cycles=%0d",cycle_count);
    $finish;
  end
  initial begin repeat(30000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_attention_head_pipeline_h0"
    sources = [
        "attention_head_scratchpad_banked.sv",
        "int8_mac_tile_pipelined.sv",
        "mixed_precision_mac_tile_pipelined.sv",
        "attention_qk_group_scheduler.sv",
        "unsigned_divider_iterative.sv",
        "exp_neg_q16_lut_bram.sv",
        "attention_softmax_row_q16.sv",
        "attention_score_group_softmax_stream.sv",
        "attention_pv_group_scheduler.sv",
        "attention_group_pipeline.sv",
        "attention_head_pipeline.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-s", "tb_attention_head_pipeline_h0",
            "-o", str(build), *(str(RTL / item) for item in sources),
            str(testbench),
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
    assert "tb_attention_head_pipeline_h0: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) == 15056
