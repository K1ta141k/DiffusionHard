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


def test_multihead_canvas_matches_complete_h0_head0(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")
    golden_path = PACKAGE / "golden_tensors.safetensors"
    weights_path = PACKAGE / "folded_fp16_weights.safetensors"
    if not golden_path.is_file() or not weights_path.is_file():
        pytest.skip("H0 hardware package is unavailable")

    qkv = _load_tensors(golden_path, ["folded.block_00.qkv"])[
        "folded.block_00.qkv"
    ][0]
    tables = _load_tensors(weights_path, ["rotary.cos", "rotary.sin"])
    cosine = tables["rotary.cos"].float()
    sine = tables["rotary.sin"].float()
    _, attention_q, _ = fixed_attention_q12(qkv, cosine, sine)
    _, _, query_q, key_q, rotary_details = fixed_rotary_q12(
        qkv, cosine, sine
    )
    value_q = rotary_details["tensors"]["qkv_q12"][:, 2]
    attention_head = attention_q.view(64, 12, 64)[:, 0]

    qkv_hex = tmp_path / "multihead_h0_qkv.hex"
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
    expected_hex = tmp_path / "multihead_h0_expected.hex"
    expected_hex.write_text(
        "\n".join(
            f"{int(attention_head[token, channel]) & 0x3FFFF:05x}"
            for token in range(64)
            for channel in range(64)
        )
        + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_attention_multihead_canvas_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_attention_multihead_canvas_h0;
  reg clk=0,rst_n=0,block_start=0,load_valid=0,canvas_read_valid=0;
  reg [3:0] load_head=0,canvas_read_head=0;
  reg [5:0] load_token=0,load_channel=0,canvas_read_token=0;
  reg signed [17:0] load_q=0,load_k=0,load_v=0;
  wire block_start_ready,load_ready,canvas_read_data_valid,busy,done;
  wire [3:0] expected_head;
  wire [1151:0] canvas_read_data;
  reg [53:0] qkv_mem [0:4095];
  reg [17:0] expected_mem [0:4095];
  integer index,token,channel,cycle_count=0;

  attention_multihead_canvas_pipeline #(.HEADS(1)) dut(
    .clk(clk),.rst_n(rst_n),.block_start(block_start),
    .block_start_ready(block_start_ready),.load_valid(load_valid),
    .load_ready(load_ready),.load_head(load_head),.load_token(load_token),
    .load_channel(load_channel),.load_query_q12(load_q),.load_key_q12(load_k),
    .load_value_q12(load_v),.expected_head(expected_head),
    .canvas_read_valid(canvas_read_valid),.canvas_read_head(canvas_read_head),
    .canvas_read_token(canvas_read_token),
    .canvas_read_data_valid(canvas_read_data_valid),
    .canvas_read_data_packed(canvas_read_data),.busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) if(busy) cycle_count=cycle_count+1;

  initial begin
    $readmemh("{qkv_hex}",qkv_mem);$readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    block_start=1;@(negedge clk);block_start=0;
    wait(load_ready);
    for(index=0;index<4096;index=index+1) begin
      @(negedge clk);load_valid=1;load_token=index/64;load_channel=index%64;
      load_q=qkv_mem[index][17:0];load_k=qkv_mem[index][35:18];
      load_v=qkv_mem[index][53:36];
    end
    @(negedge clk);load_valid=0;
    wait(done);@(negedge clk);
    for(token=0;token<64;token=token+1) begin
      canvas_read_valid=1;canvas_read_token=token;
      @(posedge clk);#1;
      if(!canvas_read_data_valid) $fatal(1,"canvas read valid missing");
      for(channel=0;channel<64;channel=channel+1)
        if($signed(canvas_read_data[channel*18 +: 18])!==
           $signed(expected_mem[token*64+channel]))
          $fatal(1,"canvas mismatch token %0d channel %0d",token,channel);
      @(negedge clk);
    end
    canvas_read_valid=0;
    if(busy || !block_start_ready) $fatal(1,"multihead pipeline remained busy");
    $display("tb_attention_multihead_canvas_h0: PASS cycles=%0d",cycle_count);
    $finish;
  end
  initial begin repeat(40000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_attention_multihead_canvas_h0"
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
        "attention_multihead_controller.sv",
        "attention_canvas_scratchpad_banked.sv",
        "attention_multihead_canvas_pipeline.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-s", "tb_attention_multihead_canvas_h0",
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
    assert "tb_attention_multihead_canvas_h0: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) == 19158
