from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from diffusion_accel.fixed_attention import fixed_rotary_q12
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


def test_rotary_head_writeback_matches_all_h0_head0_values(
    tmp_path: Path,
) -> None:
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
    _, _, query_rotated_q12, key_rotated_q12, details = fixed_rotary_q12(
        qkv, tables["rotary.cos"].float(), tables["rotary.sin"].float()
    )
    unrotated = details["tensors"]["qkv_q12"]
    cosine_q15 = details["tensors"]["cosine_q15"]
    sine_q15 = details["tensors"]["sine_q15"]

    unrotated_hex = tmp_path / "unrotated_head0_qk.hex"
    unrotated_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(unrotated[token, 0, 0, channel]),
                    int(unrotated[token, 1, 0, channel]),
                ],
                18,
            )
            for token in range(64)
            for channel in range(64)
        )
        + "\n",
        encoding="utf-8",
    )
    constants_hex = tmp_path / "rotary_constants.hex"
    constants_hex.write_text(
        "\n".join(
            _packed_hex(
                [int(cosine_q15[token, pair]), int(sine_q15[token, pair])],
                16,
            )
            for token in range(64)
            for pair in range(32)
        )
        + "\n",
        encoding="utf-8",
    )
    expected_hex = tmp_path / "rotated_head0_qk.hex"
    expected_hex.write_text(
        "\n".join(
            _packed_hex(
                [
                    int(query_rotated_q12[token, 0, channel]),
                    int(key_rotated_q12[token, 0, channel]),
                ],
                18,
            )
            for token in range(64)
            for channel in range(64)
        )
        + "\n",
        encoding="utf-8",
    )

    testbench = tmp_path / "tb_rotary_head_writeback_h0.sv"
    testbench.write_text(
        f"""`timescale 1ns/1ps
module tb_rotary_head_writeback_h0;
  reg clk=0,rst_n=0,start=0,q_load=0,k_load=0,constant_load=0;
  reg [5:0] load_token=0,load_channel=0,constant_token=0;
  reg [4:0] constant_pair=0;
  reg signed [17:0] load_q=0,load_k=0;
  reg signed [15:0] load_cos=0,load_sin=0;
  wire start_ready,qk_read_valid,qk_data_valid,constant_read_valid;
  wire constant_data_valid,query_write,key_write,busy,done;
  wire [5:0] qk_read_token,qk_token_out,constant_read_token;
  wire [5:0] constant_token_out,write_token,write_channel;
  wire [4:0] qk_read_pair,qk_pair_out,constant_read_pair,constant_pair_out;
  wire signed [17:0] q_first,q_second,k_first,k_second,write_q,write_k;
  wire signed [15:0] cosine,sine;
  reg [35:0] unrotated_mem [0:4095];
  reg [31:0] constant_mem [0:2047];
  reg [35:0] expected_mem [0:4095];
  integer index,write_count=0,cycle_count=0;

  qk_unrotated_scratchpad_banked scratch(
    .clk(clk),.query_load_valid(q_load),.key_load_valid(k_load),
    .load_token(load_token),.load_channel(load_channel),
    .load_query_q12(load_q),.load_key_q12(load_k),
    .read_valid(qk_read_valid),.read_token(qk_read_token),
    .read_pair(qk_read_pair),.read_data_valid(qk_data_valid),
    .read_token_out(qk_token_out),.read_pair_out(qk_pair_out),
    .query_first_q12(q_first),.query_second_q12(q_second),
    .key_first_q12(k_first),.key_second_q12(k_second));
  rotary_constant_table_bram constants(
    .clk(clk),.load_valid(constant_load),.load_token(constant_token),
    .load_pair(constant_pair),.load_cosine_q15(load_cos),.load_sine_q15(load_sin),
    .read_valid(constant_read_valid),.read_token(constant_read_token),
    .read_pair(constant_read_pair),.read_data_valid(constant_data_valid),
    .read_token_out(constant_token_out),.read_pair_out(constant_pair_out),
    .cosine_q15(cosine),.sine_q15(sine));
  rotary_head_writeback_scheduler dut(
    .clk(clk),.rst_n(rst_n),.start(start),.start_ready(start_ready),.head_in(0),
    .qk_read_valid(qk_read_valid),.qk_read_token(qk_read_token),
    .qk_read_pair(qk_read_pair),.qk_read_data_valid(qk_data_valid),
    .qk_read_token_out(qk_token_out),.qk_read_pair_out(qk_pair_out),
    .query_first_q12(q_first),.query_second_q12(q_second),
    .key_first_q12(k_first),.key_second_q12(k_second),
    .constant_read_valid(constant_read_valid),
    .constant_read_token(constant_read_token),
    .constant_read_pair(constant_read_pair),
    .constant_read_data_valid(constant_data_valid),
    .constant_read_token_out(constant_token_out),
    .constant_read_pair_out(constant_pair_out),.cosine_q15(cosine),.sine_q15(sine),
    .query_write_valid(query_write),.key_write_valid(key_write),
    .write_token(write_token),.write_channel(write_channel),
    .write_query_q12(write_q),.write_key_q12(write_k),.busy(busy),.done(done));

  always #2 clk=~clk;
  always @(posedge clk) begin
    if(busy) cycle_count=cycle_count+1;
    #1;
    if(query_write || key_write) begin
      if(!query_write || !key_write) $fatal(1,"Q and K writeback split");
      if($signed(write_q)!==$signed(expected_mem[write_token*64+write_channel][17:0]) ||
         $signed(write_k)!==$signed(expected_mem[write_token*64+write_channel][35:18]))
        $fatal(1,"rotary writeback mismatch token %0d channel %0d",write_token,write_channel);
      write_count=write_count+1;
    end
  end

  initial begin
    $readmemh("{unrotated_hex}",unrotated_mem);
    $readmemh("{constants_hex}",constant_mem);$readmemh("{expected_hex}",expected_mem);
    repeat(3) @(posedge clk);@(negedge clk);rst_n=1;
    for(index=0;index<4096;index=index+1) begin
      @(negedge clk);q_load=1;k_load=1;load_token=index/64;load_channel=index%64;
      load_q=unrotated_mem[index][17:0];load_k=unrotated_mem[index][35:18];
    end
    @(negedge clk);q_load=0;k_load=0;
    for(index=0;index<2048;index=index+1) begin
      @(negedge clk);constant_load=1;constant_token=index/32;constant_pair=index%32;
      load_cos=constant_mem[index][15:0];load_sin=constant_mem[index][31:16];
    end
    @(negedge clk);constant_load=0;start=1;@(negedge clk);start=0;
    wait(done);repeat(2) @(posedge clk);
    if(write_count!=4096) $fatal(1,"missing rotary writebacks");
    if(busy) $fatal(1,"rotary scheduler remained busy");
    $display("tb_rotary_head_writeback_h0: PASS cycles=%0d",cycle_count);
    $finish;
  end
  initial begin repeat(20000) @(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_rotary_head_writeback_h0"
    sources = [
        "qk_unrotated_scratchpad_banked.sv",
        "rotary_constant_table_bram.sv",
        "rotary_qk_pair_serial.sv",
        "rotary_head_writeback_scheduler.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog", "-g2012", "-s", "tb_rotary_head_writeback_h0",
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
    assert "tb_rotary_head_writeback_h0: PASS" in run_result.stdout
    match = re.search(r"cycles=(\d+)", run_result.stdout)
    assert match is not None
    assert int(match.group(1)) == 4099
