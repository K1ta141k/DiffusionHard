import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl/tensor_engine"


def test_folded_n2_matches_direct_n6_for_all_modes(tmp_path: Path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required")

    testbench = tmp_path / "tb_ddit_shared_mac_folded_n2.sv"
    testbench.write_text(
        """`timescale 1ns/1ps
module tb_ddit_shared_mac_folded_n2;
  reg clk=0,rst_n=0,mlp_phase=0;
  reg attention_valid=0,attention_narrow=0,attention_clear=0;
  reg attention_last=0;
  reg [7:0] attention_tag=0;
  reg [2303:0] attention_activations=0;
  reg [3455:0] attention_weights=0;
  reg [2047:0] attention_narrow_activations=0;
  reg [1535:0] attention_narrow_weights=0;
  reg mlp_valid=0,mlp_clear=0,mlp_last=0;
  reg [15:0] mlp_tag=0;
  reg [2047:0] mlp_activations=0;
  reg [1535:0] mlp_weights=0;

  wire direct_attention_valid,direct_attention_narrow;
  wire folded_attention_valid,folded_attention_narrow;
  wire [7:0] direct_attention_tag,folded_attention_tag;
  wire [1151:0] direct_attention_accumulators;
  wire [1151:0] folded_attention_accumulators;
  wire [1535:0] direct_attention_narrow_accumulators;
  wire [1535:0] folded_attention_narrow_accumulators;
  wire direct_mlp_valid,folded_mlp_valid;
  wire [15:0] direct_mlp_tag,folded_mlp_tag;
  wire [1535:0] direct_mlp_accumulators,folded_mlp_accumulators;

  reg direct_attention_seen=0,folded_attention_seen=0;
  reg direct_mlp_seen=0,folded_mlp_seen=0;
  reg expected_narrow=0;
  reg [1151:0] saved_attention_accumulators;
  reg [1535:0] saved_attention_narrow_accumulators;
  reg [1535:0] saved_mlp_accumulators;
  integer index,k_tile;

  ddit_block_shared_mac_packed_m8 #(
    .MLP_TAG_WIDTH(16),.ATTENTION_PACKED(1),.PHYSICAL_N_LANES(6)
  ) direct(
    .clk(clk),.rst_n(rst_n),.mlp_phase(mlp_phase),
    .attention_request_valid(attention_valid),
    .attention_request_narrow_int8_mode(attention_narrow),
    .attention_request_clear(attention_clear),
    .attention_request_last(attention_last),
    .attention_request_tag(attention_tag),
    .attention_request_activations(attention_activations),
    .attention_request_weights(attention_weights),
    .attention_request_narrow_activations(attention_narrow_activations),
    .attention_request_narrow_weights(attention_narrow_weights),
    .attention_response_valid(direct_attention_valid),
    .attention_response_narrow_int8_mode(direct_attention_narrow),
    .attention_response_tag(direct_attention_tag),
    .attention_response_accumulators(direct_attention_accumulators),
    .attention_response_narrow_accumulators(
      direct_attention_narrow_accumulators),
    .mlp_request_valid(mlp_valid),.mlp_request_clear(mlp_clear),
    .mlp_request_last(mlp_last),.mlp_request_tag(mlp_tag),
    .mlp_request_activations(mlp_activations),
    .mlp_request_weights(mlp_weights),.mlp_response_valid(direct_mlp_valid),
    .mlp_response_tag(direct_mlp_tag),
    .mlp_response_accumulators(direct_mlp_accumulators));

  ddit_block_shared_mac_packed_m8 #(
    .MLP_TAG_WIDTH(16),.ATTENTION_PACKED(1),.PHYSICAL_N_LANES(2)
  ) folded(
    .clk(clk),.rst_n(rst_n),.mlp_phase(mlp_phase),
    .attention_request_valid(attention_valid),
    .attention_request_narrow_int8_mode(attention_narrow),
    .attention_request_clear(attention_clear),
    .attention_request_last(attention_last),
    .attention_request_tag(attention_tag),
    .attention_request_activations(attention_activations),
    .attention_request_weights(attention_weights),
    .attention_request_narrow_activations(attention_narrow_activations),
    .attention_request_narrow_weights(attention_narrow_weights),
    .attention_response_valid(folded_attention_valid),
    .attention_response_narrow_int8_mode(folded_attention_narrow),
    .attention_response_tag(folded_attention_tag),
    .attention_response_accumulators(folded_attention_accumulators),
    .attention_response_narrow_accumulators(
      folded_attention_narrow_accumulators),
    .mlp_request_valid(mlp_valid),.mlp_request_clear(mlp_clear),
    .mlp_request_last(mlp_last),.mlp_request_tag(mlp_tag),
    .mlp_request_activations(mlp_activations),
    .mlp_request_weights(mlp_weights),.mlp_response_valid(folded_mlp_valid),
    .mlp_response_tag(folded_mlp_tag),
    .mlp_response_accumulators(folded_mlp_accumulators));

  always #2 clk=~clk;
  always @(posedge clk) begin
    if(direct_attention_valid) begin
      direct_attention_seen<=1;
      saved_attention_accumulators<=direct_attention_accumulators;
      saved_attention_narrow_accumulators<=
        direct_attention_narrow_accumulators;
    end
    if(folded_attention_valid) begin
      if(!direct_attention_seen)$fatal(1,"folded attention returned first");
      if(folded_attention_tag!==direct_attention_tag ||
         folded_attention_narrow!==expected_narrow)
        $fatal(1,"folded attention metadata mismatch");
      if(expected_narrow) begin
        if(folded_attention_narrow_accumulators!==
           saved_attention_narrow_accumulators)
          $fatal(1,"folded packed attention mismatch");
      end else if(folded_attention_accumulators!==
                  saved_attention_accumulators)
        $fatal(1,"folded fixed18 attention mismatch");
      folded_attention_seen<=1;
    end
    if(direct_mlp_valid) begin
      direct_mlp_seen<=1;
      saved_mlp_accumulators<=direct_mlp_accumulators;
    end
    if(folded_mlp_valid) begin
      if(!direct_mlp_seen)$fatal(1,"folded MLP returned first");
      if(folded_mlp_tag!==direct_mlp_tag ||
         folded_mlp_accumulators!==saved_mlp_accumulators)
        $fatal(1,"folded MLP mismatch");
      folded_mlp_seen<=1;
    end
  end

  task run_attention;
    input narrow_mode;
    begin
      mlp_phase=0;expected_narrow=narrow_mode;
      direct_attention_seen=0;folded_attention_seen=0;
      for(k_tile=0;k_tile<3;k_tile=k_tile+1) begin
        @(negedge clk);
        for(index=0;index<128;index=index+1)
          attention_activations[index*18+:18]=(index+3*k_tile)%31-15;
        for(index=0;index<192;index=index+1)
          attention_weights[index*18+:18]=(2*index+5*k_tile)%29-14;
        for(index=0;index<256;index=index+1)
          attention_narrow_activations[index*8+:8]=(index+7*k_tile)%17-8;
        for(index=0;index<192;index=index+1)
          attention_narrow_weights[index*8+:8]=(3*index+k_tile)%15-7;
        attention_valid=1;attention_narrow=narrow_mode;
        attention_clear=(k_tile==0);attention_last=(k_tile==2);
        attention_tag=narrow_mode?8'h52:8'h31;
      end
      @(negedge clk);attention_valid=0;attention_clear=0;attention_last=0;
      wait(folded_attention_seen);@(negedge clk);
    end
  endtask

  task run_mlp;
    begin
      mlp_phase=1;direct_mlp_seen=0;folded_mlp_seen=0;
      for(k_tile=0;k_tile<4;k_tile=k_tile+1) begin
        @(negedge clk);
        for(index=0;index<256;index=index+1)
          mlp_activations[index*8+:8]=(index+2*k_tile)%19-9;
        for(index=0;index<192;index=index+1)
          mlp_weights[index*8+:8]=(5*index+3*k_tile)%21-10;
        mlp_valid=1;mlp_clear=(k_tile==0);mlp_last=(k_tile==3);
        mlp_tag=16'hc35a;
      end
      @(negedge clk);mlp_valid=0;mlp_clear=0;mlp_last=0;
      wait(folded_mlp_seen);@(negedge clk);
    end
  endtask

  initial begin
    repeat(3)@(posedge clk);@(negedge clk);rst_n=1;
    run_attention(0);
    run_attention(1);
    run_mlp();
    $display("tb_ddit_shared_mac_folded_n2: PASS modes=3");
    $finish;
  end
  initial begin repeat(500)@(posedge clk);$fatal(1,"timeout");end
endmodule
""",
        encoding="utf-8",
    )
    build = tmp_path / "tb_ddit_shared_mac_folded_n2"
    sources = [
        "mixed_precision_dsp48e2_cascade_cell.sv",
        "mixed_precision_token_pair_multiplier.sv",
        "mixed_precision_packed_m8_mac_tile_pipelined.sv",
        "ddit_block_shared_mac_packed_m8.sv",
    ]
    compile_result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_ddit_shared_mac_folded_n2",
            "-o",
            str(build),
            *(str(RTL / source) for source in sources),
            str(testbench),
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
        timeout=30,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    assert "tb_ddit_shared_mac_folded_n2: PASS modes=3" in run_result.stdout
