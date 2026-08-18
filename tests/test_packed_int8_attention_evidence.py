from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_retained_packed_qk_evidence_is_internally_consistent() -> None:
    evidence = json.loads(
        (
            ROOT
            / "data/results/h4-packed-int8-attention-logit-ablation.json"
        ).read_text(encoding="utf-8")
    )
    candidate = evidence["retained_twenty_sample_candidate"]
    assert candidate["sample_count"] == len(candidate["seeds"]) == 20
    assert candidate["masked_positions"] == candidate["sample_count"] * 32
    assert candidate["mean_top1_agreement"] == (
        candidate["top1_matches"] / candidate["masked_positions"]
    )
    gate = candidate["aggregate_gate"]
    assert candidate["mean_top1_agreement"] >= gate["required_top1_agreement"]
    assert gate["result"] == "pass"
    assert evidence["hardware_implication"]["retained_candidate"].startswith(
        "pack two signed INT8 QK products"
    )


def test_dynamic_qk_resource_evidence_matches_sources() -> None:
    evidence = json.loads(
        (
            ROOT / "data/results/h4-dynamic-qk-quantizer-yosys-xcup.json"
        ).read_text(encoding="utf-8")
    )
    for relative_path, expected_hash in evidence["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == (
            expected_hash
        )
    retained = evidence["retained_q17"]
    total = retained["separate_map_sum"]
    assert total["lut_primitives"] == (
        retained["scale_tracker"]["lut_primitives"]
        + retained["sixteen_lane_quantizer"]["lut_primitives"]
    )
    assert total["dsp48e2"] == 16
    assert evidence["q17_savings_vs_q24"]["dsp48e2"] == 16


def test_packed_dynamic_qk_scheduler_evidence_matches_sources() -> None:
    evidence = json.loads(
        (
            ROOT / "data/results/h4-packed-dynamic-qk-scheduler.json"
        ).read_text(encoding="utf-8")
    )
    for relative_path, expected_hash in evidence["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == (
            expected_hash
        )
    validation = evidence["real_block0_head0_validation"]
    assert validation["scores_compared"] == 64 * 64
    assert validation["representative_group_pair_cycles"] == 500
    assert validation["full_head_cycles"] == 4000
    full_head = evidence["full_head_cycle_comparison"]
    assert full_head["packed_measured_qk_cycles"] == (
        full_head["packed_group_pairs"]
        * validation["representative_group_pair_cycles"]
    )
    assert full_head["full_head_scores"] == validation["scores_compared"]
    resources = evidence["external_array_resource_comparison"]
    assert resources["increment"]["lut_primitives"] == (
        resources["packed_dynamic_m8_scheduler"]["lut_primitives"]
        - resources["current_fixed18_m4_scheduler"]["lut_primitives"]
    )
    connected = evidence["complete_head_connected_attention"]
    assert connected["qk_scores_compared"] == 64 * 64
    assert connected["softmax_probabilities_compared"] == 64 * 64
    assert connected["pv_attention_values_compared"] == 64 * 64
    assert connected["measured_cycles_saved_per_head"] == (
        connected["current_fixed18_head_attention_cycles"]
        - connected["packed_qk_softmax_and_fixed18_pv_cycles"]
    )
    assert connected["twelve_head_block_cycles_saved_if_integrated"] == (
        12 * connected["measured_cycles_saved_per_head"]
    )
    integrated = evidence["scratchpad_and_dynamic_scale_head_integration"]
    assert integrated["attention_values_compared"] == 64 * 64
    assert integrated["measured_cycles_saved"] == (
        integrated["current_fixed18_head_busy_cycles"]
        - integrated["packed_head_busy_cycles"]
    )
    qkv = evidence["connected_qkv_rotary_attention_head"]
    assert qkv["intermediate_qkv_values_checked"] == 64 * 3 * 64
    assert qkv["attention_values_compared"] == 64 * 64
    assert qkv["measured_cycles_saved"] == (
        qkv["current_fixed18_connected_cycles"]
        - qkv["packed_connected_cycles"]
    )
    assert qkv["single_external_mixed_mode_array"] is True
    integration = evidence["multihead_and_block_integration"]
    canvas = integration["two_head_canvas_control_model"]
    assert canvas["narrow_requests"] > 0
    assert canvas["wide_requests"] > 0
    subpipeline = integration["two_head_two_output_tile_subpipeline"]
    assert subpipeline["cycles_saved"] == (
        subpipeline["current_fixed18_cycles"] - subpipeline["packed_cycles"]
    )
    ddit = integration["reduced_complete_ddit_block"]
    assert ddit["cycles_saved"] == (
        ddit["current_fixed18_cycles"] - ddit["packed_cycles"]
    )
    assert integration["board_facing_default"]["packed_attention_default"]
