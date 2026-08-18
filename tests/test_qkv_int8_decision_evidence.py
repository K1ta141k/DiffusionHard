import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data/results/h4-qkv-int8-decision.json"


def test_qkv_int8_decision_is_source_backed_and_rejected() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    for relative_path, expected_hash in evidence["source_sha256"].items():
        actual_hash = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash

    plain = evidence["plain_int8_h0"]
    smooth = evidence["smoothquant_int8_h0"]
    logits = evidence["smoothquant_int8_five_canvas_logits"]
    gate = evidence["acceptance_gate"]
    assert smooth["mean_qkv_relative_rms_vs_h0"] < (
        plain["mean_qkv_relative_rms_vs_h0"] / 6
    )
    assert smooth["maximum_smoothed_activation_saturation_fraction"] == 0
    assert logits["top1_matches"] == 156
    assert logits["masked_positions"] == 160
    assert logits["mean_top1_agreement"] == 156 / 160
    assert logits["mean_top1_agreement"] < gate[
        "minimum_mean_top1_agreement"
    ]
    assert gate["passed"] is False
    assert evidence["decision"].startswith("retain INT16 QKV")
