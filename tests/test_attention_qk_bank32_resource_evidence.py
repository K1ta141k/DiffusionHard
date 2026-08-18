import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "data/results/h4-qk-bank32-resource-decision.json"
CURRENT_MAP = ROOT / (
    "data/results/attention-qk-pair-softmax-packed-external-banked-yosys-xcup.json"
)


def test_qk_bank32_resource_decision_is_live_and_consistent() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    current_map = json.loads(CURRENT_MAP.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for relative_path in current_map["rtl_paths"]:
        source = ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    assert digest.hexdigest() == current_map["rtl_sha256"]

    baseline = decision["baseline_flat64_buffer"]
    retained = decision["retained_bank32_buffer"]
    reduction = decision["reduction"]
    usage = current_map["device_capacity_comparison"]["mapped_usage"]
    assert current_map["parameters"] == {"INTERNAL_MAC": 0}
    assert usage["clb_lut_equivalent_primitives"] == 24_845
    assert usage["flip_flop_primitives"] == 16_351
    assert retained["clb_lut_equivalent_primitives"] == usage[
        "clb_lut_equivalent_primitives"
    ]
    assert reduction["clb_lut_equivalent_primitives"] == (
        baseline["clb_lut_equivalent_primitives"]
        - retained["clb_lut_equivalent_primitives"]
    )
    assert reduction["cycles"] == (
        baseline["cycles_for_all_group_pairs"]
        - retained["cycles_for_all_group_pairs"]
    )
    assert retained["h0_result"] == "bit exact PASS"
