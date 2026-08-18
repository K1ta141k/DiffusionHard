from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rejected_pv_reduction_summary_matches_screen_artifacts() -> None:
    ablation = json.loads(
        (ROOT / "data/results/h4-packed-int8-attention-logit-ablation.json")
        .read_text(encoding="utf-8")
    )["pv_reduction_followup"]
    artifacts = {
        "unsigned8_probability_signed8_value_five_seed": (
            "h4-packed-uint8-pv-logits-5seed.json"
        ),
        "sum255_corrected_unsigned8_probability_signed8_value_five_seed": (
            "h4-packed-uint8-pv-sum255-logits-5seed.json"
        ),
        "unsigned8_probability_signed9_value_five_seed": (
            "h4-packed-uint8-int9-pv-logits-5seed.json"
        ),
        "unsigned8_probability_signed9_value_twenty_seed": (
            "h4-packed-uint8-int9-pv-logits-20seed.json"
        ),
    }
    for summary_name, artifact_name in artifacts.items():
        measured = json.loads(
            (ROOT / "data/results" / artifact_name).read_text(encoding="utf-8")
        )["summary"]
        retained = ablation[summary_name]
        assert retained["top1_matches"] == measured["top1_matches"]
        assert retained["masked_positions"] == measured["masked_positions"]
        assert retained["mean_top1_agreement"] == measured[
            "mean_top1_agreement"
        ]
