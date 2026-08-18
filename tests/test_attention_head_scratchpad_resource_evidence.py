from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_combined_query_key_scratchpad_resource_evidence_matches_rtl() -> None:
    rtl_path = Path(
        "rtl/tensor_engine/attention_head_scratchpad_qk_combined_banked.sv"
    )
    evidence = json.loads(
        (
            ROOT
            / "data/results/attention-head-scratchpad-qk-combined-yosys-xcup.json"
        ).read_text(encoding="utf-8")
    )
    digest = hashlib.sha256()
    digest.update(str(rtl_path).encode("utf-8"))
    digest.update(b"\0")
    digest.update((ROOT / rtl_path).read_bytes())
    digest.update(b"\0")

    assert evidence["rtl_sha256"] == digest.hexdigest()
    assert evidence["primitive_cells"]["RAMB18E2"] == 32
    assert evidence["device_capacity_comparison"]["mapped_usage"][
        "bram36_equivalent_blocks"
    ] == 16

    integrated = json.loads(
        (ROOT / "data/results/h4-packed-dynamic-qk-scheduler.json").read_text(
            encoding="utf-8"
        )
    )["attention_scratchpad_bram_optimization"]
    assert integrated["after_ramb18e2"] == evidence["primitive_cells"][
        "RAMB18E2"
    ]
    assert integrated["ramb18e2_saved"] == (
        integrated["before_ramb18e2"] - integrated["after_ramb18e2"]
    )
