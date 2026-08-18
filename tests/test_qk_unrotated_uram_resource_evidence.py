from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paired_qk_uram_resource_evidence_matches_rtl() -> None:
    rtl_path = Path("rtl/tensor_engine/qk_unrotated_scratchpad_paired_uram.sv")
    evidence = json.loads(
        (
            ROOT
            / "data/results/qk-unrotated-scratchpad-paired-uram-yosys-xcup.json"
        ).read_text(encoding="utf-8")
    )
    digest = hashlib.sha256()
    digest.update(str(rtl_path).encode("utf-8"))
    digest.update(b"\0")
    digest.update((ROOT / rtl_path).read_bytes())
    digest.update(b"\0")

    assert evidence["rtl_sha256"] == digest.hexdigest()
    assert evidence["primitive_cells"]["URAM288"] == 4
    assert evidence["device_capacity_comparison"]["mapped_usage"][
        "clb_lut_equivalent_primitives"
    ] == 2

    integrated = json.loads(
        (ROOT / "data/results/h4-packed-dynamic-qk-scheduler.json").read_text(
            encoding="utf-8"
        )
    )["pre_rotary_qk_uram_optimization"]
    assert integrated["after_uram288"] == evidence["primitive_cells"]["URAM288"]
    assert integrated["clb_lut_equivalent_primitives_saved"] == (
        integrated["before_clb_lut_equivalent_primitives"]
        - integrated["after_logic_lut_primitives"]
    )
