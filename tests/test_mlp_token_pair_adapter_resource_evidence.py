from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pairing_buffer_resource_evidence_matches_rtl() -> None:
    rtl_path = Path("rtl/tensor_engine/mlp_token_pair_adapters.sv")
    evidence = json.loads(
        (ROOT / "data/results/mlp-token-pair-input-adapter-yosys-xcup.json")
        .read_text(encoding="utf-8")
    )
    digest = hashlib.sha256()
    digest.update(str(rtl_path).encode("utf-8"))
    digest.update(b"\0")
    digest.update((ROOT / rtl_path).read_bytes())
    digest.update(b"\0")

    assert evidence["rtl_sha256"] == digest.hexdigest()
    assert evidence["primitive_cells"]["RAM32M16"] == 64
    assert evidence["primitive_cells"]["LUT2"] == 5
    assert evidence["primitive_cells"]["LUT3"] == 1
    assert evidence["timing_validated"] is False

    integrated_evidence = json.loads(
        (ROOT / "data/results/h4-packed-dynamic-qk-scheduler.json").read_text(
            encoding="utf-8"
        )
    )["mlp_pairing_buffer_resource_optimization"]
    assert integrated_evidence["after_total_primitive_cells"] == evidence[
        "total_primitive_cells"
    ]
    assert integrated_evidence["after_logic_lut_primitives"] == sum(
        count
        for primitive, count in evidence["primitive_cells"].items()
        if primitive.startswith("LUT")
    )
    assert integrated_evidence["after_flip_flop_primitives"] == evidence[
        "primitive_cells"
    ]["FDRE"]
