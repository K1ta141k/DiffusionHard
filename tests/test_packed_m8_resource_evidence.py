from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dsp_cascade_resource_evidence_matches_retained_rtl() -> None:
    evidence_path = (
        ROOT
        / "data/results/mixed-precision-packed-m8-dsp-cascade-yosys-xcup.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256()
    for relative_path in evidence["rtl_paths"]:
        source_path = ROOT / relative_path
        source_hash.update(relative_path.encode("utf-8"))
        source_hash.update(b"\0")
        source_hash.update(source_path.read_bytes())
        source_hash.update(b"\0")
    assert source_hash.hexdigest() == evidence["rtl_sha256"]

    primitives = evidence["primitive_cells"]
    mapped = evidence["device_capacity_comparison"]["mapped_usage"]
    lut_count = sum(
        count for name, count in primitives.items() if name.startswith("LUT")
    )
    assert lut_count == mapped["lut_primitives"] == 54_082
    assert primitives["FDRE"] == mapped["flip_flop_primitives"] == 40_878
    assert primitives["DSP48E2"] == mapped["dsp_primitives"] == 768
    assert sum(primitives.values()) == evidence["total_primitive_cells"]
