import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/results/hidden-canvas-mlp-wide-shared-yosys-xcup.json"
M8 = ROOT / "data/results/hidden-canvas-mlp-wide-current-focused-yosys-xcup.json"
M4 = ROOT / "data/results/hidden-canvas-mlp-m4-sync-current-focused-yosys-xcup.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_live(evidence: dict) -> None:
    digest = hashlib.sha256()
    for relative_path in evidence["rtl_paths"]:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative_path).read_bytes())
        digest.update(b"\0")
    assert digest.hexdigest() == evidence["rtl_sha256"]


def test_sync_activation_and_two_residual_banks_reduce_m8_mlp() -> None:
    baseline = _load(OLD)
    current = _load(M8)
    _assert_live(current)
    old_usage = baseline["device_capacity_comparison"]["mapped_usage"]
    usage = current["device_capacity_comparison"]["mapped_usage"]
    assert old_usage["clb_lut_equivalent_primitives"] == 143_251
    assert usage["clb_lut_equivalent_primitives"] == 80_941
    assert usage["clb_lut_equivalent_primitives"] < (
        old_usage["clb_lut_equivalent_primitives"] * 0.57
    )
    assert usage["distributed_ram_lut_equivalents"] == 10_632
    assert usage["uram_primitives"] == 29


def test_m4_sync_memory_is_explicit_lower_area_design_point() -> None:
    m8 = _load(M8)
    m4 = _load(M4)
    _assert_live(m4)
    m8_usage = m8["device_capacity_comparison"]["mapped_usage"]
    m4_usage = m4["device_capacity_comparison"]["mapped_usage"]
    assert m4["parameters"] == {"DOWN_SYNC_ACTIVATION_MEMORY": 1}
    assert m4_usage["clb_lut_equivalent_primitives"] == 44_161
    assert m4_usage["uram_primitives"] == 15
    assert m4_usage["clb_lut_equivalent_primitives"] < (
        m8_usage["clb_lut_equivalent_primitives"] * 0.55
    )
