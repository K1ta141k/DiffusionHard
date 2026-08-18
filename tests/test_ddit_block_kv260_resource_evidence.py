import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / (
    "data/results/"
    "ddit-block-kv260-axi-top-packed-m4-pv-stream16-qk-bank32-yosys-xcup.json"
)


def test_full_block_resource_map_is_live_and_drives_fit_work() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for relative_path in evidence["rtl_paths"]:
        source = ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    assert digest.hexdigest() == evidence["rtl_sha256"]

    assert evidence["top"] == "ddit_block_kv260_axi_top"
    assert evidence["scope"] == (
        "open-source-technology-mapping-not-vendor-place-and-route"
    )
    assert evidence["timing_validated"] is False
    assert evidence["resource_counts_validated_by_vitis"] is False
    comparison = evidence["device_capacity_comparison"]
    usage = comparison["mapped_usage"]
    percent = comparison["capacity_comparison_percent"]
    assert usage["clb_lut_equivalent_primitives"] == 183_645
    assert usage["bram36_equivalent_blocks"] == 162.5
    assert usage["uram_primitives"] == 43
    assert usage["dsp_primitives"] == 936
    assert usage["flip_flop_primitives"] == 130_821
    assert percent["clb_luts"] > 100
    assert percent["bram_blocks"] > 100
    assert percent["uram_blocks"] < 100
    assert percent["dsp_slices"] < 100
    assert percent["clb_flip_flops"] < 100
