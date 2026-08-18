import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data/results/h4-packed-all12-heads-h0-rtl.json"


def test_all_twelve_head_h0_run_is_live_exact_evidence() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    for relative_path, expected_hash in evidence["source_sha256"].items():
        actual_hash = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash

    assert evidence["head_count"] == 12
    assert evidence["attention_values_compared"] == 12 * 64 * 64
    assert evidence["rtl_testbench_exit_code"] == 0
    assert evidence["rtl_testbench_pass"] is True
    assert evidence["pytest_exit_code"] == 1
    assert evidence["corrected_cycle_window"]["minimum"] <= evidence[
        "cycle_count"
    ] <= evidence["corrected_cycle_window"]["maximum"]
