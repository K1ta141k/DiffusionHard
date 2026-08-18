from pathlib import Path

import pytest

from diffusion_accel.attention_int8 import screen_packed_int8_attention


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"


def test_packed_int8_attention_screen_is_reproducible() -> None:
    report = screen_packed_int8_attention(PACKAGE)
    summary = report["summary"]
    assert len(report["blocks"]) == 12
    assert summary["mean_attention_relative_rms_vs_fixed18"] == pytest.approx(
        0.030985677614808083
    )
    assert summary["maximum_attention_relative_rms_vs_fixed18"] == pytest.approx(
        0.03738418594002724
    )
    assert summary["minimum_attention_cosine_similarity_vs_fixed18"] > 0.9993
    assert summary["maximum_attention_absolute_error_q12"] == 721
