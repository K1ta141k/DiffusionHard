import pytest

from diffusion_accel.quality import (
    collect_token_windows,
    evaluate_quality_gates,
    masked_token_metrics,
)


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [int(value) for value in text.split()]


def test_collect_token_windows_is_deterministic() -> None:
    texts = ["1 2 3 4 5", "6 7 8 9 10"]
    first = collect_token_windows(
        texts,
        FakeTokenizer(),
        window_tokens=4,
        samples=2,
        seed=3,
    )
    second = collect_token_windows(
        texts,
        FakeTokenizer(),
        window_tokens=4,
        samples=2,
        seed=3,
    )
    assert first == second
    assert all(len(window) == 4 for window in first)


def test_masked_token_metrics_scores_accuracy_and_nll() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[[4.0, 1.0, 0.0], [0.0, 3.0, 1.0]]])
    targets = torch.tensor([[0, 1]])
    metrics = masked_token_metrics(logits, targets, mask_token_id=2)
    assert metrics["correct"] == 2
    assert metrics["tokens"] == 2
    assert metrics["negative_log_likelihood_sum"] > 0


def test_quality_gate_rejects_accuracy_regression() -> None:
    result = evaluate_quality_gates(
        original_accuracy=0.30,
        isolated_accuracy=0.20,
        original_nll=4.0,
        isolated_nll=4.1,
        cached_top1_agreement=1.0,
        maximum_cached_logit_nrmse=1e-6,
        maximum_accuracy_drop_fraction=0.05,
        maximum_nll_increase_fraction=0.05,
        maximum_logit_nrmse=1e-5,
    )
    assert result["pass"] is False
    assert result["checks"]["isolated_accuracy_drop_within_limit"] is False


def test_quality_gate_accepts_exact_cache_and_small_quality_drop() -> None:
    result = evaluate_quality_gates(
        original_accuracy=0.30,
        isolated_accuracy=0.29,
        original_nll=4.0,
        isolated_nll=4.1,
        cached_top1_agreement=1.0,
        maximum_cached_logit_nrmse=1e-6,
        maximum_accuracy_drop_fraction=0.05,
        maximum_nll_increase_fraction=0.05,
        maximum_logit_nrmse=1e-5,
    )
    assert result["pass"] is True
