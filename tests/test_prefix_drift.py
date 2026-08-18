import pytest

from diffusion_accel.prefix_drift import tensor_drift_metrics


def test_tensor_drift_metrics_detect_exact_match() -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    metrics = tensor_drift_metrics(reference, reference.clone())
    assert metrics["exact_equal"] is True
    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["normalized_rmse"] == 0.0
    assert metrics["fraction_changed_gt_1e_6"] == 0.0


def test_tensor_drift_metrics_detect_change() -> None:
    torch = pytest.importorskip("torch")
    reference = torch.tensor([1.0, 2.0, 3.0])
    changed = torch.tensor([1.0, 2.5, 3.0])
    metrics = tensor_drift_metrics(reference, changed)
    assert metrics["exact_equal"] is False
    assert metrics["max_absolute_error"] == pytest.approx(0.5)
    assert metrics["fraction_changed_gt_1e_6"] == pytest.approx(1 / 3)


def test_tensor_drift_metrics_requires_matching_shapes() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="same shape"):
        tensor_drift_metrics(torch.zeros(2), torch.zeros(3))
