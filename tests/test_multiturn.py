import pytest

from diffusion_accel.multiturn import _cache_bytes


def test_cache_bytes_counts_completed_tokens() -> None:
    torch = pytest.importorskip("torch")

    class Config:
        n_blocks = 2
        hidden_dim = 8

    class FakeModel:
        config = Config()

        def parameters(self):
            yield torch.zeros(1, dtype=torch.float32)

    assert _cache_bytes(FakeModel(), 3) == 2 * 2 * 3 * 8 * 4
