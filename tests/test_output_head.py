import pytest

from diffusion_accel.output_head import (
    tiled_fused_output_candidates,
    validate_tiled_output_head,
)


def test_tiled_output_head_matches_dense_reference() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn((5, 7), generator=generator, dtype=torch.float64)
    weights = torch.randn((13, 7), generator=generator, dtype=torch.float64)
    bias = torch.randn(13, generator=generator, dtype=torch.float64)
    noise = torch.randn((5, 13), generator=generator, dtype=torch.float64)
    reference = hidden @ weights.transpose(0, 1) + bias + noise
    reference[:, 12] = -torch.inf

    for position_tile, vocabulary_tile in ((1, 1), (2, 4), (8, 16)):
        actual = tiled_fused_output_candidates(
            hidden,
            weights,
            bias,
            noise,
            mask_token_id=12,
            position_tile=position_tile,
            vocabulary_tile=vocabulary_tile,
        )
        assert actual.equal(reference.argmax(dim=-1))


def test_tiled_output_head_full_shape_state_fits_on_chip() -> None:
    pytest.importorskip("torch")
    result = validate_tiled_output_head(trials=3)

    assert result["passed"] is True
    assert result["fp16_baseline_validation"]["agreement"] == 1.0
    assert result["hardware_shape"]["candidate_cache_bytes"] == 144
    assert result["hardware_shape"]["total_local_state_bytes"] == 65_680
    assert result["hardware_shape"]["fp16_activation_local_state_bytes"] == 114_832
    assert result["hardware_shape"]["full_int8_output_weight_bytes"] == 38_598_144
