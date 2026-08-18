import pytest


mx = pytest.importorskip("mlx.core")

from diffusion_accel.mlx_mdlm import (  # noqa: E402
    _reveal_groups,
    compile_mlx_event_sampler,
    run_compiled_mlx_event_sampler,
    run_mlx_event_sampler,
)


class _FixedTokenModel:
    def selected_logits(self, token_ids, positions):
        del token_ids
        token = mx.array(42, dtype=mx.int32)
        return mx.full((positions.shape[0], 50_258), -1e9).at[:, token].add(1e9)


def test_reveal_groups_partition_every_position_once():
    groups = _reveal_groups(positions=64, steps=32, seed=7)
    flattened = [position for group in groups for position in group]
    assert sorted(flattened) == list(range(64))


def test_reveal_groups_reject_invalid_dimensions():
    with pytest.raises(ValueError, match="positive"):
        _reveal_groups(positions=64, steps=0, seed=7)


def test_compiled_sampler_matches_event_sampler_and_preserves_prefix():
    model = _FixedTokenModel()
    prefix = [11, 12, 13]
    expected, expected_metrics = run_mlx_event_sampler(
        model,
        canvas_tokens=4,
        prefix_token_ids=prefix,
        steps=4,
        seed=3,
    )
    sampler, compiled_metadata = compile_mlx_event_sampler(
        model,
        canvas_tokens=4,
        steps=4,
        reveal_seed=3,
        prefix_tokens=len(prefix),
    )
    run_compiled_mlx_event_sampler(
        sampler,
        canvas_tokens=4,
        prefix_token_ids=prefix,
        sampling_seed=3,
    )
    actual, _ = run_compiled_mlx_event_sampler(
        sampler,
        canvas_tokens=4,
        prefix_token_ids=prefix,
        sampling_seed=3,
    )

    assert actual.tolist() == expected.tolist()
    assert actual[0, : len(prefix)].tolist() == prefix
    assert actual[0, len(prefix) :].tolist() == [42, 42, 42, 42]
    assert compiled_metadata["model_evaluations"] == expected_metrics["model_evaluations"]
