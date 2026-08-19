import sys
import types

import pytest


mx = pytest.importorskip("mlx.core")

from diffusion_accel.mlx_mdlm import (  # noqa: E402
    CiderQuantizationPlan,
    MLXMDLM,
    MLXQuantizationPlan,
    _reveal_groups,
    apply_cider_quantization,
    apply_mlx_quantization,
    compile_mlx_event_sampler,
    run_compiled_mlx_event_sampler,
    run_mlx_event_sampler,
)
import mlx.nn as nn  # noqa: E402


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


def test_mixed_quantization_targets_only_requested_families():
    model = MLXMDLM()
    apply_mlx_quantization(
        model,
        MLXQuantizationPlan(
            output_head_bits=8,
            mlp_up_bits=4,
        ),
    )

    assert isinstance(model.backbone.output_layer.linear, nn.QuantizedLinear)
    assert isinstance(model.backbone.blocks[0].mlp[0], nn.QuantizedLinear)
    assert isinstance(model.backbone.blocks[0].mlp[2], nn.Linear)
    assert isinstance(model.backbone.blocks[0].attn_qkv, nn.Linear)


def test_cider_quantization_targets_selected_block_families(monkeypatch):
    class _FakeCiderLinear(nn.Module):
        @classmethod
        def from_float(cls, module, **kwargs):
            del module, kwargs
            return cls()

    cider = types.ModuleType("cider")
    cider.is_available = lambda: True
    cider_nn = types.ModuleType("cider.nn")
    cider_nn.CiderLinear = _FakeCiderLinear
    monkeypatch.setitem(sys.modules, "cider", cider)
    monkeypatch.setitem(sys.modules, "cider.nn", cider_nn)

    model = MLXMDLM()
    replacements = apply_cider_quantization(
        model,
        CiderQuantizationPlan(mlp_down_layers=tuple(range(12))),
    )

    assert replacements == 36
    assert isinstance(model.backbone.blocks[0].attn_qkv, _FakeCiderLinear)
    assert isinstance(model.backbone.blocks[0].mlp[0], _FakeCiderLinear)
    assert isinstance(model.backbone.blocks[0].mlp[2], _FakeCiderLinear)
    assert isinstance(model.backbone.blocks[0].attn_out, nn.Linear)


def test_cider_quantization_plan_rejects_invalid_group_size():
    with pytest.raises(ValueError, match="group_size"):
        CiderQuantizationPlan(group_size=32)


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
