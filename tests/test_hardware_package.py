from diffusion_accel.hardware_package import (
    _hardware_weight_order,
    _validate_binary_layout,
    _tensor_manifest,
    deterministic_hardware_input,
)


def test_deterministic_hardware_input_has_fixed_mask_pattern() -> None:
    values = deterministic_hardware_input(64, 50_258, 50_257)

    assert len(values) == 64
    assert values[0] == 17
    assert values[1] == 50_257
    assert values[2] == 50_257
    assert values[3] == (3 * 7919 + 17) % 50_257
    assert sum(value == 50_257 for value in values) == 42
    assert all(0 <= value <= 50_257 for value in values)


def test_tensor_manifest_records_shape_dtype_and_bytes() -> None:
    import torch

    manifest = _tensor_manifest(
        {
            "a": torch.zeros((2, 3), dtype=torch.float16),
            "b": torch.zeros(4, dtype=torch.int32),
        }
    )

    assert manifest["tensor_count"] == 2
    assert manifest["tensor_bytes"] == 28
    assert manifest["tensors"]["a"] == {
        "dtype": "float16",
        "shape": [2, 3],
        "bytes": 12,
    }


def test_hardware_weight_order_matches_execution_groups() -> None:
    assert _hardware_weight_order(
        [
            "rotary.cos",
            "output.weight",
            "block_01.qkv.weight",
            "embedding.weight",
            "block_00.qkv.weight",
        ]
    ) == [
        "embedding.weight",
        "block_00.qkv.weight",
        "block_01.qkv.weight",
        "output.weight",
        "rotary.cos",
    ]


def test_binary_layout_validation_rejects_unaligned_offset() -> None:
    manifest = {
        "weights": {"tensors": {"embedding.weight": {"bytes": 8}}},
        "weight_binary_layout": {
            "alignment_bytes": 4096,
            "total_bytes": 4096,
            "tensors": {
                "embedding.weight": {
                    "offset_bytes": 2,
                    "length_bytes": 8,
                }
            },
        },
    }
    assert _validate_binary_layout(manifest) == [
        "embedding.weight offset is not aligned"
    ]
