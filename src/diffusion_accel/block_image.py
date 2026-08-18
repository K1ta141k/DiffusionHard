"""Execution-order DDR image for one fixed MDLM DDiT block."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from .fixed_attention import (
    fixed_attention_projection_q10,
    fixed_attention_q12,
    fixed_qkv_projection_q12,
    fixed_rotary_q12,
)
from .fixed_mlp import _load_tensors, _quantize_weight_per_output
from .fixed_norm import fixed_layer_norm_q12


SECTION_ALIGNMENT = 4096
AXI_BEAT_BYTES = 64
SMOOTHQUANT_ALPHAS = (
    0.75, 0.5, 0.5, 0.75, 0.5, 0.5,
    0.75, 0.5, 0.5, 0.5, 0.5, 0.75,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pack_values(values: Iterable[int], width: int) -> bytes:
    if width < 1:
        raise ValueError("packed value width must be positive")
    packed = 0
    count = 0
    mask = (1 << width) - 1
    for count, value in enumerate(values, start=1):
        packed |= (int(value) & mask) << ((count - 1) * width)
    return packed.to_bytes((count * width + 7) // 8, "little")


def _pack_fields(fields: Sequence[tuple[Iterable[int], int]]) -> bytes:
    packed = 0
    offset = 0
    for values, width in fields:
        if width < 1:
            raise ValueError("packed field width must be positive")
        mask = (1 << width) - 1
        for value in values:
            packed |= (int(value) & mask) << offset
            offset += width
    return packed.to_bytes((offset + 7) // 8, "little")


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _build_section(
    records: Iterable[bytes],
    *,
    stride: int,
    description: str,
    order: str,
) -> tuple[bytes, dict[str, Any]]:
    if stride < 1 or stride % AXI_BEAT_BYTES != 0 and stride not in {4}:
        raise ValueError("record stride must be four bytes or AXI-beat aligned")
    payload = bytearray()
    count = 0
    record_bytes: int | None = None
    for record in records:
        if len(record) > stride:
            raise ValueError("record exceeds configured stride")
        if record_bytes is None:
            record_bytes = len(record)
        elif record_bytes != len(record):
            raise ValueError("section records must have a fixed payload size")
        payload.extend(record)
        payload.extend(bytes(stride - len(record)))
        count += 1
    if count == 0 or record_bytes is None:
        raise ValueError("section must contain at least one record")
    data = bytes(payload)
    return data, {
        "description": description,
        "order": order,
        "record_count": count,
        "record_payload_bytes": record_bytes,
        "record_stride_bytes": stride,
        "length_bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def export_block_execution_image(
    package_dir: Path,
    out_path: Path,
    *,
    block: int = 0,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Export one block in the exact record order consumed by H4 RTL."""
    if not 0 <= block < 12:
        raise ValueError("block must be between 0 and 11")
    artifact_path = package_dir / "mlp_interstage_int8.safetensors"
    if not artifact_path.is_file():
        raise FileNotFoundError("MLP interstage artifact is unavailable")

    block_name = f"block_{block:02d}"
    folded_name = f"folded.block_{block:02d}"
    residual_name = (
        "folded.embedding"
        if block == 0
        else f"folded.block_{block - 1:02d}.output"
    )
    goldens = _load_tensors(
        package_dir / "golden_tensors.safetensors",
        [residual_name, f"{folded_name}.norm2_unaffine"],
    )
    weights = _load_tensors(
        package_dir / "folded_fp16_weights.safetensors",
        [
            f"{block_name}.qkv.weight",
            f"{block_name}.qkv.bias",
            f"{block_name}.attention_out.weight",
            f"{block_name}.mlp_up.weight",
            "rotary.cos",
            "rotary.sin",
        ],
    )
    mlp = _load_tensors(
        artifact_path,
        [
            f"{block_name}.up_smoothing_reciprocal_q15",
            f"{block_name}.up_output_factor",
            f"{block_name}.up_bias_q10",
            f"{block_name}.interstage_multiplier",
            f"{block_name}.down_weight",
            f"{block_name}.down_output_multiplier",
            f"{block_name}.down_bias_q10",
        ],
    )

    _, norm1_q12, _ = fixed_layer_norm_q12(goldens[residual_name][0])
    fixed_qkv, _, qkv_details = fixed_qkv_projection_q12(
        norm1_q12,
        weights[f"{block_name}.qkv.weight"],
        weights[f"{block_name}.qkv.bias"],
    )
    _, attention_q12, _ = fixed_attention_q12(
        fixed_qkv, weights["rotary.cos"], weights["rotary.sin"]
    )
    _, _, projection_details = fixed_attention_projection_q10(
        attention_q12, weights[f"{block_name}.attention_out.weight"]
    )
    _, _, _, _, rotary_details = fixed_rotary_q12(
        fixed_qkv, weights["rotary.cos"], weights["rotary.sin"]
    )

    qkv_weight = qkv_details["tensors"]["weight_int16"]
    qkv_multiplier = qkv_details["tensors"]["requant_multiplier_q28"]
    qkv_bias = qkv_details["tensors"]["bias_q12"]
    projection_weight = projection_details["tensors"]["weight_int8"]
    projection_multiplier = projection_details["tensors"][
        "requant_multiplier_q24"
    ]
    cosine = rotary_details["tensors"]["cosine_q15"]
    sine = rotary_details["tensors"]["sine_q15"]

    norm2 = goldens[f"{folded_name}.norm2_unaffine"][0]
    up_weight = weights[f"{block_name}.mlp_up.weight"]
    alpha = SMOOTHQUANT_ALPHAS[block]
    smoothing = (
        norm2.abs().amax(dim=0).clamp_min(1e-8).pow(alpha)
        / up_weight.abs().amax(dim=0).clamp_min(1e-8).pow(1.0 - alpha)
    ).clamp_min(1e-8)
    up_weight_q, _ = _quantize_weight_per_output(
        up_weight * smoothing[None, :], 8
    )

    tile_rows: list[list[int | None]] = []
    for head in range(12):
        for kind in range(3):
            for channel_tile in range(11):
                tile_rows.append(
                    [
                        kind * 768 + head * 64 + channel
                        if channel < 64 else None
                        for channel in range(
                            channel_tile * 6, channel_tile * 6 + 6
                        )
                    ]
                )

    sections: list[tuple[str, bytes, dict[str, Any]]] = []
    qkv_metadata = (
        _pack_fields(
            [
                ([0 if row is None else qkv_multiplier[row] for row in rows], 24),
                ([0 if row is None else qkv_bias[row] for row in rows], 18),
            ]
        )
        for rows in tile_rows
    )
    data, info = _build_section(
        qkv_metadata,
        stride=64,
        description="six QKV requant multipliers and biases",
        order="head, kind Q-K-V, six-channel output tile",
    )
    sections.append(("qkv_metadata", data, info))

    qkv_weights = (
        _pack_values(
            [
                0 if row is None else qkv_weight[row, input_tile * 32 + lane]
                for row in rows
                for lane in range(32)
            ],
            16,
        )
        for rows in tile_rows
        for input_tile in range(24)
    )
    data, info = _build_section(
        qkv_weights,
        stride=384,
        description="6 by 32 signed INT16 QKV weight tiles",
        order="head, kind Q-K-V, output tile, input tile",
    )
    sections.append(("qkv_weights", data, info))

    data, info = _build_section(
        (
            _pack_values([cosine[token, pair], sine[token, pair]], 16)
            for token in range(64)
            for pair in range(32)
        ),
        stride=4,
        description="signed Q1.15 rotary cosine and sine pairs",
        order="token, half-head pair",
    )
    sections.append(("rotary_constants", data, info))

    data, info = _build_section(
        (
            _pack_values(projection_multiplier[tile * 6 : tile * 6 + 6], 24)
            for tile in range(128)
        ),
        stride=64,
        description="six folded attention-output requant multipliers",
        order="six-channel output tile",
    )
    sections.append(("projection_metadata", data, info))

    data, info = _build_section(
        (
            _pack_values(
                [
                    projection_weight[tile * 6 + lane, input_tile * 32 + k]
                    for lane in range(6)
                    for k in range(32)
                ],
                8,
            )
            for tile in range(128)
            for input_tile in range(24)
        ),
        stride=192,
        description="6 by 32 signed INT8 attention-output weight tiles",
        order="six-channel output tile, input tile",
    )
    sections.append(("projection_weights", data, info))

    reciprocal = mlp[f"{block_name}.up_smoothing_reciprocal_q15"]
    data, info = _build_section(
        (_pack_values([value], 18) for value in reciprocal),
        stride=4,
        description="unsigned Q3.15 SmoothQuant reciprocal table",
        order="hidden channel",
    )
    sections.append(("mlp_smoothing_reciprocal", data, info))

    up_output_factor = mlp[f"{block_name}.up_output_factor"]
    up_bias = mlp[f"{block_name}.up_bias_q10"]
    interstage = mlp[f"{block_name}.interstage_multiplier"]
    data, info = _build_section(
        (
            _pack_fields(
                [
                    (up_output_factor[tile * 6 : tile * 6 + 6], 18),
                    (up_bias[tile * 6 : tile * 6 + 6], 32),
                    (interstage[tile * 6 : tile * 6 + 6], 24),
                ]
            )
            for tile in range(512)
        ),
        stride=64,
        description="MLP-up factors, biases, and interstage multipliers",
        order="six-channel output tile",
    )
    sections.append(("mlp_up_metadata", data, info))

    data, info = _build_section(
        (
            _pack_values(
                [
                    up_weight_q[tile * 6 + lane, input_tile * 32 + k]
                    for lane in range(6)
                    for k in range(32)
                ],
                8,
            )
            for tile in range(512)
            for input_tile in range(24)
        ),
        stride=192,
        description="6 by 32 signed INT8 MLP-up weight tiles",
        order="six-channel output tile, input tile",
    )
    sections.append(("mlp_up_weights", data, info))

    down_weight = mlp[f"{block_name}.down_weight"]
    down_multiplier = mlp[f"{block_name}.down_output_multiplier"]
    down_bias = mlp[f"{block_name}.down_bias_q10"]
    data, info = _build_section(
        (
            _pack_fields(
                [
                    (
                        [
                            down_multiplier[tile * 6 + lane]
                            for token_lane in range(4)
                            for lane in range(6)
                        ],
                        24,
                    ),
                    (
                        [
                            down_bias[tile * 6 + lane]
                            for token_lane in range(4)
                            for lane in range(6)
                        ],
                        32,
                    ),
                ]
            )
            for tile in range(128)
        ),
        stride=192,
        description="24 MLP-down multipliers and biases",
        order="six-channel output tile, token lane, output lane",
    )
    sections.append(("mlp_down_metadata", data, info))

    data, info = _build_section(
        (
            _pack_values(
                [
                    down_weight[tile * 6 + lane, input_tile * 32 + k]
                    for lane in range(6)
                    for k in range(32)
                ],
                8,
            )
            for tile in range(128)
            for input_tile in range(96)
        ),
        stride=192,
        description="6 by 32 signed INT8 MLP-down weight tiles",
        order="six-channel output tile, input tile",
    )
    sections.append(("mlp_down_weights", data, info))

    image = bytearray()
    manifest_sections: dict[str, Any] = {}
    for name, data, info in sections:
        offset = _align(len(image), SECTION_ALIGNMENT)
        image.extend(bytes(offset - len(image)))
        image.extend(data)
        manifest_sections[name] = {"offset_bytes": offset, **info}
    final_size = _align(len(image), SECTION_ALIGNMENT)
    image.extend(bytes(final_size - len(image)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(image)
    resolved_manifest_path = manifest_path or out_path.with_suffix(
        out_path.suffix + ".json"
    )
    image_reference = os.path.relpath(out_path, resolved_manifest_path.parent)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "format": "mdlm-ddit-block-execution-order",
        "block": block,
        "source_package": str(package_dir),
        "image": {
            "file": image_reference,
            "bytes": len(image),
            "sha256": _sha256_bytes(bytes(image)),
            "section_alignment_bytes": SECTION_ALIGNMENT,
            "axi_beat_bytes": AXI_BEAT_BYTES,
        },
        "sections": manifest_sections,
    }
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(resolved_manifest_path)
    return manifest


def validate_block_execution_image(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_path = manifest_path.parent / manifest["image"]["file"]
    errors: list[str] = []
    if not image_path.is_file():
        return {"passed": False, "errors": ["image file is missing"]}
    image = image_path.read_bytes()
    if len(image) != manifest["image"]["bytes"]:
        errors.append("image byte count does not match manifest")
    if _sha256_bytes(image) != manifest["image"]["sha256"]:
        errors.append("image SHA-256 does not match manifest")
    previous_end = 0
    for name, section in sorted(
        manifest["sections"].items(), key=lambda item: item[1]["offset_bytes"]
    ):
        offset = section["offset_bytes"]
        length = section["length_bytes"]
        stride = section["record_stride_bytes"]
        if offset % manifest["image"]["section_alignment_bytes"]:
            errors.append(f"{name} offset is not section aligned")
        if offset < previous_end:
            errors.append(f"{name} overlaps the preceding section")
        payload = image[offset : offset + length]
        if len(payload) != length:
            errors.append(f"{name} extends beyond the image")
        if length != section["record_count"] * stride:
            errors.append(f"{name} record geometry is inconsistent")
        if _sha256_bytes(payload) != section["sha256"]:
            errors.append(f"{name} SHA-256 does not match")
        pad_start = section["record_payload_bytes"]
        for record_offset in range(0, len(payload), stride):
            if any(payload[record_offset + pad_start : record_offset + stride]):
                errors.append(f"{name} contains nonzero record padding")
                break
        if any(image[previous_end:offset]):
            errors.append(f"{name} has nonzero section-alignment padding")
        previous_end = offset + length
    if any(image[previous_end:]):
        errors.append("image has nonzero final alignment padding")
    return {
        "passed": not errors,
        "errors": errors,
        "image": str(image_path),
        "bytes": len(image),
        "section_count": len(manifest["sections"]),
    }


def read_block_image_record(
    manifest_path: Path, section_name: str, record_index: int
) -> bytes:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    section = manifest["sections"][section_name]
    if not 0 <= record_index < section["record_count"]:
        raise IndexError("block image record is out of range")
    image_path = manifest_path.parent / manifest["image"]["file"]
    offset = section["offset_bytes"] + record_index * section["record_stride_bytes"]
    with image_path.open("rb") as stream:
        stream.seek(offset)
        return stream.read(section["record_payload_bytes"])
