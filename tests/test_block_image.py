from __future__ import annotations

import json
from pathlib import Path

import pytest

from diffusion_accel.block_image import (
    _pack_fields,
    _pack_values,
    export_block_execution_image,
    read_block_image_record,
    validate_block_execution_image,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/hardware/mdlm-owt-169m-h0"


def test_packed_values_preserve_lsb_lane_order_and_signed_bits() -> None:
    assert _pack_values([-1, 2, -3], 8) == bytes([0xFF, 0x02, 0xFD])
    assert _pack_fields([([1, 2], 4), ([-1], 8)]) == bytes([0x21, 0xFF])


def test_block_zero_execution_image_is_aligned_and_self_validating(
    tmp_path: Path,
) -> None:
    if not (PACKAGE / "mlp_interstage_int8.safetensors").is_file():
        pytest.skip("H0 hardware package is unavailable")
    image_path = tmp_path / "block_00_execution.bin"
    manifest = export_block_execution_image(PACKAGE, image_path, block=0)
    manifest_path = Path(manifest["manifest_path"])
    validation = validate_block_execution_image(manifest_path)
    assert validation["passed"], validation["errors"]
    assert validation["section_count"] == 10
    assert 8_000_000 < validation["bytes"] < 10_000_000

    expected_records = {
        "qkv_metadata": 396,
        "qkv_weights": 9_504,
        "rotary_constants": 2_048,
        "projection_metadata": 128,
        "projection_weights": 3_072,
        "mlp_smoothing_reciprocal": 768,
        "mlp_up_metadata": 512,
        "mlp_up_weights": 12_288,
        "mlp_down_metadata": 128,
        "mlp_down_weights": 12_288,
    }
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, count in expected_records.items():
        section = loaded["sections"][name]
        assert section["record_count"] == count
        assert section["offset_bytes"] % 4096 == 0
        record = read_block_image_record(manifest_path, name, count - 1)
        assert len(record) == section["record_payload_bytes"]

    with pytest.raises(IndexError, match="out of range"):
        read_block_image_record(manifest_path, "mlp_down_weights", 12_288)


def test_block_image_validator_rejects_mutated_payload(tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.bin"
    image_path.write_bytes(bytes(4096))
    manifest_path = tmp_path / "tiny.json"
    manifest_path.write_text(
        json.dumps(
            {
                "image": {
                    "file": image_path.name,
                    "bytes": 4096,
                    "sha256": "0" * 64,
                    "section_alignment_bytes": 4096,
                },
                "sections": {},
            }
        ),
        encoding="utf-8",
    )
    result = validate_block_execution_image(manifest_path)
    assert not result["passed"]
    assert result["errors"] == ["image SHA-256 does not match manifest"]


def test_rtl_address_constants_match_committed_block_manifest() -> None:
    manifest_path = ROOT / "data/results/block-00-execution-image.json"
    if not manifest_path.is_file():
        pytest.skip("committed block image manifest is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rtl = (
        ROOT
        / "rtl/tensor_engine/mdlm_block_parameter_address_generator.sv"
    ).read_text(encoding="utf-8")
    section_order = [
        ("SECTION_QKV_METADATA", "qkv_metadata"),
        ("SECTION_QKV_WEIGHTS", "qkv_weights"),
        ("SECTION_ROTARY_CONSTANTS", "rotary_constants"),
        ("SECTION_PROJECTION_METADATA", "projection_metadata"),
        ("SECTION_PROJECTION_WEIGHTS", "projection_weights"),
        ("SECTION_MLP_RECIPROCAL", "mlp_smoothing_reciprocal"),
        ("SECTION_MLP_UP_METADATA", "mlp_up_metadata"),
        ("SECTION_MLP_UP_WEIGHTS", "mlp_up_weights"),
        ("SECTION_MLP_DOWN_METADATA", "mlp_down_metadata"),
        ("SECTION_MLP_DOWN_WEIGHTS", "mlp_down_weights"),
    ]
    case_statement = rtl.index("case (section_id)")
    for constant, name in section_order:
        section = manifest["sections"][name]
        case_start = rtl.index(f"{constant}: begin", case_statement)
        case_end = rtl.index("end", case_start)
        case_body = rtl[case_start:case_end]
        assert f"section_offset = 32'd{section['offset_bytes']};" in case_body
        assert f"record_count = 14'd{section['record_count']};" in case_body
        assert (
            f"record_payload_bytes = 10'd{section['record_payload_bytes']};"
            in case_body
        )
