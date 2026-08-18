from pathlib import Path
import shutil

import pytest

from diffusion_accel.rtl_synthesis import (
    _k26_capacity_comparison,
    synthesize_candidate_reveal,
    synthesize_rtl,
)


def test_k26_bram_comparison_uses_36k_equivalent_blocks() -> None:
    comparison = _k26_capacity_comparison(
        {"RAMB18E2": 10, "RAMB36E2": 7, "RAM32M16": 3, "RAM64M8": 4}
    )

    assert comparison["mapped_usage"]["bram_primitives"] == 17
    assert comparison["mapped_usage"]["bram36_equivalent_blocks"] == 12
    assert comparison["mapped_usage"]["distributed_ram_lut_equivalents"] == 56
    assert comparison["mapped_usage"]["clb_lut_equivalent_primitives"] == 56
    assert comparison["capacity_comparison_percent"]["bram_blocks"] == pytest.approx(
        100.0 * 12 / 144
    )


def test_candidate_reveal_maps_to_ultrascale_plus_primitives() -> None:
    if shutil.which("yosys") is None:
        pytest.skip("Yosys is unavailable")

    project_root = Path(__file__).resolve().parents[1]
    result = synthesize_candidate_reveal(
        project_root / "rtl" / "candidate_reveal" / "candidate_reveal_stream.sv",
        device_reference="k26",
    )

    assert result["target_family"] == "xcup"
    assert result["estimated_logic_cells"] > 0
    assert result["primitive_cells"]["FDRE"] > 0
    assert result["primitive_cells"]["LUT6"] > 0
    assert result["timing_validated"] is False
    comparison = result["device_capacity_comparison"]
    assert comparison["reference"]["capacity"]["clb_luts"] == 117_120
    assert comparison["mapped_usage"]["lut_primitives"] == 97
    assert comparison["mapped_usage"]["flip_flop_primitives"] == 33
    assert comparison["mapped_usage"]["bram_primitives"] == 0
    assert comparison["mapped_usage"]["bram36_equivalent_blocks"] == 0
    assert comparison["mapped_usage"]["dsp_primitives"] == 0
    assert comparison["capacity_comparison_percent"]["clb_luts"] < 0.1


def test_generic_synthesis_accepts_iterative_philox_top() -> None:
    if shutil.which("yosys") is None:
        pytest.skip("Yosys is unavailable")

    project_root = Path(__file__).resolve().parents[1]
    result = synthesize_rtl(
        project_root / "rtl" / "philox_gumbel" / "philox4x32_iterative.sv",
        top="philox4x32_iterative",
        device_reference="k26",
    )

    comparison = result["device_capacity_comparison"]
    assert comparison["mapped_usage"]["dsp_primitives"] == 16
    assert comparison["mapped_usage"]["lut_primitives"] == 712
    assert result["timing_validated"] is False


def test_synthesis_failure_includes_yosys_diagnostic() -> None:
    if shutil.which("yosys") is None:
        pytest.skip("Yosys is unavailable")
    project_root = Path(__file__).resolve().parents[1]

    with pytest.raises(RuntimeError, match="Module `missing_top' not found"):
        synthesize_rtl(
            project_root / "rtl" / "candidate_reveal"
            / "candidate_reveal_stream.sv",
            top="missing_top",
        )


def test_synthesis_rejects_invalid_parameter_name() -> None:
    project_root = Path(__file__).resolve().parents[1]

    with pytest.raises(ValueError, match="Verilog identifiers"):
        synthesize_rtl(
            project_root / "rtl" / "candidate_reveal"
            / "candidate_reveal_stream.sv",
            parameters={"BAD-NAME": 2},
        )
