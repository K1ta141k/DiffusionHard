"""Optional open-source synthesis checks for standalone RTL blocks."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
import subprocess
from typing import Dict, Optional, Sequence, Union


K26_RESOURCE_REFERENCE = {
    "device": "XCK26-SFVC784-2LV",
    "document": "AMD DS987 K26 SOM Data Sheet, revision 1.6",
    "document_date": "2026-03-25",
    "source": "https://docs.amd.com/r/en-US/ds987-k26-som/Programmable-Logic",
    "capacity": {
        "clb_luts": 117_120,
        "clb_flip_flops": 234_240,
        "bram_blocks": 144,
        "uram_blocks": 64,
        "dsp_slices": 1_248,
    },
}

_DISTRIBUTED_RAM_LUT_EQUIVALENTS = {
    "RAM32M16": 8,
    "RAM64M8": 8,
}


def _k26_capacity_comparison(cell_counts: Dict[str, int]) -> Dict[str, object]:
    """Compare mapped primitives with documented K26 capacities.

    The LUT total is deliberately a primitive sum rather than a packed CLB LUT
    count. Vendor placement may pack it differently, so this is a conservative
    pre-place-and-route comparison rather than a utilization report.
    """
    bram18_primitives = sum(
        count for cell, count in cell_counts.items() if cell.startswith("RAMB18")
    )
    bram36_primitives = sum(
        count for cell, count in cell_counts.items() if cell.startswith("RAMB36")
    )
    logic_lut_primitives = sum(
        count
        for cell, count in cell_counts.items()
        if re.fullmatch(r"LUT[1-6]", cell)
    )
    distributed_ram_lut_equivalents = sum(
        count * _DISTRIBUTED_RAM_LUT_EQUIVALENTS.get(cell, 0)
        for cell, count in cell_counts.items()
    )
    mapped_usage = {
        "lut_primitives": logic_lut_primitives,
        "flip_flop_primitives": sum(
            count for cell, count in cell_counts.items() if cell.startswith("FD")
        ),
        "bram_primitives": sum(
            count for cell, count in cell_counts.items() if cell.startswith("RAMB")
        ),
        "bram18_primitives": bram18_primitives,
        "bram36_primitives": bram36_primitives,
        "bram36_equivalent_blocks": (
            bram36_primitives + bram18_primitives / 2.0
        ),
        "uram_primitives": sum(
            count for cell, count in cell_counts.items() if cell.startswith("URAM")
        ),
        "distributed_ram_primitives": sum(
            count
            for cell, count in cell_counts.items()
            if cell.startswith("RAM") and not cell.startswith("RAMB")
        ),
        "distributed_ram_lut_equivalents": distributed_ram_lut_equivalents,
        "clb_lut_equivalent_primitives": (
            logic_lut_primitives + distributed_ram_lut_equivalents
        ),
        "dsp_primitives": sum(
            count for cell, count in cell_counts.items() if cell.startswith("DSP")
        ),
    }
    capacity = K26_RESOURCE_REFERENCE["capacity"]
    comparison_percent = {
        "clb_luts": (
            100.0
            * mapped_usage["clb_lut_equivalent_primitives"]
            / capacity["clb_luts"]
        ),
        "clb_flip_flops": (
            100.0
            * mapped_usage["flip_flop_primitives"]
            / capacity["clb_flip_flops"]
        ),
        "bram_blocks": (
            100.0
            * mapped_usage["bram36_equivalent_blocks"]
            / capacity["bram_blocks"]
        ),
        "uram_blocks": (
            100.0 * mapped_usage["uram_primitives"] / capacity["uram_blocks"]
        ),
        "dsp_slices": (
            100.0 * mapped_usage["dsp_primitives"] / capacity["dsp_slices"]
        ),
    }
    return {
        "reference": K26_RESOURCE_REFERENCE,
        "mapped_usage": mapped_usage,
        "capacity_comparison_percent": comparison_percent,
        "scope": "pre-place-and-route-primitive-comparison",
    }


def synthesize_rtl(
    rtl_path: Union[Path, Sequence[Path]],
    *,
    top: str = "candidate_reveal_stream",
    family: str = "xcup",
    device_reference: Optional[str] = None,
    parameters: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """Run Yosys technology mapping and return the final primitive counts."""
    yosys = shutil.which("yosys")
    if yosys is None:
        raise RuntimeError("yosys is required for open-source RTL synthesis")
    rtl_paths = [rtl_path] if isinstance(rtl_path, Path) else list(rtl_path)
    if not rtl_paths:
        raise ValueError("at least one RTL path is required")
    for source_path in rtl_paths:
        if not source_path.is_file():
            raise ValueError("RTL path does not exist: %s" % source_path)
    if family not in {"xcup", "xcu", "xc7"}:
        raise ValueError("family must be xcup, xcu, or xc7")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", top):
        raise ValueError("top must be a Verilog identifier")
    resolved_parameters = {} if parameters is None else dict(parameters)
    for name, value in resolved_parameters.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("parameter names must be Verilog identifiers")
        if not isinstance(value, int):
            raise ValueError("parameter values must be integers")

    source_hash = hashlib.sha256()
    for source_path in rtl_paths:
        source_hash.update(str(source_path).encode("utf-8"))
        source_hash.update(b"\0")
        source_hash.update(source_path.read_bytes())
        source_hash.update(b"\0")
    rtl_arguments = " ".join(
        '"%s"'
        % str(source_path).replace("\\", "\\\\").replace('"', '\\"')
        for source_path in rtl_paths
    )
    parameter_command = "".join(
        "chparam -set %s %d %s; " % (name, value, top)
        for name, value in sorted(resolved_parameters.items())
    )
    command = (
        "read_verilog -lib +/xilinx/cells_xtra.v; "
        "read_verilog -sv %s; "
        "%s"
        "hierarchy -check -top %s; "
        "synth_xilinx -family %s -top %s -noiopad -noclkbuf; "
        "stat"
    ) % (rtl_arguments, parameter_command, top, family, top)
    completed = subprocess.run(
        [yosys, "-Q", "-p", command],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stdout + completed.stderr)[-8_000:]
        raise RuntimeError(
            "Yosys synthesis failed with exit code %d:\n%s"
            % (completed.returncode, diagnostic)
        )
    output = completed.stdout
    final_statistics = output.rsplit("4. Printing statistics.", maxsplit=1)[-1]
    cell_counts = {
        cell: int(count)
        for count, cell in re.findall(
            r"^\s+(\d+)\s+([A-Z][A-Z0-9_]*)\s*$",
            final_statistics,
            flags=re.MULTILINE,
        )
    }
    if not cell_counts:
        raise RuntimeError("Yosys output did not contain final primitive counts")
    logic_cell_matches = re.findall(
        r"Estimated number of LCs:\s+(\d+)",
        output,
    )
    if not logic_cell_matches:
        raise RuntimeError("Yosys output did not contain an LC estimate")
    version = subprocess.run(
        [yosys, "-V"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result: Dict[str, object] = {
        "tool": version,
        "target_family": family,
        "top": top,
        "parameters": resolved_parameters,
        "rtl_paths": [str(source_path) for source_path in rtl_paths],
        "rtl_sha256": source_hash.hexdigest(),
        "primitive_cells": dict(sorted(cell_counts.items())),
        "total_primitive_cells": sum(cell_counts.values()),
        "estimated_logic_cells": int(logic_cell_matches[-1]),
        "scope": (
            "open-source-technology-mapping-not-vendor-place-and-route"
        ),
        "timing_validated": False,
        "resource_counts_validated_by_vitis": False,
    }
    if len(rtl_paths) == 1:
        result["rtl_path"] = str(rtl_paths[0])
    if device_reference is not None:
        if device_reference != "k26":
            raise ValueError("device_reference must be k26 or None")
        result["device_capacity_comparison"] = _k26_capacity_comparison(
            cell_counts
        )
    return result


def synthesize_candidate_reveal(
    rtl_path: Union[Path, Sequence[Path]],
    *,
    top: str = "candidate_reveal_stream",
    family: str = "xcup",
    device_reference: Optional[str] = None,
) -> Dict[str, object]:
    """Compatibility wrapper for the original candidate-reveal command."""
    return synthesize_rtl(
        rtl_path,
        top=top,
        family=family,
        device_reference=device_reference,
    )
