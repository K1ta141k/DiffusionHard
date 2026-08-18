#!/usr/bin/env python3
"""Generate the factorized Q5.10 Gumbel lookup tables for HLS."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

MANTISSA_BITS = 8
FRACTION_BITS = 10
CORRECTION_EXPONENTS = 3


def _values() -> tuple[list[int], list[list[int]], int, int]:
    scale = 1 << FRACTION_BITS
    base = []
    correction = [[] for _ in range(CORRECTION_EXPONENTS)]
    for index in range(1 << MANTISSA_BITS):
        normalized = ((1 << MANTISSA_BITS) + index + 0.5) / (
            1 << MANTISSA_BITS
        )
        base.append(round(-math.log(normalized) * scale))
        for exponent in range(CORRECTION_EXPONENTS):
            delta = normalized * 2.0 ** (-(exponent + 1))
            correction[exponent].append(
                round(-math.log(-math.log1p(-delta)) * scale)
            )
    top_uniform = 1.0 - 0.5 / float(1 << 32)
    top = round(-math.log(-math.log(top_uniform)) * scale)
    log_two = round(math.log(2.0) * scale)
    return base, correction, top, log_two


def _row(values: list[int], indent: str = "    ") -> str:
    chunks = []
    for start in range(0, len(values), 16):
        chunks.append(indent + ", ".join(str(value) for value in values[start:start + 16]))
    return ",\n".join(chunks)


def render() -> str:
    base, correction, top, log_two = _values()
    corrections = []
    for values in correction:
        corrections.append("    {{\n%s\n    }}" % _row(values, "        "))
    return """#ifndef DIFFUSION_ACCEL_GUMBEL_LUT_HPP
#define DIFFUSION_ACCEL_GUMBEL_LUT_HPP

#include <array>
#include <cstdint>

namespace diffusion_accel {

constexpr std::uint32_t kGumbelFractionBits = %d;
constexpr std::uint32_t kGumbelCorrectionExponents = %d;
constexpr std::int16_t kGumbelLogTwoQ = %d;
constexpr std::int16_t kGumbelTopWordQ = %d;

constexpr std::array<std::int16_t, 256> kGumbelBaseQ = {{
%s
}};

constexpr std::array<std::array<std::int16_t, 256>, 3> kGumbelCorrectionQ = {{
%s
}};

}  // namespace diffusion_accel

#endif
""" % (
        FRACTION_BITS,
        CORRECTION_EXPONENTS,
        log_two,
        top,
        _row(base),
        ",\n".join(corrections),
    )


def render_mem() -> str:
    base, correction, _, _ = _values()
    values = base + [value for table in correction for value in table]
    return "".join("%04x\n" % (value & 0xFFFF) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mem-out", type=Path)
    args = parser.parse_args()
    if args.out is None and args.mem_out is None:
        parser.error("at least one of --out or --mem-out is required")
    if args.out is not None:
        args.out.write_text(render(), encoding="utf-8")
    if args.mem_out is not None:
        args.mem_out.write_text(render_mem(), encoding="utf-8")


if __name__ == "__main__":
    main()
