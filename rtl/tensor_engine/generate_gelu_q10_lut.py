#!/usr/bin/env python3
"""Generate the frozen tanh-GELU Q5.10 lookup table used by RTL."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


FRACTION_BITS = 10
LUT_ADDRESS_BITS = 10
STEP_BITS = 4
OUTPUT_WIDTH = 16
LOWER_Q = -(8 << FRACTION_BITS)


def gelu_tanh(value: float) -> float:
    coefficient = math.sqrt(2.0 / math.pi)
    return 0.5 * value * (
        1.0 + math.tanh(coefficient * (value + 0.044715 * value**3))
    )


def render() -> str:
    lines = []
    mask = (1 << OUTPUT_WIDTH) - 1
    for address in range(1 << LUT_ADDRESS_BITS):
        input_q = LOWER_Q + (address << STEP_BITS)
        value = input_q / float(1 << FRACTION_BITS)
        output_q = round(gelu_tanh(value) * (1 << FRACTION_BITS))
        lines.append(f"{output_q & mask:04x}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="ascii") != expected:
            raise SystemExit("GELU LUT is stale")
        return
    args.out.write_text(expected, encoding="ascii")


if __name__ == "__main__":
    main()
