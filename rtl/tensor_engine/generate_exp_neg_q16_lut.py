#!/usr/bin/env python3
"""Generate the fixed attention exp(-x) Q0.16 lookup table."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def render(step_bits: int = 6, cutoff: float = 16.0) -> str:
    entries = int(cutoff * (1 << step_bits)) + 1
    values = [
        min(65535, round(math.exp(-index / (1 << step_bits)) * 65536))
        for index in range(entries)
    ]
    return "\n".join(f"{value:04x}" for value in values) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != expected:
            raise SystemExit("exp LUT is missing or stale")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(expected, encoding="utf-8")


if __name__ == "__main__":
    main()
