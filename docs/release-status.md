# v0.1 release status

## Stable

- exact candidate-cache factorization;
- native C++ reveal kernel;
- Philox and fixed-point Gumbel stream;
- streaming noisy argmax and reveal RTL;
- small machine-readable result records.

Focused gate on 2026-08-18:

```text
28 passed in 4.28s
```

## Experimental

Attention, MLP, DDiT, and KV260-facing RTL remain in progress. The last
unrestricted run, before the Apple sampler tests were added, reported:

```text
228 passed, 10 skipped, 13 failed in 685.64s
```

The failures cover stale resource hashes, changed cycle expectations, and one
automatic MLP integration check. They are outside the focused sampler gate.

The last clean full-top Yosys screen used 183,645 CLB-LUT-equivalent
primitives, 162.5 BRAM36-equivalent blocks, 43 UltraRAMs, 936 DSPs, and 130,821
flip-flops. It exceeded raw K26 LUT and BRAM capacity and predates current RTL.

## Not completed

- Vivado implementation and timing closure
- KV260 bitstream and board execution
- measured FPGA latency, bandwidth, and power
- end-to-end FPGA text output
