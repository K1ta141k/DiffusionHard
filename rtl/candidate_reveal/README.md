# Streaming candidate reveal RTL

This module is the cycle-accurate streaming form of the candidate-reveal HLS
kernel. It accepts one position per cycle under ready/valid flow control.

The stream carries the current token, candidate token, active bit, candidate
valid bit, and one 32-bit random word. The command supplies a position count and
an unsigned Q0.32 reveal threshold. The output supplies the updated token and
active state plus a changed bit for each accepted position.

If any token changes, `invalidate_all` is asserted with `done`. A downstream
bitmap controller must clear all candidate-valid bits. This separate signal is
required because a later changed position can invalidate candidates that were
already emitted earlier in the stream.

Run the local simulation with:

```bash
iverilog -g2012 -Wall \
  -s tb_candidate_reveal_stream \
  -o /tmp/tb_candidate_reveal_stream \
  rtl/candidate_reveal/candidate_reveal_stream.sv \
  rtl/candidate_reveal/tb_candidate_reveal_stream.sv

vvp /tmp/tb_candidate_reveal_stream
```

The testbench covers zero and unit probability, backpressure, partial commits,
invalid commands, and 1,024 deterministic randomized position checks. It is a
functional simulation, not a post-synthesis timing result.

When Yosys is available, map the block to UltraScale+ primitives with:

```bash
.venv/bin/diffusion-accel synthesize-candidate-rtl \
  --family xcup \
  --device-reference k26 \
  --out data/results/candidate-reveal-yosys-xcup.json
```

The K26 comparison uses capacities from AMD DS987 revision 1.6. The current
mapping reports 97 LUT primitives, 33 flip-flop primitives, and no BRAM,
UltraRAM, or DSP primitives. The raw sums correspond to 0.0828% of K26 CLB LUTs
and 0.0141% of its CLB flip-flops. This is useful for structural cell counts.
It is not a substitute for K26 placement, routing, clock analysis, or the Vitis
HLS report.
