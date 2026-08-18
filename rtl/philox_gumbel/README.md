# Philox and Gumbel streaming RTL

This SystemVerilog block emits one raw Philox4x32-10 word and one factorized
Q5.10 Gumbel score per accepted cycle. Its command fields match the C++ counter
mapping: vocabulary block, position, evaluation ID, stream ID, and request seed.

`philox_gumbel_stream.sv` is the simple baseline. Its ten Philox rounds are
combinational, and its four result words are serialized with ready/valid
backpressure. It does not imply that the long path can reach 300 MHz.

The selected architecture is `philox_gumbel_farm_stream.sv`. It combines four
ten-cycle iterative Philox cores, a four-block burst FIFO, a block-to-pair
adapter, and a dual-lane elastic Gumbel transform. The output exposes ready and
valid flow control, two score lanes, and a valid mask for partial vocabulary
blocks.

The 1,024-entry signed INT16 ROM contains one normalized-mantissa table and
three exact low-exponent correction tables. It is generated from the same
source as the C++ header:

```bash
.venv/bin/python hls/rng/generate_gumbel_lut.py \
  --mem-out rtl/philox_gumbel/gumbel_lut_q10.mem
```

Run the functional simulation from the repository root:

```bash
iverilog -g2012 -Wall \
  -s tb_philox_gumbel_stream \
  -o /tmp/tb_philox_gumbel_stream \
  rtl/philox_gumbel/philox_gumbel_stream.sv \
  rtl/philox_gumbel/tb_philox_gumbel_stream.sv

vvp /tmp/tb_philox_gumbel_stream
```

The testbench checks the zero-counter Random123 known-answer vector, reference
Gumbel scores, backpressure stability, deterministic replay, partial vocabulary
tiles, and invalid commands.

Run the integrated stream test and open mapping:

```bash
iverilog -g2012 -s tb_philox_gumbel_farm_stream \
  -o /tmp/tb_philox_gumbel_farm_stream \
  rtl/philox_gumbel/philox4x32_iterative.sv \
  rtl/philox_gumbel/philox4x32_farm.sv \
  rtl/philox_gumbel/gumbel_q10_dual.sv \
  rtl/philox_gumbel/philox_gumbel_farm_stream.sv \
  rtl/philox_gumbel/tb_philox_gumbel_farm_stream.sv

vvp /tmp/tb_philox_gumbel_farm_stream

.venv/bin/diffusion-accel synthesize-rtl \
  --rtl rtl/philox_gumbel/philox4x32_iterative.sv \
        rtl/philox_gumbel/philox4x32_farm.sv \
        rtl/philox_gumbel/gumbel_q10_dual.sv \
        rtl/philox_gumbel/philox_gumbel_farm_stream.sv \
  --top philox_gumbel_farm_stream \
  --out data/results/philox-gumbel-farm-stream-yosys-xcup.json
```

Yosys maps the integrated stream to 66 DSP, 4,136 LUT, 1,880 flip-flop, 2 BRAM,
and 14 distributed-RAM primitives. This does not validate placement, routing,
or the assumed 300 MHz clock.
