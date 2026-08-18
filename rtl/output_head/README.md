# Noisy output-head reduction RTL

`ordered_noisy_argmax_reducer.sv` consumes two pre-biased signed Q5.10 model
scores and two Q5.10 Gumbel scores per accepted transfer. It retains only the
best unmasked token for the current position and emits one compact candidate at
each position boundary.

`philox_noisy_argmax_stream.sv` joins that reducer with the four-core Philox and
dual-lane Gumbel stream. Its lockstep handshake advances RNG and model-score
inputs together, so bubbles or backpressure cannot misalign token metadata.

The compact reducer relies on position-major, token-major input. This matches
the selected RNG stream. Lower token IDs win exact score ties, independent of
lane order.

Run the integrated simulation from the repository root:

```bash
iverilog -g2012 -s tb_philox_noisy_argmax_stream \
  -o /tmp/tb_philox_noisy_argmax_stream \
  rtl/philox_gumbel/philox4x32_iterative.sv \
  rtl/philox_gumbel/philox4x32_farm.sv \
  rtl/philox_gumbel/gumbel_q10_dual.sv \
  rtl/philox_gumbel/philox_gumbel_farm_stream.sv \
  rtl/output_head/ordered_noisy_argmax_reducer.sv \
  rtl/output_head/philox_noisy_argmax_stream.sv \
  rtl/output_head/tb_philox_noisy_argmax_stream.sv

vvp /tmp/tb_philox_noisy_argmax_stream
```

Yosys maps the score-input boundary to 66 DSP, 4,787 LUT, 2,056 flip-flop, 2
BRAM, and 14 distributed-RAM primitives.

`dual_requantizer_q20.sv` adds two exact signed Q12.20 requantization lanes with
half-away-from-zero rounding and signed bias addition.
`philox_accumulator_argmax_stream.sv` connects these lanes to the score-input
boundary. The resulting accumulator-to-candidate top maps to 74 DSP, 5,117
LUT, 2,123 flip-flop, 2 BRAM, and 14 distributed-RAM primitives. This still
excludes the dot-product MAC producer and does not establish vendor timing
closure.
