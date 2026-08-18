# Tiled output-head kernel

This experimental integer kernel projects INT8 hidden states through INT8
output weights, adds a
signed race-noise score, and retains only one candidate token per position. It
never writes a position-by-vocabulary logit tensor.

One signed Q12.20 multiplier per output channel requantizes each INT8 dot
product into Q5.10 logit units before adding the Q5.10 Gumbel score. Bias values
use the same Q5.10 scale. This makes per-output-channel weight scaling explicit
rather than silently comparing accumulator units with probability noise.

The vocabulary is processed in 16-token tiles. A 64 by 768 hidden-state buffer
is loaded once, each weight value is read once per evaluation, and the weight
is reused across every active position. The local accumulator tile is 64 by 16
INT32 values.

Race noise arrives through a stream in vocabulary-tile, position, lane order.
This makes the future counter-based RNG and Gumbel transform a separate
pipeline stage rather than an external-memory tensor. That producer is not yet
implemented, so this kernel does not establish categorical distribution
quality on hardware.

The checkpoint experiments currently support output-weight-only INT8, not INT8
activations. The Python validation therefore keeps an IEEE FP16 tiled baseline
as the accepted precision path. The C++ INT8 by INT8 datapath is a resource and
interface prototype until activation calibration and full-generation quality
pass. An FP16 activation buffer would increase the modeled local state from
65,680 to 114,832 bytes, which still fits comfortably within K26 on-chip
memory.

Run the native macOS test with:

```bash
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  hls/output_head/output_head.cpp \
  hls/output_head/output_head_test.cpp \
  -o /tmp/diffusion_accel_output_head_test

/tmp/diffusion_accel_output_head_test
```

The test covers a partial vocabulary tile, exact comparison with a dense
integer reference, deterministic tie behavior, mask-token exclusion, and
invalid controls.

When Vitis HLS is available, run:

```bash
export DIFFUSION_ACCEL_FPGA_PART='<installed-part-name>'
export DIFFUSION_ACCEL_CLOCK_PERIOD_NS='3.333'
vitis_hls -f hls/output_head/run_hls.tcl
```

Until that report exists, initiation interval, resource use, timing, AXI
throughput, and achievable parallelism remain unvalidated.
