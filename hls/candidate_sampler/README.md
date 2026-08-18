# Candidate reveal kernel

This is the first standalone FPGA-oriented block in DiffusionAccel. It performs
the cache-hit transition after candidate tokens have already been sampled from
the model distribution.

## Interface

- Up to 256 positions per canvas.
- Packed 16-bit token and candidate IDs, sufficient for MDLM's 50,258-token
  vocabulary.
- Four 64-bit active bitmap words.
- Four 64-bit candidate-valid bitmap words.
- One independent 32-bit random word per position.
- One unsigned Q0.32 reveal threshold in a 64-bit container. `2^32` represents
  probability one exactly.
- One packed 64-bit return value containing 32-bit status and changed-count
  fields.

The kernel commits an active, valid candidate when its random word is below the
threshold. If any position commits, all remaining candidate-valid bits are
cleared because bidirectional model probabilities may have changed globally.
When no position commits, candidates remain valid for the next cache-hit
transition.

## Native macOS test

```bash
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  hls/candidate_sampler/candidate_sampler.cpp \
  hls/candidate_sampler/candidate_sampler_test.cpp \
  -o /tmp/diffusion_accel_candidate_sampler_test

/tmp/diffusion_accel_candidate_sampler_test
```

The source contains Vitis HLS interface and pipeline directives guarded by
`__SYNTHESIS__`. Native compilation validates functional behavior only. Timing,
resource use, AXI behavior, and achieved initiation interval still require a
Vitis HLS build targeting the selected FPGA part.

When Vitis HLS is available, select the exact installed part and run:

```bash
export DIFFUSION_ACCEL_FPGA_PART='<installed-part-name>'
export DIFFUSION_ACCEL_CLOCK_PERIOD_NS='3.333'
vitis_hls -f hls/candidate_sampler/run_hls.tcl
```

The part is intentionally an environment input rather than a hard-coded board
guess. The script runs C simulation, synthesis, and IP export under
`hls/candidate_sampler/build/`, which is ignored by Git.
