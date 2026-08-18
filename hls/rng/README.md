# Philox RNG stream

This block implements Random123 Philox4x32 with ten rounds and emits one
32-bit random word per vocabulary token in the exact tile, position, lane order
consumed by the fused output-head kernel.

The counter mapping is:

- word 0: vocabulary token block divided by four;
- word 1: position;
- word 2: model evaluation ID;
- word 3: independent stream ID;
- key words: low and high halves of the request seed.

The implementation matches all three
[official Random123 Philox4x32-10 known-answer vectors](https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/tests/kat_vectors#L27-L29).
The generator is deterministic and stateless: replaying
the same counter and key reproduces the same candidate noise without retaining
per-session RNG arrays.

The accepted software design uses an 8-bit normalized-mantissa base table and
three exact correction tables for the common low exponents. The remaining tail
uses an exponent term plus the base table. The four signed INT16 tables occupy
2,048 bytes. `philox_gumbel_stream_kernel` emits Q5.10 signed scores directly
in the output-head stream order.

Run the native macOS test with:

```bash
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  hls/rng/philox_rng.cpp \
  hls/rng/philox_rng_test.cpp \
  -o /tmp/diffusion_accel_philox_rng_test

/tmp/diffusion_accel_philox_rng_test
```

The generated LUT header is reproducible with
`hls/rng/generate_gumbel_lut.py`. Functional behavior is validated natively,
but the complete stream still requires Vitis synthesis and integration with
the output-head IP.
