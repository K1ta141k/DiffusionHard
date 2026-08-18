# Milestone results

## 2026-08-16: Counter-based RNG and fixed-point Gumbel stream

The fused output head now has a deterministic noise producer. It implements
ten-round Philox4x32 and matches all three
[official Random123 known-answer vectors](https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/tests/kat_vectors#L27-L29).
The counter maps vocabulary block, position, model evaluation, and independent
stream ID; the two key words hold the request seed. No mutable per-session RNG
array is required.

A flat uniformly addressed Gumbel table did not preserve enough precision in
the upper tail for a 50,258-token vocabulary. The selected transform instead
normalizes the distance from probability one into a leading-zero exponent and
an 8-bit mantissa. One base table handles the tail, while three correction
tables handle the common low exponents exactly. All scores are signed Q5.10.

| Gumbel validation | Result |
| --- | ---: |
| Categorical Monte Carlo trials, 16 classes | 200,000 |
| Total-variation distance | 0.004420 |
| Accepted maximum | 0.01 |
| Full-vocabulary pathwise samples | 1,024 |
| Four-bit mantissa agreement | 98.828% |
| Six-bit mantissa agreement | 99.707% |
| Eight-bit factorized agreement | 100.000% |
| Selected signed INT16 LUT storage | 2,048 bytes |

The native C++ integration now runs the complete candidate-producer boundary:
Philox, factorized Gumbel conversion, per-output-channel Q12.20 requantization,
tiled INT8 matrix projection, and fused noisy argmax. It matches an independent
dense integer reference, including a partial vocabulary tile. Random scores
flow directly between blocks and never become a DDR tensor.

Adding the Gumbel tables and one tile of requantization multipliers and biases
brings the FP16-activation local-state estimate to approximately 117,008 bytes.
This excludes stream FIFOs, alignment, and double buffering.

### RTL architecture result

The first serial score stream used a fully combinational ten-round Philox
function. It passed RTL simulation, but open-source UltraScale+ mapping used
155 DSP primitives and 6,481 LUT primitives. At one score per cycle it would
also take 10.722 ms to produce a 64 by 50,258 score field at an assumed 300 MHz,
slower than the 8.041 ms ideal compute bound of the modeled 1,024-MAC output
head. Its clock rate was not validated.

The revised design executes one Philox round per cycle in four parallel cores,
buffers four-word bursts, and feeds a dual-lane pipelined Gumbel transform. The
integrated RTL passes official known-answer vectors, exact Q5.10 reference
scores, partial vocabulary blocks, output backpressure stability, a sustained
111-score run with stalls, and completion checks.

| Open UltraScale+ mapping | DSP | LUT primitives | FF primitives | BRAM | Distributed RAM primitives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fully unrolled serial stream | 155 | 6,481 | 306 | 0 | 0 |
| One iterative Philox core | 16 | 712 | 326 | 0 | 0 |
| Integrated four-core stream | 66 | 4,136 | 1,880 | 2 | 14 |

The integrated mapping corresponds to 5.29% of K26 DSPs, 3.53% of K26 LUTs,
1.39% of K26 BRAMs, and 0.80% of K26 flip-flops before vendor placement. Its
analytical generation rate is 1.6 scores per cycle. For the 64 by 50,258 field,
the steady-state bound is 6.701 ms at 300 MHz, which clears the output head's
required average of 1.333 scores per cycle. This rate assumes the design reaches
300 MHz and is therefore not a measured FPGA latency.

### Direct noisy-argmax integration

The RNG stream now joins two signed Q5.10 model-score lanes directly into an
ordered noisy-argmax reducer. Ready/valid coupling prevents either producer
from advancing alone. The reducer excludes the mask token, handles partial
pairs, compares score ties by the lower token ID, and emits one compact
candidate per position.

A generic metadata-addressed reducer mapped to 4,057 LUTs and 3,304
flip-flops because it retained a 34-bit best score for every position. The
selected reducer exploits the position-ordered RNG stream and retains only the
current position plus one elastic candidate output. This reduced its standalone
mapping to 526 LUTs and 146 flip-flops.

The complete Philox, Gumbel, score join, and argmax boundary maps to 66 DSPs,
4,787 LUTs, 2,056 flip-flops, 2 BRAMs, and 14 distributed-RAM primitives. This
is 5.29% of K26 DSP capacity, 4.09% of LUT capacity, 0.88% of flip-flop
capacity, and 1.39% of BRAM capacity before placement. Compared with the RNG
stream alone, the ordered join and reducer add 651 LUTs and 176 flip-flops with
no additional DSP or BRAM primitives.

A dual elastic requantizer now implements the native kernel's signed Q12.20
scale, half-away-from-zero rounding, and signed bias addition. The standalone
requantizer maps to 8 DSPs, 330 LUTs, and 67 flip-flops. The full
accumulator-to-candidate boundary therefore maps to 74 DSPs, 5,117 LUTs, 2,123
flip-flops, 2 BRAMs, and 14 distributed-RAM primitives, or 5.93% of K26 DSPs
and 4.37% of K26 LUTs before placement.

RTL tests cover producer bubbles, candidate-output backpressure, exact Philox
and Gumbel scores, mask exclusion, tie behavior, a final two-token position,
partial pairs, and rejection of out-of-order position metadata. The remaining
boundary is the actual dot-product MAC producer for the two accumulator lanes.

### Decision

Accept the four-core iterative Philox farm, dual Q5.10 score interface, and
factorized 2 KiB Gumbel table as the open-mapped RNG design point. Retain the
eight-bit mantissa even though six bits passed the numerical gate because the
extra 1.5 KiB buys useful tail margin at negligible K26 capacity cost.

The result is not yet a hardware RNG quality claim. Full TestU01-style testing,
vendor place-and-route timing and a board measurement remain open. INT8
activation quality also remains unvalidated, so the FP16 activation path is
still the accepted system baseline.

Artifacts:

- `src/diffusion_accel/rng.py`
- `data/results/rng-gumbel-validation.json`
- `hls/rng/philox_rng.cpp`
- `hls/rng/gumbel_lut.hpp`
- `hls/integration/rng_output_head_test.cpp`
- `rtl/philox_gumbel/philox_gumbel_farm_stream.sv`
- `rtl/output_head/ordered_noisy_argmax_reducer.sv`
- `rtl/output_head/philox_noisy_argmax_stream.sv`
- `rtl/output_head/dual_requantizer_q20.sv`
- `rtl/output_head/philox_accumulator_argmax_stream.sv`
- `data/results/rng-hardware-architecture.json`
- `data/results/philox-gumbel-farm-stream-yosys-xcup.json`
- `data/results/philox-noisy-argmax-stream-yosys-xcup.json`
- `data/results/philox-accumulator-argmax-stream-yosys-xcup.json`

## 2026-08-16: Tiled output projection with fused candidate reduction

The output head now has a tiled software baseline and a hardware-shaped C++
kernel. Both process the vocabulary incrementally and retain only the best
noisy token ID per position. Neither path materializes a complete logit or
probability tensor.

Across 64 deterministic randomized trials, the tiled float64 path matched the
dense projection for all 448 position-level candidates. Repeating the same
comparison with IEEE FP16 also matched all 448 candidates. Tests cover tiles
that do not evenly divide the position or vocabulary dimensions.

The standalone C++ kernel uses 16-token vocabulary tiles, preloads the active
hidden states once, reads each INT8 output weight once per evaluation, reuses
that weight across positions, and performs a fused noisy argmax. Its native
macOS test matches a dense integer reference and checks partial tiles, exact
tie behavior, mask-token exclusion, and invalid controls.

| MDLM-sized local state, 64 positions and tile width 16 | Bytes |
| --- | ---: |
| INT8 hidden-state buffer | 49,152 |
| FP16 hidden-state buffer alternative | 98,304 |
| INT8 weight tile | 12,288 |
| INT32 accumulator tile | 4,096 |
| Candidate cache | 144 |
| Total with INT8 activations | 65,680 |
| Total with FP16 activations | 114,832 |

The full INT8 output matrix remains 38,598,144 bytes and must be streamed from
DDR under the current design. Tiling solves intermediate capacity and weight
reuse across positions; it does not eliminate output-weight bandwidth.

### Decision

Accept the tiled FP16 path as the functional baseline. Keep INT8 output weights
as an experimental option based on the measured low distribution drift. The
current C++ datapath also uses INT8 activations, which have not passed a
checkpoint quality gate, so it is an interface and resource prototype rather
than an accepted precision configuration.

Race-noise scores arrive through a stream in vocabulary-tile, position, lane
order. This prevents the noise values from becoming another DDR tensor and
creates a clean boundary for a counter-based RNG plus fixed-point Gumbel
transform. That stream producer is the next standalone correctness target.

Vitis HLS is still unavailable on this Mac. Initiation interval, achieved
clock, AXI behavior, resource use, and full-size synthesis remain open.

Artifacts:

- `src/diffusion_accel/output_head.py`
- `data/results/tiled-output-head-validation.json`
- `hls/output_head/output_head.cpp`
- `hls/output_head/output_head_test.cpp`
- `hls/output_head/run_hls.tcl`

## 2026-08-16: Output-head INT8 gate and K26 lane sweep

Whole-model naive INT8 was previously rejected because its mean and worst-case
top-1 agreement were too low. Quantizing only the output head isolates the
largest repeatedly streamed matrix while leaving the transformer unchanged.

Five real-checkpoint probes covered 160 masked positions across deterministic
seeds 0 through 4. Per-output-channel symmetric INT8 preserved all 160 FP32
top-1 choices. Mean normalized logit RMSE was 0.712%, and the worst seed was
0.739%. Mean total-variation distance between FP32 and INT8 token
distributions was 0.782%, with a 1.960% worst position. Mean KL divergence was
0.000222 nats. The quantized module set contained 38,794,752 parameters.

A full candidate-cached DDPM replay then compared identical random streams over
five 64-token generations. Three generations were token-identical. Aggregate
token agreement was 91.25%; seed 4 was the worst at 60.94%. All five active and
changed-token schedules, model-evaluation counts, and candidate-cache hit
counts remained exact. INT8 therefore preserves reveal-controller behavior but
does not provide pathwise-equivalent token sampling.

The output-head roofline then replayed the 38 measured model evaluations and
their 1,293 total active token positions. Each design point assumes one output
weight-matrix read per evaluation, one MAC per lane per cycle, 300 MHz, and
12.48 GB/s effective DDR bandwidth.

| Weight format | MAC lanes | Total head lower bound | DDR-bound evaluations |
| --- | ---: | ---: | ---: |
| FP16 | 512 | 364.9 ms | 13 / 38 |
| FP16 | 768 | 277.3 ms | 20 / 38 |
| FP16 | 1,024 | 245.7 ms | 28 / 38 |
| FP16 | 1,248 | 235.8 ms | 35 / 38 |
| INT8 | 512 | 335.5 ms | 7 / 38 |
| INT8 | 768 | 232.2 ms | 9 / 38 |
| INT8 | 1,024 | 182.4 ms | 13 / 38 |
| INT8 | 1,248 | 157.9 ms | 16 / 38 |

### Decision

For FP16 weights, 1,024 conceptual MAC lanes are a reasonable upper knee. The
last 224 lanes improve the output-head bound by only 4.0% while consuming the
remainder of the K26's 1,248-DSP budget under a one-lane-per-DSP assumption.
This is not a placement recommendation because real DSP packing, routing, and
transformer-engine sharing are unresolved.

Output-head INT8 remains the preferred bandwidth experiment because it halves
the dense projection matrix from 77.2 MB to 38.6 MB and the observed
distribution drift is small. It is not accepted as an exact replay mode. The
hardware plan should retain an FP16 fallback until a larger held-out
reconstruction and generation-quality evaluation accepts INT8 statistically.

Artifacts:

- `data/results/mdlm-int8-output-head-only-seed-0.json` through seed 4
- `data/results/mdlm-output-head-int8-generation.json`
- `data/results/output-head-lane-sweep.json`

## 2026-08-16: Fused streaming candidate producer

MDLM's exponential-race categorical sampler can select a candidate without
materializing softmax probabilities. The softmax normalization is shared by
every vocabulary entry, so it cancels from the argmax. A chunked implementation
retains only the best noisy score and token ID for each active position.

The float64 validation processed the 50,258-entry vocabulary in 512-entry
chunks. All 512 candidates matched the probability-tensor reference exactly.
This is pathwise agreement with identical logits and uniform random values, not
only agreement in distribution.

| 64-position MDLM producer state | Result |
| --- | ---: |
| Materialized FP16 probability tensor | 6,433,024 bytes |
| External write and read round trip | 12,866,048 bytes |
| Streaming score and token accumulators | 392 bytes |
| Final candidate cache | 144 bytes |
| Round trip to candidate-cache ratio | 89,348x |

For the assumed 1,024 MAC lanes at 300 MHz, FP16 output weights, and 12.48 GB/s
effective DDR bandwidth, one dense 64-position MDLM output projection has an
8.041 ms compute lower bound. Fusion reduces its memory lower bound from 7.224
ms to 6.193 ms, but both variants remain compute-bound at 8.041 ms. The
standalone roofline therefore predicts no latency gain from fusion alone.

The complete measured-mask trace gives a less idealized result because early
logit tensors exceed the modeled on-chip capacity. With the existing masked
output head and INT8 weight-traffic scenario, fusion produced:

| KV260-class analytical replay, 38 model evaluations | Materialized logits | Fused candidates | Change |
| --- | ---: | ---: | ---: |
| Total modeled latency | 947.65 ms | 925.33 ms | -2.35% |
| HBM traffic | 7.301 GB | 6.804 GB | -6.81% |
| Peak SRAM | 2.868 MiB | 0.375 MiB | -86.9% |
| Total modeled compute work | unchanged | unchanged | 0% |

### Decision

Keep the fused producer in the architecture because it removes a large
intermediate tensor and makes the 144-byte candidate cache physically useful.
Do not pitch it as the primary speedup. The dominant output-head work remains
2.470 billion MACs and 77.2 MB of FP16 weights for the dense 64-position case.
The next design iteration should tile the vocabulary projection around the
available DSP lanes and DDR bandwidth, then exploit the falling active-position
schedule.

The software proof uses float64 logarithms. A counter-based RNG, logarithm or
equivalent exponential-race transform, fixed-point score format, and
distribution-error gate remain necessary before this producer becomes RTL.

Artifacts:

- `src/diffusion_accel/candidate_producer.py`
- `data/results/streaming-candidate-producer.json`
- `data/traces/mdlm-owt-ddpm-cache-seed0-masked-head-int8-fused-candidate.jsonl`
- `data/results/kv260-ddpm64-cache-int8-masked-head-fused-candidate.json`

## 2026-08-16: Streaming RTL and K26 capacity comparison

The compact candidate cache now has a cycle-accurate SystemVerilog reveal
controller. The ready/valid interface accepts one canvas position per cycle,
commits eligible candidate tokens, reports the changed count, and requests a
global candidate invalidation whenever any input token changes.

Icarus Verilog passes boundary checks for zero and unit reveal probability,
backpressure, partial commits, invalid commands, and 1,024 deterministic
randomized positions. This validates RTL behavior, including stalled streams.

| Derived 64-position, 23-hit cycle model | Result |
| --- | ---: |
| Cycles per cache-hit transition at assumed II=1 | 66 |
| Total cycles | 1,518 |
| Latency at an assumed 300 MHz | 5.06 us |
| Fraction of 776.8 ms measured model-forward time | 0.000651% |
| Stream traffic per hit | 680 bytes |
| Probability tensor to stream-traffic ratio | 9,460x |

Yosys 0.68 maps the controller to UltraScale+ primitives as 97 LUT
primitives, 33 flip-flop primitives, 14 carry elements, and no BRAM, UltraRAM,
or DSP primitives. Against the K26 capacities documented in
[AMD DS987 revision 1.6](https://docs.amd.com/r/en-US/ds987-k26-som/Programmable-Logic),
the raw primitive sums correspond to 0.0828% of CLB LUTs and 0.0141% of CLB
flip-flops. Vendor packing may change the final occupied counts.

### Decision

The reveal controller is small enough that it is not the chip-cost or latency
bottleneck worth optimizing next. The next useful accelerator boundary is the
producer side: generate candidate IDs without materializing or retaining the
full position-by-vocabulary probability tensor, then stream those IDs directly
into the reveal state.

These results do not establish a 300 MHz clock or final K26 utilization. Yosys
performed open-source technology mapping only. Vendor synthesis, placement,
routing, timing closure, AXI integration, and measured board power remain open.

Artifacts:

- `rtl/candidate_reveal/candidate_reveal_stream.sv`
- `rtl/candidate_reveal/tb_candidate_reveal_stream.sv`
- `data/results/candidate-reveal-kernel-model.json`
- `data/results/candidate-reveal-yosys-xcup.json`

## 2026-08-15: Distribution-equivalent candidate cache

The exact no-change probability cache originally retained one probability for
every position and vocabulary token. The replacement samples one candidate
token per masked position immediately after a model evaluation, then stores
only that candidate while scalar reveal decisions are retried. Any changed
input invalidates all remaining candidates.

The factorization follows directly from the normalized DDPM transition:

- stay masked with probability `move_s / move_t`;
- reveal with probability `(move_t - move_s) / move_t`;
- conditional on reveal, sample a token from the unchanged model distribution.

| Validation gate | Result |
| --- | ---: |
| Maximum analytic probability error | `1.11e-16` |
| Maximum analytic row-sum error | `7.77e-16` |
| Monte Carlo trials | 500,000 |
| Total-variation distance | `0.003102` |
| Allowed total-variation distance | `0.01` |
| Overall gate | Pass |

| 64-position MDLM cache state | Bytes |
| --- | ---: |
| FP16 probability tensor, 50,258 vocabulary entries | 6,433,024 |
| Packed 16-bit candidate IDs | 128 |
| Active and valid bitmaps | 16 |
| Total candidate state | 144 |

The compact representation is 44,674 times smaller. It fits trivially in local
FPGA SRAM or registers and removes the need to preserve a 6.135 MiB probability
tensor across no-change transitions.

A real MPS trace using the pinned MDLM checkpoint completed all 64 transitions
with 41 model evaluations and 23 candidate-cache hits. Measured model-forward
time was 776.8 ms. The generated continuation was readable, although this
single stochastic output is not a quality evaluation. It cannot be compared
token for token with the original cache under one seed because factorized
sampling consumes a different random-number stream.

### Decision

Candidate caching passes the pre-RTL correctness and capacity gate. It is a
better standalone FPGA control-kernel target than full probability storage.
That standalone interface is now implemented in portable C++ for up to 256
positions. It accepts packed 16-bit candidate IDs, active and valid bitmaps,
one 32-bit random word per position, and a Q0.32 reveal threshold. It commits
eligible candidates at an intended pipeline initiation interval of one, reports
the changed count, and globally invalidates remaining candidates after any
commit.

The macOS testbench compiles with warnings treated as errors and passes five
boundary suites: zero reveal probability, probability one including the maximum
random word, partial commits across bitmap words, invalid candidate suppression,
and invalid control rejection. A parameterized Vitis HLS script is included,
but Vitis is not installed on this Mac, so resource and timing claims remain
open.

Artifacts:

- `data/results/candidate-cache-equivalence.json`
- `data/traces/mdlm-owt-ddpm-candidate-cache-seed0.jsonl`
- `hls/candidate_sampler/candidate_sampler.cpp`
- `hls/candidate_sampler/candidate_sampler_test.cpp`
- `hls/candidate_sampler/run_hls.tcl`

## 2026-08-15: Bounded session cache and layer-local DDR trace

The session experiment now includes a real cache lifecycle rather than assuming
that every conversation remains resident. The manager tracks transcript state,
exact K/V hits, expiry, LRU eviction, oversized entries, finalization cost, and
incremental DDR writes. It also lowers each request into a hardware-neutral
K/V-only trace with active answer K/V split by layer.

The MDLM-shaped workload replayed four interleaved two-turn conversations. Each
answer used 16 model evaluations, matching the measured two-turn experiment.

| MDLM model-work comparison | Token positions | Reduction |
| --- | ---: | ---: |
| Full block recomputation during denoising | 12,288 | baseline |
| Request-local prefix cache | 2,688 | 78.1% |
| Cross-request retention, ready for another turn | 2,496 | 79.7% |
| Cross-request retention, no unused terminal finalization | 2,432 | 80.2% |

Cross-request retention therefore adds a 7.1% reduction beyond a request-local
prefix cache when every final answer is prepared for another turn. With final
answers that received no observed follow-up left unfinalized, it adds 9.5%.
This is substantially smaller than the comparison against full recomputation.

A deliberately constrained 24 MiB cache served all four follow-ups as hits,
peaked at 23.625 MiB, and performed one LRU eviction only after that session's
last request. With FP32 MDLM weights and a 512 MiB runtime reserve, the simple
4 GiB board budget leaves 2,937.0 MiB for session K/V, enough for 372 complete
112-token caches under the stated accounting. Real deployment capacity will be
lower after allocator, alignment, and additional runtime buffers.

| K/V-only KV260-class replay | All DDR | Layer-local active K/V in SRAM |
| --- | ---: | ---: |
| MDLM HBM traffic over eight requests | 1,009.8 MB | 858.8 MB |
| Transfer-only lower bound | 80.9 ms | 68.8 ms |
| Peak active K/V SRAM | 0 | 96 KiB |

Session K/V reads still contributed 825.8 MB and remained in external DDR.
Moving the layer-local active K/V scratchpad on chip removed 151.0 MB, or 15.0%
of modeled HBM traffic.

### 8B feasibility result

An 8B full-MHA design uses 512 KiB of FP16 K/V per cached token. Raw 4-bit
weights require 4.0 billion bytes. Those weights plus a 512 MiB runtime reserve
exceed the KV260's 4 GiB DDR by about 231 MiB before session K/V, packing
metadata, or alignment. The lifecycle model therefore gives the 8B KV260 case
zero cache capacity and rejects both attempted cache entries.

A hypothetical 8 GiB, eight-KV-head GQA design with a dedicated 256 MiB session
budget needs 68 MiB for the modeled 544-token conversation. It hits on the
follow-up, but eager finalization for another possible request erases the token
work saved by cross-request retention. If the final answer is not finalized,
the observed incremental reduction is only 1.9%. Its layer-local active block
peaks at 1 MiB and fits the K26 SRAM budget, reducing K/V-only HBM traffic from
2.139 GB to 1.334 GB. This remains hypothetical because vanilla LLaDA is not
block causal and the trace excludes weights and model compute.

### Decision

The hardware priority is now clear. Request-local prefix isolation and a
layer-local active K/V scratchpad are first-order optimizations. Cross-request
retention is useful but should be opportunistic, especially when final answer
K/V requires an extra pass. An 8B LLaDA-class model is not a credible KV260
resident design under the current 4 GiB budget. MDLM-sized silicon work remains
the practical path for a real FPGA demo.

Primary artifacts:

- `data/results/session-lifecycle-mdlm-24mib.json`
- `data/traces/session-lifecycle-mdlm-24mib.jsonl`
- `data/results/session-lifecycle-mdlm-24mib-canvas-sram.json`
- `data/results/session-lifecycle-8b-kv260.json`
- `data/results/session-lifecycle-8b-gqa-256mib.json`

## 2026-08-15: Exact two-turn session K/V retention

The real MDLM-OWT checkpoint completed three two-turn continuation sessions on
MPS. Each session used 64 initial tokens, a generated 16-token first answer, a
fixed 16-token follow-up block, and a generated 16-token second answer. The
implementation finalized completed answers into per-layer K/V, appended the
follow-up block, and reused all completed blocks while denoising the next
answer.

The cached path was compared with full block-causal recomputation using the
same random seeds:

| Correctness and capacity | Result |
| --- | ---: |
| Exact first answers | 3 of 3 |
| Exact second answers | 3 of 3 |
| Identical first-answer transition schedules | 3 of 3 |
| Identical second-answer transition schedules | 3 of 3 |
| Minimum second-answer token agreement | 100% |
| Final cached session length | 112 tokens |
| Final FP32 session K/V | 7.875 MiB |

| Median MPS cost around the second request | Time | Reduction vs recompute |
| --- | ---: | ---: |
| Full second-answer recomputation | 729.5 ms | baseline |
| Cached second-answer model forwards | 279.9 ms | 61.6% |
| Cached follow-up prefill plus second answer | 292.5 ms | 59.9% |
| Above plus first-answer finalization | 305.5 ms | 58.1% |
| Above plus second-answer finalization | 322.8 ms | 55.8% |

Median first-answer finalization took 12.0 ms, follow-up materialization took
12.7 ms, and second-answer finalization took 14.6 ms. The finalization passes
skip the vocabulary output head because only final per-layer K/V is retained.

### Decision

For this prefix-safe model contract, retaining completed-turn K/V across a
near-term follow-up is worthwhile. Its measured benefit remains material after
all exact boundary costs are included. Production should use a session TTL and
LRU eviction because the cache grows linearly with transcript length. Vanilla
full-attention LLaDA remains a recompute baseline because later blocks can
change earlier hidden states.

This experiment validates execution and measured latency, not chat quality.
MDLM-OWT is a small text-continuation checkpoint and some visible generations
remain poor. The next hardware step is to expose this session lifecycle in the
trace and measure external-DDR traffic under a bounded cache manager.

Reproduce with:

```bash
.venv/bin/diffusion-accel run-mdlm-two-turn \
  --device auto \
  --prefix-tokens 64 \
  --first-answer-tokens 16 \
  --followup-tokens 16 \
  --second-answer-tokens 16 \
  --samples 3 \
  --steps 64 \
  --seed 0 \
  --local-files-only \
  --out data/results/mdlm-two-turn-wikitext2.json
```

## 2026-08-15: Conditioned DDPM with retained prefix K/V

The real MDLM-OWT checkpoint generated five 32-token continuations conditioned
on 64-token WikiText-2 prefixes. Each sample ran 64 requested ancestral DDPM
transitions with exact no-change probability caching through three paths:

1. Original full bidirectional attention.
2. Prefix-isolated attention with full prefix recomputation.
3. Prefix-isolated attention using retained per-layer prefix K/V.

The isolated recompute and retained-cache paths reset to identical random seeds.

| End-to-end cache result | Value |
| --- | ---: |
| Exact cached sequences | 5 of 5 |
| Identical DDPM transition schedules | 5 of 5 |
| Minimum token agreement | 100% |
| Median model evaluations | 26 |
| Median exact probability-cache hits | 38 |
| FP32 prefix K/V | 4.5 MiB |

| MPS timing | Recompute | Retained cache | Reduction |
| --- | ---: | ---: | ---: |
| Median model-forward time per continuation | 669.5 ms | 323.8 ms | 51.6% |
| Median sampling wall time | 745.9 ms | 403.1 ms | 46.0% |

The one-time 64-token prefix prefill took a median 22.7 ms. It is small relative
to the repeated denoising work and is amortized within the first continuation.

Original full attention and prefix isolation agreed on 42.5% of generated
tokens. Several continuations were locally coherent, but some remained
repetitive or factually inconsistent. Exact target-token agreement was 1.875%
for original attention and 0.625% for prefix isolation over only 160 generated
tokens. This small stochastic sample is not sufficient for a generation-quality
claim.

Terminal K/V was invalid after all five generations because the completed
suffix differed from the input to the last model evaluation. A suffix
finalization forward is therefore necessary before the new answer can be added
to a multi-turn session cache.

### Decision

Retained prefix K/V passes the end-to-end execution-correctness gate and gives a
material measured speedup on the Mac. Generation quality remains the open model
question. The next implementation should materialize final answer K/V and run a
real two-turn cached conversation. A larger continuation evaluation or a small
prefix-isolation fine-tune is still required before presenting the attention
change as a quality-preserving model improvement.

Reproduce with:

```bash
.venv/bin/diffusion-accel generate-mdlm-conditioned \
  --device auto \
  --prefix-tokens 64 \
  --suffix-tokens 32 \
  --samples 5 \
  --steps 64 \
  --seed 0 \
  --local-files-only \
  --out data/results/mdlm-conditioned-ddpm-wikitext2.json
```

## 2026-08-15: Prefix-isolation held-out quality gate

The prefix-isolated execution contract was evaluated on the pinned WikiText-2
raw test split using the real MDLM-OWT checkpoint on MPS. Thirty-two
deterministic held-out windows were evaluated at nine combinations of prefix
length 16, 64, or 128 and suffix length 16, 32, or 64. This produced 10,752
scored token decisions.

| Aggregate quality metric | Original full attention | Prefix isolated | Change |
| --- | ---: | ---: | ---: |
| One-step true-token top-1 accuracy | 4.157% | 4.055% | 2.46% relative loss |
| Mean negative log likelihood | 7.376 | 7.331 | 0.61% improvement |

The predefined quality gate allowed at most 5% relative accuracy loss and 5%
negative-log-likelihood increase. Both checks passed.

| Cache correctness metric | Result |
| --- | ---: |
| Minimum cached-versus-recomputed top-1 agreement | 100% |
| Maximum normalized logit RMSE | `1.93e-6` |
| Correctness gate | Pass |

The steady-state cached path became more valuable as the reusable prefix grew:

| Prefix length | FP32 prefix K/V | MPS latency reduction range |
| ---: | ---: | ---: |
| 16 | 1.125 MiB | 41.6% to 45.3% |
| 64 | 4.5 MiB | 43.8% to 56.8% |
| 128 | 9.0 MiB | 60.6% to 65.3% |

The timing excludes one-time prefix prefill and includes only steady-state
active-suffix execution versus full prefix-isolated recomputation. The result
file includes two held-out one-step completion previews. These previews are
repetitive, and the low absolute top-1 accuracy confirms that this is a narrow
reconstruction test rather than an end-to-end generation-quality claim.

### Decision

The prefix-isolated cache policy passes its first quality and correctness gate.
The next step is to integrate retained prefix K/V into conditioned DDPM sampling
and compare full generated continuations, model evaluations, latency, and
traffic. FPGA work remains gated on that end-to-end result and robust
quantization.

Reproduce with:

```bash
.venv/bin/diffusion-accel evaluate-mdlm-prefix-isolation \
  --device auto \
  --prefix-lengths 16 64 128 \
  --suffix-lengths 16 32 64 \
  --samples 32 \
  --seed 0 \
  --local-files-only \
  --out data/results/mdlm-prefix-quality-wikitext2.json
```

## 2026-08-15: Real MDLM prefix K/V correctness experiment

The official pinned MDLM-OWT checkpoint was run on MPS with a 15-token fixed
prefix and a 32-token masked suffix. One suffix token was replaced between two
forwards, and every layer's prefix key and value projection was captured.

With the checkpoint's original full bidirectional attention:

- Layer 0 prefix K/V remained exactly equal, as expected from unchanged prefix
  embeddings.
- Prefix K/V changed in every layer from layer 1 onward for all five seeds.
- For seed 0, layer 8 value K/V reached normalized RMSE `0.342`.
- Mean suffix top-1 agreement between the original and edited canvas was only
  20.0% across five seeds.

This directly falsifies exact vanilla prefix-K/V reuse after an active-suffix
edit.

The same weights were then evaluated under an experimental prefix-isolated
attention mask. Prefix queries could attend only to the fixed prefix, while
active suffix queries could attend to the prefix and suffix. A manual cached
path evaluated only the suffix using retained per-layer prefix K/V.

| Metric across five seeds | Result |
| --- | ---: |
| Prefix K/V exact after suffix edit | 5 of 5 |
| Cached versus recomputed suffix top-1 agreement | 100% |
| Maximum normalized suffix-logit RMSE | `9.30e-7` |
| Median cached-suffix MPS latency | 9.36 ms |
| Median full prefix-isolated MPS latency | 17.03 ms |
| Preliminary steady-state latency reduction | 45.0% |

Each per-seed latency is the median of five forwards after warming that MPS
path. The cached timing excludes the one-time prefix prefill, which is amortized
over later denoising evaluations or follow-up requests.

This is a cache-mechanics and numerical-equivalence result. MDLM-OWT was not
trained with prefix isolation, and its one-step token previews are not a chat
quality evaluation. Model-quality validation or training with the intended
mask remains required before calling this a serving-model improvement.

Reproduce one seed with:

```bash
.venv/bin/diffusion-accel analyze-mdlm-prefix-drift \
  --device auto \
  --suffix-tokens 32 \
  --changed-suffix-tokens 1 \
  --timing-repeats 5 \
  --seed 0 \
  --local-files-only \
  --out data/results/mdlm-prefix-drift-seed0.json
```

## 2026-08-15: Multi-turn session-cache experiment

The executable `analyze-session-cache` model now compares exact full-attention
recomputation with exact prefix-isolated K/V retention. For two turns of
`128:256` and `32:128` user/answer tokens:

| Design point | Conditioning token-position work | K/V after turn two | KV260 capacity result |
| --- | ---: | ---: | --- |
| Full-attention recompute | 544 | 0 retained | Exact baseline |
| Prefix cache, answer finalization required | 416 | 272 MiB | 8B footprint does not fit |
| Prefix cache, terminal K/V already valid | 160 | 272 MiB | 8B footprint does not fit |
| Prefix cache, 8 K/V heads | 416 | 68 MiB | 8B footprint still does not fit |
| 169.6M MDLM-shape, INT8 weights | 416 | 19.125 MiB | Fits analytical budget |

The 8B capacity check assumes the optimistic lower bound of raw INT4 weights:
4.0 GB decimal (3.725 GiB). Adding only a 512 MiB runtime reserve already
exceeds 4 GiB before session K/V, quantization scales, alignment, activations,
or DMA buffers. Reducing K/V with grouped-query attention cannot fix the weight
capacity failure.

The first retained answer needs a 256-token finalization pass if terminal K/V is
not already valid. It saves only 128 token positions over the observed two-turn
session, a 23.5% reduction; if terminal K/V is available, it saves 384 positions,
a 70.6% reduction. The hardware/software interface should therefore expose a
`terminal_kv_valid` condition rather than assuming completed-answer K/V is free.

Reproduce with:

```bash
diffusion-accel analyze-session-cache \
  --turn 128:256 \
  --turn 32:128 \
  --out data/results/session-cache-8b-kv260.json
```

## 2026-08-15: Real MDLM trace on Apple Silicon

### Run

- Host: Apple M5 Pro with 24 GB unified memory.
- Device: PyTorch MPS.
- Checkpoint: `kuleshov-group/mdlm-owt` at pinned revision
  `d0958fa851335ece6c15260ce0025f030673c0fb`.
- Loaded model: 169,627,218 parameters and 678,508,872 bytes in FP32.
- Smoke workload: one 64-token canvas, four confidence-unmask steps, 16
  positions committed per step, after one warmup forward.
- Measured checkpoint-forward time: 101.90 ms total. Individual steps measured
  33.50, 25.14, 20.79, and 22.48 ms.

The generated token IDs are not a text-quality result. This run intentionally
uses a deterministic confidence-unmask smoke sampler to validate real execution,
mask evolution, and trace capture.

### Analytical replay

The measured mask schedule was lowered into 56 operations: token embedding,
12 bidirectional transformer blocks, full vocabulary projection, and confidence
selection for each of four steps.

| Design point | Modeled latency | HBM traffic | FLOPs | Change vs. baseline |
| --- | ---: | ---: | ---: | ---: |
| All HBM, full head | 10.268 ms | 2.875 GB | 63.885 GFLOP | baseline |
| 16 MiB canvas SRAM, full head | 9.860 ms | 2.752 GB | 63.885 GFLOP | 3.98% latency |
| All HBM, masked-only head | 10.131 ms | 2.837 GB | 56.474 GFLOP | 1.34% latency |
| Canvas SRAM + masked-only head | 9.848 ms | 2.752 GB | 56.474 GFLOP | 4.10% latency |

These latency values describe the repository's generic accelerator
configuration, not the Mac and not a KV260. The Mac timing only establishes that
the real model path works and provides the mask schedule.

### Decision

The first result weakens a broad claim that modest canvas SRAM alone will be a
large win for this model. It also shows that masked-only vocabulary projection
cuts 11.6% of arithmetic in this four-step schedule but gives little latency
benefit while full model weights are streamed every step.

The next experiment should therefore combine:

1. INT8/INT4 weight traffic and compute assumptions.
2. A KV260-class DDR bandwidth and on-chip SRAM configuration.
3. A sweep over canvas length and denoising-step count.
4. Accuracy checks before treating quantization as a valid implementation.

No FPGA purchase is justified by this result alone; the quantized KV260 sweep
is the next purchase gate.

## 2026-08-15: KV260 INT8 purchase gate

### Documented board point and modeled result

The KV260 configuration uses AMD's documented 1.2 peak INT8 TOPS, 4 GB of
64-bit DDR4 at 2400 Mb/s (19.2 GB/s theoretical), and 26.6 Mb of aggregate
on-chip memory. Efficiency and on-chip bandwidth remain explicit assumptions.

| KV260-class design point | Modeled latency | External-memory traffic | Status |
| --- | ---: | ---: | --- |
| FP32 weights, full head | 230.38 ms | 2.875 GB | reference only |
| INT8 weights, full head | 113.30 ms | 0.840 GB | accuracy unproven |
| INT8 + masked head + canvas SRAM | 101.95 ms | 0.781 GB | accuracy unproven |

The combined point is 55.7% faster than the FP32-weight reference and is close
to balanced between modeled compute and DDR limits. This is strong enough to
justify continued FPGA-oriented design work, but not yet a board purchase.

### Quantization check

Per-output-channel symmetric INT8 fake quantization was compared with FP32 on
five deterministic, half-masked 64-token canvases. It quantized 169,454,592
weight parameters.

- Mean top-1 agreement: 87.5%.
- Range of top-1 agreement: 65.6% to 100%.
- Mean logit cosine similarity: 0.946.
- Mean normalized RMSE: 0.282.

This naive post-training quantizer fails the robustness gate. Preserving the
FP32 output head did not improve the seed-0 result, indicating that error is
already accumulating in the transformer/embedding path.

### Purchase decision

**Hold the KV260 purchase for now.** The architecture is plausible and the
board is appropriately sized, but the next milestone must demonstrate one of:

1. Calibration-aware INT8 with at least 98% masked-position top-1 agreement
   across a broader validation set.
2. A mixed-precision design whose lower traffic still gives at least a 25%
   modeled KV260 latency win.
3. A synthesized standalone kernel that fits the K26 resource envelope and
   represents at least 20% of end-to-end work.

Passing any two of these three is the board-purchase trigger.

## 2026-08-15: Official DDPM output and cross-step caching

### Human-visible output

The macOS path now implements the upstream SUBS parameterization, log-linear
noise schedule, ancestral categorical update, final mask removal, and GPT-2
decoding. A 64-token, 64-transition seed-0 sample was:

> `<|endoftext|> phenomenon, where there’s such a suspense about history based`
> `on the last several generations of very few people from the world. I think`
> `of all kinds of fears. Now, terrorism. You have seen reality already lost,`
> `Daesh. IS always mentioned. Of course there are ISIS… in and around terror. I`

Other seeds were non-collapsed but sometimes repetitive, which is a model and
sampling-quality limitation rather than a trace failure. All runs fully removed
the mask token.

### Exact probability-cache reuse

For this checkpoint, time conditioning is disabled. When a transition changes
no tokens, the next transition can reuse the same model probability tensor.
This preserves the sampled output exactly for the same random seed.

| Run | Transitions | Model evaluations | Cache hits | Measured MPS forward time |
| --- | ---: | ---: | ---: | ---: |
| DDPM, 64 steps | 64 | 64 | 0 | 1,253 ms |
| DDPM-cache, 64 steps | 64 | 38 | 26 | 704 ms |
| DDPM, 256 requested steps | 252 | 252 | 0 | 4,764 ms |
| DDPM-cache, same run | 252 | 57 | 195 | 1,077 ms |

The cache reduced measured model-forward time by 43.9% in the 64-step run and
by 77.4% (4.42x speedup) in the longer run, with byte-identical decoded text for
each matching seed.

### KV260 replay of the realistic 64-step schedule

| Design point | Modeled latency | External-memory traffic |
| --- | ---: | ---: |
| Uncached DDPM, FP32 weights | 3.686 s | 46.003 GB |
| DDPM-cache, FP32 weights | 2.189 s | 27.314 GB |
| Cache + INT8 + masked head + SRAM | 0.948 s | 7.301 GB |

Caching is an accuracy-safe, diffusion-specific win and should be part of the
software baseline before RTL. The final INT8 row remains conditional on solving
the quantization-accuracy failure documented above.
