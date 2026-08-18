# Modeling assumptions

Every numerical assumption must be classified as measured, documented,
derived, or assumed. The synthetic milestone contains assumed and derived
values only.

## Two-turn session cache experiment

- **Inputs:** three deterministic WikiText-2 test windows, each with 64 initial
  tokens, a generated 16-token first answer, 16 fixed follow-up tokens, and a
  generated 16-token second answer.
- **Attention contract:** completed blocks are isolated from all later blocks.
  The active answer remains bidirectional within its own block and can attend
  to every completed block.
- **Sampler:** each answer uses 64 requested ancestral DDPM transitions with
  exact no-change probability caching.
- **Replay:** cached and fully recomputed paths reset to the same per-answer
  seed. Exactness requires identical generated tokens and transition schedules.
- **Finalization:** after an answer completes, a separate projection-only pass
  materializes its final per-layer K/V. The output head is not evaluated because
  its probabilities are not part of the session cache.
- **Cross-request cost:** the comparison includes first-answer finalization,
  follow-up materialization, and second-answer model forwards. A separate
  ready-for-third-turn measurement also includes second-answer finalization.
- **Scope:** the experiment validates execution and latency for a block-causal
  adaptation of MDLM-OWT. It is a text-continuation proxy, not a chat-quality
  evaluation and not evidence that vanilla full-attention LLaDA supports exact
  prefix caching.

## Bounded session lifecycle trace

- **Cache policy:** exact completed-block K/V is retained in external DDR under
  a configurable inactivity TTL and least-recently-used capacity policy.
- **Correctness after eviction:** transcript token counts remain available, so
  a cache miss recomputes completed context rather than using stale K/V.
- **Model work units:** token-position counts compare full context recomputation
  on every denoising evaluation, request-local context prefill, and cross-request
  retention. They are architecture work counts, not latency predictions.
- **DDR traffic scope:** the trace includes context K/V reads, context K/V
  writes, completed-answer finalization, and active answer K/V writes. It omits
  model weights, hidden activations, logits, and control traffic.
- **Layer granularity:** active answer K/V is emitted one layer at a time. The
  all-layer session cache remains in DDR, while a layer-local active block may
  fit in FPGA SRAM.
- **MDLM workload:** four interleaved two-turn sessions use 64 initial tokens,
  16-token answers and follow-ups, and 16 model evaluations per answer. The 16
  evaluations come from the measured two-turn MDLM experiment.
- **8B workload:** one two-turn session uses 128 and 32 user tokens, 256 and 128
  answer tokens, and 16 evaluations per answer. This is an assumed design point,
  not a measured LLaDA trace.
- **8B attention:** the GQA and full-MHA results assume a hypothetical
  completed-block-causal model. Vanilla LLaDA full attention does not satisfy
  the exact K/V reuse contract.
- **Traffic latency:** simulator timing for these traces is a K/V-transfer-only
  lower bound because every emitted operation has zero model FLOPs.

## Candidate-token cache

- **Factorization:** for one masked position, the DDPM transition first chooses
  whether to stay masked or reveal, then samples the revealed token from the
  unchanged model distribution. These choices are independent while the model
  input remains unchanged.
- **Correctness class:** sampling one candidate at a model evaluation and
  retaining it across no-change transitions is distribution-equivalent, not
  pathwise identical under the same stateful random-number stream.
- **Invalidation:** any changed input token invalidates every remaining
  candidate because bidirectional model probabilities can change globally.
- **Analytic validation:** 64 transitions, 64 positions, and 257 synthetic
  token probabilities per position are checked in float64 against the original
  normalized transition distribution.
- **Statistical validation:** 500,000 categorical trials over 16 token classes
  use a maximum accepted total-variation distance of 0.01.
- **Storage:** candidate IDs are packed to the minimum whole-byte width for the
  vocabulary. The estimate also includes separate active and valid bitmaps. It
  omits RNG state because the engine can consume a shared counter-based stream.
- **Trace timing:** real candidate-cache traces measure model forwards but still
  omit scalar reveal-sampling overhead on cache-hit transitions.
- **Standalone kernel:** the C++ kernel is functionally validated with its
  native macOS testbench. The corresponding streaming SystemVerilog is
  functionally validated with Icarus Verilog, including ready/valid stalls and
  1,024 deterministic randomized positions.
- **Cycle model:** 66 cycles per 64-position cache hit assumes one command
  cycle, one accepted position per cycle, and one downstream invalidation
  cycle. The reported 300 MHz latency is derived, not timing-validated.
- **Open-source mapping:** Yosys 0.68 technology mapping reports raw UltraScale+
  primitives. The LUT sum does not account for final vendor packing, placement,
  or routing and must not be presented as achieved K26 utilization.
- **Device comparison:** K26 capacities come from
  [AMD DS987 revision 1.6](https://docs.amd.com/r/en-US/ds987-k26-som/Programmable-Logic):
  117,120 CLB LUTs, 234,240 CLB flip-flops, 144 BRAM blocks, 64 UltraRAM
  blocks, and 1,248 DSP slices.
- **Open hardware gates:** FPGA timing, achieved initiation interval, AXI
  throughput, final LUT/FF/BRAM use, and power remain unmeasured until vendor
  synthesis and a board run.
- **Token width:** the first kernel uses 16-bit token IDs and supports up to 256
  positions. A model with more than 65,536 vocabulary entries requires a wider
  interface or remapping.
- **Random input:** one external 32-bit random word is supplied per position.
  Statistical quality depends on the future counter-based RNG implementation,
  which is outside the current kernel.

## Streaming candidate producer

- **Algebraic transform:** the producer maximizes logit minus the logarithm of
  MDLM's exponential noise. This is mathematically the same argmax as dividing
  normalized probabilities by the same noise because the softmax denominator
  is shared across the vocabulary.
- **Pathwise validation:** 512 float64 samples over 50,258 vocabulary entries
  matched the materialized probability reference using identical logits and
  uniforms. This does not validate a reduced-precision hardware transform.
- **Accumulator estimate:** each of 64 positions retains one 32-bit score, one
  16-bit token ID, and one valid bit. Packing and implementation overhead are
  omitted.
- **Output-head model:** 1,024 MAC lanes, 300 MHz, FP16 weights and activations,
  19.2 GB/s theoretical DDR bandwidth, and 65% DDR efficiency are assumed.
  Weights are read once per evaluation and reused across all active positions.
- **Trace transform:** projection and categorical FLOPs remain unchanged. Only
  the external logit write and later read become compact candidate-state
  accesses, isolating the memory effect rather than claiming compute removal.
- **Open correctness gate:** counter-based random generation, the exponential
  noise transform, fixed-point score precision, and statistical distribution
  error are not yet implemented or synthesized.

## Output-head lane sweep

- **Measured schedule:** 38 model evaluations and 1,293 active token positions
  come from `mdlm-owt-ddpm-cache-seed0.jsonl`.
- **Weight reuse:** the output matrix is read once from DDR per model evaluation
  and reused across every active position in that evaluation. No cross-request
  or cross-evaluation weight residency is assumed.
- **Lane model:** one lane performs one MAC per 300 MHz cycle. The sweep does
  not claim a physical mapping from lanes to DSP slices or account for
  reduction-tree, routing, control, RNG, or buffering cost.
- **Roofline:** per-evaluation latency is the larger of projection compute time
  and output-weight DDR time. Hidden-state and result traffic are omitted from
  this isolated head comparison because output weights dominate them after
  candidate fusion.
- **INT8 checkpoint gate:** five seeds compare FP32 and fake-quantized output
  head logits over 32 masked positions each. Top-1 agreement and logit error do
  not substitute for full DDPM generation quality.
- **INT8 generation gate:** five unprompted 64-token candidate-cached DDPM
  samples reset FP32 and INT8 to identical random seeds. Exact token replay is
  deliberately stricter than distributional quality equivalence.
- **INT8 status:** one-step distribution drift is small, but two of five full
  generations diverged. INT8 remains experimental and requires an FP16
  fallback until a larger held-out statistical quality gate passes.

## Tiled output-head kernel

- **Functional baseline:** dense and tiled projections use identical hidden
  states, weights, bias, and additive race-noise scores. Exact candidate IDs
  are required in both float64 and IEEE FP16 software tests.
- **Weight layout:** vocabulary-major contiguous rows are streamed in
  16-token tiles. Each weight is read once per evaluation and reused across all
  active positions buffered on chip.
- **Local-state accounting:** activation and weight buffers, an INT32
  accumulator tile, and the 144-byte candidate cache are included. Control,
  alignment, double buffering, RNG state, and interconnect FIFOs are omitted.
- **Integer prototype:** the native C++ kernel uses INT8 activations and
  weights. Only output-weight INT8 has checkpoint evidence, so the complete
  integer datapath is not an accepted accuracy mode.
- **Noise interface:** signed race-noise scores are supplied as a stream. The
  future RNG and fixed-point Gumbel producer must match this stream order and
  pass a distributional quality gate.
- **Synthesis:** native compilation validates function only. Vitis HLS timing,
  initiation interval, DSP packing, BRAM mapping, and AXI throughput are open.

## Counter RNG and Gumbel stream

- **Generator:** Philox4x32 uses ten rounds and constants from Random123. Three
  official known-answer vectors pin the implementation bit for bit.
- **Counter mapping:** vocabulary block, position, model-evaluation ID, and
  stream ID occupy the four counter words. Request seed halves occupy the key.
  Tuple uniqueness is the caller's responsibility.
- **Uniform conversion:** one 32-bit word represents the midpoint of its open
  uniform interval, `(word + 0.5) / 2^32`.
- **Gumbel format:** scores use signed Q5.10. The selected factorization uses
  one 256-entry mantissa table and three 256-entry correction tables, totaling
  2,048 bytes of signed INT16 storage.
- **Statistical gate:** 200,000 trials over 16 classes measure categorical
  total variation. The 1,024-sample full-vocabulary test measures pathwise
  candidate agreement, not complete distribution quality.
- **Requantization:** one signed Q12.20 multiplier per output channel converts
  the integer dot product to Q5.10 before bias and Gumbel addition. Multiplier
  calibration against real activation ranges remains open.
- **RTL architecture:** four ten-cycle iterative Philox cores generate an
  analytical 1.6 words per cycle. A four-block FIFO absorbs completion bursts,
  and two elastic Gumbel lanes consume two words per cycle. Integrated
  functional RTL holds score data and metadata stable under output backpressure.
- **Open mapping:** Yosys maps the integrated stream to 66 DSP, 4,136 LUT,
  1,880 flip-flop, 2 BRAM, and 14 distributed-RAM primitives. These are raw
  primitive counts, not placed utilization or a timing result.
- **Argmax boundary:** the selected position-ordered reducer accepts two
  pre-biased signed Q5.10 model scores in lockstep with the RNG, adds Gumbel
  noise, excludes the mask token, and emits one candidate per position. The
  combined boundary maps to 66 DSP, 4,787 LUT, 2,056 flip-flop, 2 BRAM, and 14
  distributed-RAM primitives.
- **Requantization boundary:** two elastic lanes convert signed INT32
  accumulators through Q12.20 multipliers using half-away-from-zero rounding,
  then add signed INT32 Q5.10 biases into 33-bit model scores. The complete
  accumulator-to-candidate boundary maps to 74 DSP, 5,117 LUT, 2,123
  flip-flop, 2 BRAM, and 14 distributed-RAM primitives.
- **Ordering:** the compact reducer requires position-major, token-major score
  order. It rejects a skipped or decreasing position instead of silently
  corrupting candidate state.
- **Open hardware gates:** vendor place-and-route timing, formal
  counter-collision checks, and a comprehensive RNG statistical suite are not
  yet complete. The output-head dot-product MAC producer remains external to
  the RTL boundary.

## Prefix-isolation quality gate

- **Dataset:** `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, test
  split, pinned revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
- **Sampling:** 32 deterministic token windows selected with seed 0.
- **Sweep:** prefix lengths 16, 64, and 128; suffix lengths 16, 32, and 64.
- **Quality protocol:** one-step true-token reconstruction of a fully masked
  suffix. This is not complete DDPM generation quality or chat evaluation.
- **Quality gates:** no more than 5% relative top-1 accuracy loss and no more
  than 5% mean negative-log-likelihood increase.
- **Cache gates:** 100% cached-versus-recomputed top-1 agreement and maximum
  normalized logit RMSE of `1e-5`.
- **Timing:** median across the 32 held-out windows after one warmup of each MPS
  shape and attention path. Prefix prefill is reported separately from cached
  suffix execution.

## Conditioned DDPM cache experiment

- **Inputs:** five deterministic WikiText-2 test windows with a 64-token fixed
  prefix and a 32-token generated suffix.
- **Sampler:** 64 requested ancestral DDPM transitions with the exact
  no-change probability cache enabled for the checkpoint's non-time-conditioned
  output distribution.
- **Replay:** every execution path resets to the same per-sample random seed.
- **Original comparison:** original full attention and prefix-isolated attention
  intentionally define different output distributions. Token agreement between
  them is descriptive, not a cache-correctness requirement.
- **Cache correctness:** retained-prefix execution must match the fully
  recomputed prefix-isolated token sequence and transition schedule.
- **Timing:** model-forward latency is separated from sampling wall time. The
  one-time prefix prefill is reported but excluded from steady-state model
  evaluation time.
- **Terminal state:** terminal K/V is valid only if the final model-evaluated
  suffix is identical to the completed suffix.

## Real MDLM trace

- **Documented:** checkpoint `kuleshov-group/mdlm-owt`, pinned to revision
  `d0958fa851335ece6c15260ce0025f030673c0fb`.
- **Assumed equivalent:** the published checkpoint code requires CUDA
  FlashAttention. On macOS the two used APIs are emulated with PyTorch
  scaled-dot-product attention.
- **Measured:** step latency surrounds each complete real checkpoint forward
  after an optional warmup.
- **Derived:** per-operation FLOPs and bytes come from the loaded checkpoint
  configuration and parameter storage.
- **Assumed:** the smoke sampler commits a deterministic fraction of remaining
  mask positions by confidence at each step. It validates execution and
  changing masks but does not reproduce the paper's DDPM sampling quality.
- **Documented and implemented:** the `ddpm` sampler uses the upstream MDLM
  SUBS parameterization, exponential-race categorical sampling, ancestral DDPM
  update, log-linear noise schedule, and final noise-removal pass.
- **Documented and implemented:** `ddpm-cache` reuses the last model probability
  tensor when a transition changes no tokens, matching the upstream behavior
  for this non-time-conditioned checkpoint. Trace replay includes model
  evaluations and currently omits the much smaller cache-hit categorical
  sampling overhead.
- **Measured baseline:** the unmodified model computes a full-vocabulary
  projection for every canvas position. `masked-only-output-head` is recorded
  as an optimization candidate, not silently assumed in the baseline.

## Timing

- **Assumed:** operation latency is the maximum of compute, HBM, and SRAM time.
- **Assumed:** compute and memory transfers overlap perfectly within one
  operation, but operations execute sequentially.
- **Derived:** effective throughput is peak throughput multiplied by a
  configurable efficiency factor.
- **Assumed:** GB/s and TOPS use decimal SI units. SRAM capacity named `mb` in
  YAML is interpreted as MiB (`1024^2` bytes) to match common architecture
  reporting.

## KV260 estimate

- **Documented:** 1.2 peak INT8 TOPS for the B4096F DPU at 300 MHz.
- **Documented:** 4 GB of 64-bit DDR4 at 2400 Mb/s, giving 19.2 GB/s
  theoretical peak bandwidth.
- **Documented:** 26.6 Mb total on-chip memory across distributed RAM, BRAM,
  and UltraRAM.
- **Assumed:** 55% compute utilization, 65% DDR efficiency, and 70% effective
  utilization of a 200 GB/s logical on-chip SRAM interface.
- **Assumed design point:** weight-only INT8 reduces weight traffic by 4x from
  the FP32 checkpoint. It does not change activation/logit storage or operation
  FLOPs in the trace.
- **Not yet validated:** quantization accuracy and exact DSP packing. No result
  using this configuration is a measured KV260 result.
- **Measured warning:** a five-input, 64-token fake-quantization probe produced
  87.5% mean top-1 agreement with FP32 and a 65.6% worst case. Naive
  per-output-channel INT8 is therefore not accepted as an accuracy-preserving
  implementation.

## Memory placement

- **Assumed:** the all-HBM policy places every tensor in HBM.
- **Assumed:** the canvas-SRAM policy may place canvas state, canvas K/V,
  logits, and diffusion metadata in SRAM when capacity permits.
- **Assumed:** an SRAM read miss fetches the tensor from HBM, writes it into
  SRAM, and then services the read from SRAM.
- **Assumed:** a newly produced tensor can be written directly into SRAM
  without an HBM read.
- **Assumed:** operation-lifetime tensors are released after their operation;
  step-lifetime tensors after their denoising step; canvas-lifetime tensors
  after the request.
- **Assumed:** dirty write-back at request completion is not yet modeled.

## Synthetic workload

- **Assumed:** the default synthetic model resembles a small dense masked
  diffusion transformer; it is not a performance proxy for MDLM or LLaDA.
- **Derived:** tensor sizes follow configured token count, hidden size,
  vocabulary size, layer count, and bytes per element.
- **Assumed:** the number of masked tokens falls linearly across denoising
  steps.
- **Assumed:** every layer reads its weight shard once per denoising step.

No synthetic result should be presented as a claim about real hardware or a
real model.
