# Full-hardware roadmap for the 169.6M MDLM

## Scope decision

The first complete hardware target is the pinned
`kuleshov-group/mdlm-owt` checkpoint at revision
`d0958fa851335ece6c15260ce0025f030673c0fb`.

This is a full inference target, not a sampling-only target. The first board
milestone runs all 12 model blocks and produces decoded text with a 64-token
canvas and batch size one. The final bitstream is intentionally specialized for
this exact checkpoint and tensor shape. Reusability lives in the generator,
verification framework, and operator library rather than in a generic runtime
overlay.

Host software may perform tokenization, request control, and final decoding.
All neural-network forward operations, diffusion sampling, and candidate
selection must execute in programmable logic for the complete-board milestone.

## Demonstration contract

The project has two linked operating modes:

1. Reference mode runs the pinned checkpoint and establishes correctness.
2. Sprint mode runs a hardware-tuned derivative of that checkpoint and
   establishes the model-specific acceleration claim.

For one 64-token block, the board target is less than 500 ms from token IDs to
completed token IDs. The stretch target is less than 250 ms. Both numbers must
include every model evaluation and sampling operation. Tokenization and final
text decoding are reported separately.

The measured 64-transition candidate-cache trace required 41 model evaluations
and spent 776.8 ms in model forwards on Apple MPS. The existing analytical
KV260 INT8 estimate is 947.6 ms. Porting the current graph unchanged therefore
does not satisfy the demonstration contract.

Sprint mode targets eight model evaluations. The current K26 commit point uses
768 effective INT8 MAC lanes, 250 MHz, and 75 percent utilization. Its
architecture-derived bound is about 469 ms per generation including 25 ms of
fixed overhead. A packed-INT8 stretch point reaches an analytical bound near
193 ms. These are targets for design-space pruning, not performance claims,
until vendor place-and-route and board measurements exist.

## Model-specific optimization ladder

The optimizations are applied in this order, with a quality and latency result
recorded after every rung:

1. Freeze the 64 by 768 by 12 graph and remove every dynamic tensor shape.
2. Bake matrix dimensions, strides, rotary constants, normalization constants,
   and the layer command stream into the bitstream.
3. Fuse LayerNorm, adaptive modulation, projections, residuals, and activation
   boundaries so intermediate tensors remain on chip.
4. Use masked-position vocabulary projection and stream accumulators directly
   into the sampler.
5. Calibrate state-aware INT8 activations and per-channel INT8 weights, with
   selective higher precision only where the checkpoint fails.
6. Add delayed or blockwise K/V reuse as an approximate sprint-mode option,
   never as an unlabelled exact optimization.
7. Tune confidence-profile parallel commits to reduce model evaluations without
   retraining.
8. Distill or fine-tune an eight-evaluation schedule if training-free decoding
   cannot meet the quality gate.
9. Co-search precision by layer, tensor tiling, DSP packing, and denoising
   schedule against the physical FPGA reports.

The main research result is the Pareto curve across latency, quality, power,
and FPGA resources. The best point becomes the sprint bitstream. The unmodified
checkpoint remains available as the reference bitstream so the speedup cannot
be attributed to silently changing the task.

### Proven checkpoint-specific fold

The checkpoint sets the timestep input to zero whenever time conditioning is
disabled. Consequently, its timestep embedding and every adaptive-normalization
projection produce constants for every request and denoising evaluation. The
hardware exporter therefore computes them once and folds them into the QKV,
attention output, MLP, and vocabulary projection matrices and biases.

This removes 7,380,736 runtime conditioning parameters and evaluates 56,832
constant values offline. Those constants, plus 19,200 normalization weights,
are absorbed into the adjacent matrices and biases. It also removes the
timestep MLP, all adaptive-normalization matrix multiplications, modulation
multiplications, and residual gates from the runtime graph. The algebraic fold
is covered by randomized equivalence tests and a real-checkpoint comparison.

The full token-embedding table is stored in DDR but is not streamed on every
evaluation. At most 64 embedding rows are read for the first canvas, and the
98 KB embedded canvas can then be updated only at changed token positions. With
constant folding and this lookup behavior, per-evaluation weight traffic is
about 247.4 MB in FP16 or an analytical 123.7 MB for an INT8 core, rather
than the complete checkpoint file size.

## Frozen model

| Property | Value |
| --- | ---: |
| Parameters | 169,627,218 |
| Transformer blocks | 12 |
| Hidden dimension | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| MLP dimension | 3,072 |
| Vocabulary | 50,258 |
| Maximum checkpoint length | 1,024 |
| First hardware canvas | 64 |
| Attention | Bidirectional |
| Position encoding | Rotary |
| MLP activation | GELU, tanh approximation |
| Time conditioning | Disabled in checkpoint |

The parameter count decomposes as follows:

| Parameter group | Parameters |
| --- | ---: |
| Token embedding | 38,598,144 |
| Timestep embedding | 49,408 |
| Twelve DDiT blocks | 92,132,352 |
| Final normalization | 768 |
| Vocabulary projection and bias | 38,648,402 |
| Final adaptive normalization | 198,144 |
| Total | 169,627,218 |

Raw parameter storage is approximately 678.5 MB in FP32, 339.3 MB in FP16,
and 169.6 MB in INT8. Both the FP16 reference and future mixed-precision model
fit in the K26 SOM's 4 GB external DDR.

## System boundary

```text
Arm processing system
  tokenization, request setup, DDPM schedule, decoded text
             |
             v
Programmable logic
  command processor and DMA
             |
             v
  embedding lookup
             |
             v
  12 x DDiT block
    LayerNorm and adaptive modulation
    QKV projection
    rotary position transform
    bidirectional attention and softmax
    attention output projection and gated residual
    LayerNorm and adaptive modulation
    768 -> 3072 -> 768 GELU MLP and gated residual
             |
             v
  final normalization and vocabulary projection
             |
             v
  Q12.20 requantization
             |
             v
  Philox, Gumbel, noisy argmax, candidate cache, reveal
             |
             v
  updated token canvas
```

The existing RTL covers the path from signed output-head accumulators through
requantization, sampling, and final candidate IDs. The remaining neural path is
the reusable tensor engine, vector operations, attention pipeline, embedding,
and system memory/control integration.

## Reusable hardware architecture

### 1. Tensor engine

A tiled matrix engine is the central reusable block. The initial design point
uses 512 signed 16-bit MAC lanes with wider accumulation. It must support:

- matrix-vector and small matrix-matrix multiplication;
- rectangular QKV, attention-output, MLP, and vocabulary matrices;
- weight reuse across the 64 sequence positions;
- double-buffered activation and weight tiles;
- a later packed INT8 mode without changing the command protocol.

At an assumed 300 MHz, 512 MAC lanes provide a 153.6 GMAC/s arithmetic ceiling.
The 64-token model contains approximately 8.0 billion MACs per full forward, so
the compute-only lower bound is about 52 ms per evaluation before utilization,
vector operations, and control overhead. This is an assumption until vendor
timing and measured execution exist.

### 2. Vector engine

The vector datapath handles operations that should not occupy the matrix array:

- mean and variance reduction for LayerNorm;
- affine modulation and gated residuals;
- GELU tanh approximation;
- SiLU for the small conditioning path;
- rotary position transforms;
- scaling, conversion, clipping, and format checks.

The first implementation uses table or piecewise-polynomial approximations with
bit-accurate software references. Each approximation needs an explicit error
gate against the pinned FP32 checkpoint.

### 3. Attention engine

For the 64-token first target, Q, K, and V occupy 294,912 bytes in FP16. All
attention state can remain on chip while weights stream from DDR. The engine
performs:

- twelve independent 64-dimensional heads;
- bidirectional QK transpose multiplication;
- row maximum and exponent approximation;
- normalized softmax;
- probability times V;
- head concatenation into 768-wide activations.

The design must never assume causal attention or autoregressive KV-cache reuse
inside the active denoising canvas.

### 4. Memory system

Weights remain in external DDR and are laid out in execution order. A forward
streams each dense matrix once and reuses each fetched weight across active
positions. Embeddings use indexed row reads and are retained on chip.

At the modeled 12.48 GB/s effective DDR rate, specialized FP16 weight traffic
has a lower bound near 20 ms per evaluation and specialized INT8 traffic near
10 ms. The 512-lane FP16 tensor engine is
therefore expected to be compute-limited first, although routing and achieved
clock rate can change that conclusion.

Required on-chip FP16 working sets for a 64-token canvas include:

| Buffer | Bytes |
| --- | ---: |
| One 64 x 768 activation canvas | 98,304 |
| QKV state | 294,912 |
| MLP intermediate | 393,216 |
| Attention score state | 98,304 |

These buffers are time-multiplexed rather than allocated for every layer. The
architecture uses activation ping-pong buffers, a tiled weight buffer, QKV and
attention scratch, and a streamed MLP intermediate. Complete logits are never
materialized in DDR.

### 5. Diffusion controller

The controller owns the token canvas, mask bitmap, evaluation number, stopping
conditions, probability-cache validity, and candidate-cache validity. Because
the checkpoint disables time conditioning, an evaluation that changes no token
can reuse its model distribution exactly.

The existing candidate and reveal blocks become the controller's final stage.
The host may select the number of requested diffusion steps, but model-forward
skips and token commits occur in hardware.

## Numerical plan

### Reference mode

- FP16 weights and activations;
- FP32 or sufficiently wide fixed-point accumulation;
- software-visible intermediate checkpoints after every major operator;
- exact token-candidate comparison against the macOS reference.

Reference mode prioritizes correct complete inference over throughput. It is
allowed to be slower than the final optimized mode.

### Optimized mode

- calibration-aware INT8 or mixed-precision weights;
- FP16 or calibrated fixed-point activations;
- per-channel output scaling;
- the existing 33-bit pre-noise and 34-bit noisy score path;
- FP16 fallback at layers that fail the accuracy gate.

Naive whole-model INT8 remains rejected because its measured top-1 agreement
was only 87.5%. No optimized format is accepted until full-generation quality
and layerwise error gates pass.

### Sprint quality gates

Latency is reported only for variants that pass all applicable gates:

- constant folding must match the unfused graph within the FP32 reference
  tolerance at every block boundary;
- quantization alone must retain at least 99 percent masked-token top-1
  agreement and mean teacher-to-student KL divergence below 0.02 on the frozen
  calibration corpus;
- eight-evaluation decoding must keep held-out negative log-likelihood within
  2 percent of the 64-transition reference;
- diversity, repetition, mask completion, and length statistics must remain
  inside the reference run's 95 percent bootstrap interval;
- the board must produce the same result as the bit-accurate sprint simulator
  for identical token inputs and random seeds.

These are acceptance targets. They are not current results. If a target fails,
the corresponding precision, cache, or schedule point is excluded from the
latency Pareto curve.

Recent diffusion-language-model work supports two sprint-mode experiments that
remain approximate until validated here. Fast-dLLM reports blockwise K/V reuse
and confidence-aware parallel decoding, while dKV-Cache reports delayed K/V
caching. Both techniques were evaluated on other model configurations, so this
project treats them as hypotheses for MDLM-OWT rather than inherited results.
Few-step distillation becomes the fallback when training-free schedule search
cannot reach eight evaluations without failing quality.

Primary references:

- [Fast-dLLM](https://arxiv.org/abs/2505.22618)
- [dKV-Cache](https://arxiv.org/abs/2505.15781)
- [STaR-Quant](https://arxiv.org/abs/2606.04945)
- [Multi-Mask Diffusion](https://arxiv.org/abs/2607.19686)

## Implementation milestones

### H0: Frozen reference package

Status: complete. The generated artifacts, hashes, real-checkpoint numerical
results, and regeneration commands are recorded in
`docs/h0-hardware-package.md`.

Deliverables:

- pinned model manifest and parameter inventory;
- deterministic 64-token inputs and intermediate tensors;
- per-operator golden outputs;
- decoded seed outputs;
- byte-exact weight export with checksums.

Acceptance: a clean macOS run regenerates every golden tensor and manifest.

### H1: Tiled tensor engine

Status: in progress. All twelve MLPs now pass the local RTL-equivalent W8A8,
requantization, and GELU reference. The registered 768-lane MAC and activation
and weight traversal controller pass portable RTL simulation. Ping-pong weight
slices, serialized requantization/GELU, MLP-down residual integration, a
streamable SmoothQuant INT8 interstage with 2 KB of tile staging, and a
two-pass runtime up-activation quantizer, a fixed LayerNorm producer, and a
512-bit weight-stream adapter are now tested. A one-outstanding AXI burst
reader and model-specific QKV weight-slice DMA now reach the existing compute
loader exactly. Physical memory inference, metadata loading, multi-outstanding
traffic, and vendor timing remain open. Evidence is tracked in
`docs/h1-fixed-mlp.md`.

Deliverables:

- synthesizable dual-buffered GEMM core;
- signed 16-bit reference mode and wide accumulation;
- matrix-shape command interface;
- native, RTL, and randomized reference tests;
- Yosys and Vitis resource reports.

Acceptance: all MDLM matrix shapes match the bit-accurate software reference,
including partial tiles.

### H2: Vector operations

Status: in progress. The unaffine LayerNorm used at all 25 runtime boundaries
now has a bit-accurate software model, an all-boundary H0 error sweep, and a
streamed four-token RTL core. The core performs one Q13.10 statistics pass and
can replay normalized Q5.12 values to multiple consumers without storing a
second normalized canvas. Its first connected use drives the MLP-up runtime
quantizer directly. Adaptive modulation is absent at runtime because the
checkpoint-specific H0 fold absorbed its constants. Residual addition and GELU
are already present in the MLP path. Rotary is exact for every coordinate of a
real 64 by 64 head and is connected to the QKV producer and attention
scratchpad. The attention-side residual boundary is also exact.

Deliverables:

- LayerNorm;
- adaptive shift, scale, and gates;
- residual addition;
- GELU, SiLU, and rotary transforms;
- numerical-error report for every operator.

Acceptance: one operator pipeline matches the frozen checkpoint tolerances and
fits alongside the tensor engine.

### H3: Bidirectional attention

Status: RTL functional gate complete; vendor implementation remains open.
Fixed rotary, QK, streamed four-row softmax,
probability-times-V, complete 64-token head scheduling, sequential 12-head
reuse, the 768-channel attention canvas, folded INT8 output projection, and the
Q13.10 residual boundary now have exact checkpoint-derived RTL tests. The
complete attention reference has 0.1354% mean and 0.1725% maximum relative RMS
error. The fixed LayerNorm and INT16 QKV cascade preserves attention at 0.1608%
mean error. Output projection has 0.9353% mean error and the post-residual
boundary has 0.3460% mean error across all twelve H0 blocks. QKV storage maps to 48
RAMB18E2 primitives, the attention canvas to 64 RAMB18E2 primitives, and the
residual canvas to 24 URAM288 primitives. Attention and projection now share
one physical 768-multiplier array in a connected exact RTL test. A streamed
QKV output tile is exact in 969 cycles and its normalized canvas maps to 32
URAM288 primitives. Rotary staging now writes all 4,096 Q/K coordinates into
the real attention scratchpad in 4,099 cycles, and a connected fixed-QKV head
test is exact. The automatic 33-tile controller, Q/K/V output router, separate
Q/K and V scratchpad write addresses, and one-head shared-array pipeline are
now connected. A full head takes 47,127 cycles in RTL with one physical
192-lane MAC array and matches every real block-0 head-0 QKV and attention
value exactly. The 12-head producer, attention canvas, folded output
projection, and residual stage now form one structural pipeline. Reduced
two-head and two-output-tile tests cover the cross-module sequencing. Two
opt-in exhaustive block-0 runs now check all 49,152 attention-canvas integers
and all 49,152 post-residual integers around the exact canvas boundary. The
next model-integration gate is one complete DDiT block, followed by vendor
implementation. Evidence is tracked in
`docs/h3-fixed-attention.md`.

Deliverables:

- QKV buffering and head scheduler;
- QK transpose and probability-times-V commands;
- stable softmax implementation;
- 64-token, 12-head reference comparison.

Acceptance: attention output meets the agreed error and top-1 downstream gates.

### H4: One complete DDiT block

Status: in progress. The exact H3 attention output now overwrites and reuses
the existing residual UltraRAM canvas. That canvas feeds an automatic
three-pass norm2 and SmoothQuant frontend, a ping-pong MLP up engine, streamed
GELU-to-INT8 staging, MLP down, and saturating residual add. Up and down share
one physical external MAC interface. A block controller and concrete
weight-plus-metadata loaders automatically schedule the entire MLP sublayer.
A reduced structural run completes from one start pulse in 11,372 simulated
cycles with 3,120 shared-array requests and exact outputs. A captured block-0
run now covers four real tokens, all 512 up tiles, the complete 3,072-channel
interstage, and the first six down outputs. It matches all 24 post-residual
integers exactly after 12,384 shared-array requests and 34,881 simulated cycles.
The exhaustive automatic MLP gate now closes the production MLP shape for the
captured block-0 state. All 64 tokens, 512 up tiles, 3,072 interstage channels,
128 down tiles, and 49,152 post-residual integers pass exactly. The run issues
393,216 physical-array requests and completes in 471,939 simulated cycles. Two
consecutive Icarus runs passed in 572.54 and 574.10 host seconds. This is
cycle-accurate functional evidence, not KV260 timing.
`ddit_block_pipeline` now supplies that phase top. It sequences attention,
folded projection, in-place hidden-canvas handoff, norm2, and automatic MLP
through one physical mixed-precision array. A reduced connected test with one
computed head, two projected output tiles, four MLP tokens, 128 up tiles, and
one down tile completes in 66,408 simulated cycles with exact zero-reference
outputs. The eleven heads omitted from this control test are explicitly
zero-padded only in the testbench. The next gate is the complete connected
block with all 12 real attention heads feeding the now-closed full MLP, followed
by randomized captured block boundaries. Vendor implementation remains open.

The first concrete H4 DDR artifact is also complete. The optimized block-0
execution image is 9,064,448 bytes, contains ten 4 KiB-aligned sections, and is
independently validated by whole-image and per-section hashes. Synthesizable
address logic maps every H4 section and record index to a 64-byte-aligned burst
address, beat count, payload size, and compact-record byte offset. A portable
512-bit AXI master now enforces response framing, propagates backpressure, and
counts bytes and stalls. The connected parameter DMA covers aligned and compact
records plus invalid requests. A weight-slice loader issues all 24 QKV record
reads for one output tile and produces the existing compute-loader words. For
the real block-0 binary, all 24 packed INT16 tiles match an independently
recomputed frozen QKV tensor exactly after 9,216 transferred bytes. Metadata
record adapters now cover every aligned metadata geometry plus compact rotary
and reciprocal entries. A single-master QKV operator loader fetches one real
metadata record followed by all 24 real weight records and drives the existing
compute scheduler interface exactly after 9,280 transferred bytes. Projection
loading now performs the analogous 4,672-byte INT8 transfer. Connected tests
drive each loader into its physical MAC scheduler and match all 64 by 6 QKV and
projection outputs exactly, with no manual parameter injection. MLP
up and down output-tile loaders now cover their 24-record and 96-record weight
slices plus 444-bit and 1,344-bit metadata, matching real block-0 values after
4,672 and 18,624 transferred bytes. A checked round-robin arbiter now packages
all four operator loaders behind one physical 512-bit AXI read port. Multiple
outstanding reads, vendor integration, and measured DDR traffic remain open.
Layout and transport evidence is tracked in
`docs/h4-block-execution-image.md`.

An open-source K26-family synthesis screen found and removed five accidental
DSP48E2 uses from constant address strides. The shared load fabric now screens
at zero DSPs, 11,462 flip-flops, 370 carry cells, and 1,633 LUT primitives,
excluding the interpretation of top-level pad buffers. These are structural
screening counts, not Vivado placement, timing, or board measurements.

The shared load fabric is now connected directly to the DDiT controller. A
reduced one-head, two-projection-tile, 128-up-tile, one-down-tile run completes
in 93,889 simulated cycles after 4,100 AXI reads and 918,400 transferred bytes.
All attention and final outputs remain exact. The next connected gate replaces
the reduced zero fixture with all twelve real block-0 heads and the full MLP.

The static-table preloader now reads every rotary and SmoothQuant reciprocal
entry from the same block image. Its real test emits 2,048 exact rotary pairs,
stores 768 exact reciprocals, and accounts for 176 AXI reads and 11,264 bytes.
The image-fabric top now arbitrates this preloader and the dense-parameter
fabric behind one external AXI read port, gates compute on completed preload,
and passes a reduced end-to-end control run. That run accounts for 4,276 reads
and 929,664 bytes exactly. The next system step is the full real block-0 gate
with 12 attention heads, 128 projection tiles, and 128 MLP-down tiles.

That full real block-0 gate now passes autonomously. The one-port top transfers
9,060,096 bytes in 38,492 reads and completes in 1,584,457 simulated cycles.
Norm1 is computed internally while the legacy normalized input is held at zero.
All 2,048 attention boundary vectors and 2,048 final vectors match the
independent fixed reference bit for bit. At an unverified 250 MHz this is
6.337828 ms per block, which projects to 608.431488 ms for twelve blocks across
eight evaluations before the rest of the model. The target therefore needs at
least one scheduling or parallelism improvement, not just board integration.
Internal norm1 is now connected and passes both a 384-tile real-embedding test
and a reduced autonomous block run. Its standalone screen adds 32 RAMB36 and
25 DSPs.

The first model-specific schedule pass groups all four tokens in each denoising
group into one attention-canvas word and streams one projection K tile per
cycle. It also processes four requant lanes per cycle in QKV and attention
projection. Real two-tile projection stays bit exact while falling from 5,797
to 1,413 cycles. The reduced autonomous screening run falls from 140,301 to
126,393 cycles. The grouped canvas maps to 64 RAMB36E2 primitives, and the two
retained four-lane requantizers add 24 DSPs. Parallel MLP-up and MLP-down
postprocess experiments were rejected. MLP-up work was hidden by the MAC
schedule, while MLP-down saved only 18 full-block cycles for 12 extra DSPs.

The complete optimized full-shape rerun is bit exact at 1,189,573 cycles,
38,492 reads, and 9,060,096 read bytes. It saves 394,884 cycles, or 24.9224
percent, from the 1,584,457-cycle baseline. This compiled checkpoint includes
the later-rejected QKV vector drain and MLP-down four-lane experiment. Restoring
both retained scalar paths adds 258 cycles, making the equivalent checkpoint
1,189,831 cycles. At an unverified 250 MHz, that is 456.895104 ms for 12 blocks
across eight evaluations, excluding final normalization, the vocabulary head,
and sampling. This is portable RTL simulation, not board timing.

A parallel QKV scratchpad drain experiment was not retained. With four-lane
requantization active, the serial production drain completes a real block-0
head in 37,623 cycles and the vector prototype takes 37,603 cycles. The scalar
writes are already hidden behind the next QKV group compute interval, so the
prototype exposes only 20 saved cycles while adding about 2,968 estimated logic
cells across the router and two scratchpads. A generic bank-native crossbar was
also rejected at 19,495 logic cells and 9 DSPs. The earlier 47,127-cycle head
used serial requantization; 9,504 of the 9,524-cycle difference came from the
retained four-lane requantizer.

A continuous QKV scheduler is now the production path. It streams all sixteen
24-K-tile groups, carries group tags through the MAC, overlaps four-lane
requantization, and buffers two completed groups for the scalar router. A real
block-0 output tile falls from 683 to 428 cycles. The connected real head stays
bit exact and falls from 37,623 to 29,703 cycles, saving 7,920 cycles with the
same 24 DSPs in the scheduler hierarchy. Open-source mapping estimates only 13
more logic cells and 411 fewer flip-flops. Applying the head result across 12
heads projects 1,094,791 retained-production cycles, or 420.399744 ms for the
12-block, eight-evaluation backbone at an unverified 250 MHz. The complete
rerun passes at 1,094,773 cycles with all 2,048 attention and 2,048 final
vectors bit exact. It compiled the rejected four-lane MLP-down postprocess, so
the retained serial equivalent is 1,094,791 cycles. Traffic is unchanged at
38,492 reads and 9,060,096 bytes. Vendor timing remains open.

The next retained compute point doubles MLP token width from four to eight
without doubling the DSP count. Adjacent signed INT8 token activations share
each output weight and are packed into one 27 by 18 multiply. A real captured
eight-token MLP slice remains bit exact and falls from 50,834 to 32,286 cycles,
saving 18,548 cycles, or 36.4874 percent. The corresponding full-block result
is still an analytical 887,718 cycles until the full execution-image rerun
finishes.

The physical array now uses four groups of eight DSP48E2 cascade cells for each
attention output. Attention products are reduced through dedicated PCIN and
PCOUT paths, while MLP mode exposes every packed product to a narrow fabric
tree. The array passes eight randomized attention results, eight randomized
packed-MLP results, a 20,000-vector cascade-cell test, and a reduced connected
DDiT run exactly. Open-source UltraScale+ mapping reports 768 DSP48E2s, 54,082
LUT primitives, and 40,878 flip-flops. Those counts are 61.5385, 46.1766, and
17.4513 percent of the documented K26 capacities. The old four-token shared
array maps to the same 768 DSPs, 38,696 LUTs, and 58,242 flip-flops, so the
retained eight-token point buys twice the MLP token throughput for 15,386 more
LUTs and fewer fabric registers. This is an open-source structural screen of
explicit primitives. Vivado placement, cascade legality across the chosen
floorplan, achieved frequency, and board behavior remain unverified.

The first KV260 kernel boundary is also present. A DDR residual reader,
autonomous block core, sparse DDR output writer, and AXI-Lite register shell
elaborate as one top. The two edge DMAs and the register protocol pass portable
RTL tests. A reduced full-kernel run performs 6,324 reads, moves 1,191,808 read
bytes, writes the exact output record, and completes in 135,051 cycles. Input
overlap first reduced 152,127 cycles to 148,957, and grouped projection plus
four-lane requantization then removed another 13,906 cycles. The combined
reduction is 17,076 cycles, or 11.2248 percent. A
deferred Yosys check resolves the complete production hierarchy. The script
`fpga/kv260/run_synth.tcl` targets the XCK26 and requests 250 MHz, but it
remains unexecuted because Vivado is unavailable on the macOS host. The memory
layout and register contract are in `docs/kv260-kernel-interface.md`.

Deliverables:

- tensor, vector, attention, modulation, and residual pipelines connected;
- DDR weight layout for one block;
- cycle and traffic counters;
- comparison at every block boundary.

Acceptance: one real checkpoint block passes on randomized and captured hidden
states with no host arithmetic in the forward path.

### H5: Twelve-block backbone

Deliverables:

- layer command table;
- all 12 checkpoint blocks;
- embedding and final normalization;
- deterministic full-backbone hidden-state comparison.

Acceptance: the FPGA-oriented model produces a final hidden canvas within the
numerical gate for all frozen inputs.

### H6: Vocabulary projection and existing sampler

Deliverables:

- tiled 768 by 50,258 output projection;
- direct accumulator stream into the completed requantization and sampler RTL;
- no full-logit DDR round trip;
- candidate IDs matching the reference.

Acceptance: one complete model evaluation produces all 64 reference candidates.

### H7: Complete DDPM inference

Deliverables:

- token canvas and mask control;
- exact no-change probability reuse;
- candidate cache and reveal loop;
- configurable transition count;
- decoded text output through the Arm host.

Acceptance: complete 64-token generations terminate, remove all mask tokens,
and match the accepted hardware numerical mode.

### H8: AXI, runtime, and board image

Deliverables:

- AXI memory masters and control registers;
- aligned weight image and loader;
- PYNQ or native Linux control program;
- counters for cycles, DDR bytes, stalls, and cache hits;
- reproducible SD-card or boot image instructions.

Acceptance: a fresh KV260 boots and runs the complete checkpoint without a
development workstation performing neural arithmetic.

### H9: Timing, power, and optimization

Deliverables:

- Vitis synthesis and place-and-route reports;
- achieved frequency and resource utilization;
- measured board latency, DDR traffic, and power;
- mixed-precision comparison;
- 64, 128, and 256-token scaling results.

Acceptance: reported performance uses board measurements, not analytical clock
assumptions.

### H10: Model-specific sprint bitstream

Deliverables:

- checkpoint-folded weights and baked 64-token schedule;
- state-aware quantization and selective precision map;
- confidence-profile commit controller;
- optional approximate K/V reuse with an explicit quality result;
- optional eight-evaluation distilled checkpoint;
- latency, quality, power, and resource Pareto report;
- reference and sprint bitstreams on the same board.

Acceptance: the sprint bitstream passes every quality gate and completes one
64-token block in less than 500 ms. Less than 250 ms is the stretch result.
Both results must be measured from token IDs in to token IDs out.

## Purchase and execution gates

Development through H3 remains possible on macOS with native C++, Python,
Icarus Verilog, and Yosys. Vendor synthesis and board image creation require an
AMD-supported Linux environment.

A physical board is justified when H1 through H4 have vendor synthesis reports
showing that the tensor, vector, attention, and control blocks fit together at
a useful clock. Buying earlier is optional for motivation, but it does not
remove the Linux toolchain or timing-closure work.

## Cross-modal reuse

The reusable assets are the tiled tensor engine, vector engine, attention and
softmax machinery, DMA fabric, precision tools, runtime, and verification
framework. Image and voice diffusion systems can reuse these foundations, but
they require different front ends, tensor shapes, scheduling, and output
pipelines. The first goal is therefore a reusable accelerator substrate proven
by one complete 169.6M language model, not one universal bitstream.

## Completion definition

The 200M-class hardware project is complete when a physical FPGA provides two
auditable results. The reference bitstream runs the pinned 169.6M checkpoint
from token IDs through the complete DDPM loop. The sprint bitstream passes the
quality contract and completes a 64-token block in less than 500 ms. Both
report measured latency, memory traffic, utilization, and power. Host
tokenization and decoding are allowed. Host execution of model layers is not.
