# H3 fixed bidirectional attention

Status: in progress. The bit-accurate path now covers rotary, QK, four-row
stable softmax, probability-times-V, all 64 positions of one head, sequential
reuse across 12 heads, the folded attention output projection, and its
saturating residual. Attention and projection now arbitrate one physical MAC
array. QKV projection and rotary now feed the head scratchpad directly, the
head engine advances into the 12-head canvas, and that canvas feeds the folded
output projection and residual stage. Full-shape checkpoint data and vendor
implementation remain open.

The exhaustive block-0 RTL gates are opt-in because of portable simulator
cost. The 12-head QKV, rotary, and attention run compares all 49,152 canvas
integers and passed in 1,021.03 wall-clock seconds. The full 128-tile folded
output projection and residual run compares all 49,152 block outputs and
passed in 357.54 wall-clock seconds. These phases meet at the same exact
768-channel canvas boundary. Commands, source hashes, bounded RTL cycle checks,
and limitations are recorded in
`data/results/h3-full-attention-block0-rtl-validation.json`.

## Numerical boundary

The accepted reference-mode attention formats are:

| Quantity | Format |
| --- | --- |
| Q, K, V, attention output | signed 18-bit Q5.12 |
| Rotary cosine and sine | signed 16-bit Q1.15 |
| Scaled QK score | signed 18-bit Q7.10 |
| Exponential and probability | unsigned 16-bit Q0.16 |
| Exponential sum | unsigned 22-bit |
| Softmax reciprocal | `round(2^30 / sum)`, consumed with shift 14 |

`fixed_rotary_q12` was swept over every Q/K tensor in all twelve H0 blocks.
It has zero observed input saturation. Maximum relative RMS error is 0.00905%
for query and 0.00761% for key. The largest rotated integer magnitude is
82,024, within signed 18-bit range.

The stable softmax subtracts each row maximum, clips deltas below -16, and uses
a 1,025-entry `exp(-x)` table at 1/64 spacing. It computes one reciprocal per
row, not one division per probability. Across all twelve complete H0 attention
boundaries, fixed QK, softmax, probability-times-V, and Q5.12 output have
0.1354% mean and 0.1725% maximum relative RMS error. Q, K, V, and score
saturation are zero on this trace. The probability row-sum error is at most 41
units in Q16. These measurements cover one frozen H0 input and are not a
held-out quality result.

## Rotary RTL

`rotary_qk_pair_serial` accepts one token, one head, and one paired pair of Q/K
channels per cycle. Eight registered multipliers compute both halves of query
and key. At 64 tokens, 12 heads, and 32 pairs, the schedule is 24,576 input
cycles per block, or about 0.098 ms at the provisional 250 MHz target.

The checkpoint-derived test streams all 1,536 pair records for the first four
block-0 tokens and checks every rotated Q/K integer and tag exactly. Open-source
K26 mapping uses 8 DSP48E2 primitives, 416 LUT primitives, and 376 flip-flops.

## Softmax-row RTL

`attention_softmax_row_q16` stores one 64-score row, finds its maximum, scans a
synchronous BRAM exponential table, accumulates the denominator, performs one
31-bit iterative reciprocal division, and emits one Q16 probability per cycle.
It supports output backpressure and preserves head, query, and key tags.

The real block-0 head-0 query-0 test checks all 64 probabilities exactly
against the integer reference. Open-source K26 mapping uses one DSP48E2, one
RAMB36E2, 388 LUT primitives, and 300 flip-flops. Six small distributed RAM
primitives hold the score and exponential rows. Vendor timing is not validated.

## Shared-array decision

A dedicated eight-lane QK and value unit would add roughly 37.7 ms per model
evaluation, or about 302 ms across eight evaluations. That is incompatible
with the system latency budget. Reference mode will therefore generalize the
existing 768-lane tensor array for signed Q5.12 by signed Q5.12 QK and unsigned
Q16 by signed Q5.12 probability-times-V products. Both operand pairs fit a
DSP48E2 multiplier. `mixed_precision_mac_tile_pipelined` places a runtime
sign-extension mux before the existing registered multiplier tree. It supports
signed 18-bit operands for QK, nonnegative Q16 probabilities encoded in signed
18-bit storage for probability-times-V, and low-byte sign extension for the
existing MLP INT8 mode. The self-checking test covers all three modes and
accumulator reuse.

A 192-lane quarter-width mapping uses 192 DSPs, 9,671 LUT primitives, and
14,601 flip-flops. Linear projection to four token lanes is 768 DSPs, 38,684
LUT primitives, and 58,404 flip-flops. The projection is an architecture
screen, not a completed full-width synthesis result. It is materially larger
than the INT8-only tree but remains within the standalone K26 capacity. The
complete system fit still depends on SRAM inference and vendor implementation.
Attention-specific INT8 remains a sprint experiment after the complete fixed
path passes.

The first paired-token INT8 attention screen is now reproducible. Q, K, and V
use symmetric signed INT8 with one captured calibration maximum per block and
head. Softmax probabilities use levels zero through 127 so two query rows can
share one signed-INT8 packed multiplier against the same K or V value. All
requantization and reconstruction scales are represented by Q24 or Q28 integer
multipliers in the screen.

Across the same captured H0 input for all 12 blocks, the candidate has 3.0986
percent mean and 3.7384 percent maximum relative RMS error against the accepted
18-bit fixed attention. Mean cosine similarity is 0.999524 and the worst block
is 0.999305. After the existing folded attention projection, relative RMS error
is 3.3987 percent mean and 4.7002 percent maximum. The weakest individual
head keeps 90.625 percent score-argmax agreement. These values are promising
for sprint mode but do not establish text quality because calibration and
evaluation use the same single captured input.

If an eight-query scheduler and eight parallel softmax rows sustain the packed
array rate, the current 180,672-cycle full-block head-attention phase has an
idealized 90,336-cycle bound. Combining that with the still-unverified wide-MLP
projection gives 797,382 cycles per block, or 306.194688 ms for the 12-block,
eight-evaluation backbone at an unverified 250 MHz. This is an analytical
pruning number, not a simulation or board result. The next gate is held-out
end-to-end text quality before building the new attention scheduler. Evidence
and the exact reproduction command are in
`data/results/h4-packed-int8-attention-screen.json`.

The fixed calibration was then applied to five fresh deterministic half-masked
canvases, covering 60 independently evaluated block boundaries. Mean attention
relative RMS error is 3.5851 percent and the maximum is 5.5716 percent. After
the folded projection, the mean is 3.8397 percent and the maximum is 5.5766
percent. At most 0.5371 percent of any Q, K, or V operand tensor saturates. The
weakest individual head has 82.8125 percent score-argmax agreement. This is a
meaningful calibration-transfer pass, but prior-block quantization error is
not propagated. The next software gate must replace attention inside the full
12-block forward path and measure logits plus generated text. Evidence is in
`data/results/h4-packed-int8-attention-heldout-screen.json`.

All resource numbers are open-source technology mappings. They do not establish
vendor placement, routing, timing, or power.

## QKV scratchpad and complete head

`attention_head_scratchpad_banked` uses 16 Q banks, 16 K banks, and 16
transposed V banks. Q and K reads return 16 adjacent channels. V reads return
16 adjacent key positions for one channel. Open-source K26 mapping is exactly
48 RAMB18E2 primitives, 32 LUTs, 3 flip-flops, and no DSPs.

`attention_qk_group_scheduler` and `attention_pv_group_scheduler` each use two
32-element inner tiles per MAC accumulation. The connected four-query group
uses one physical `mixed_precision_mac_tile_pipelined` array for both phases.
It matches all 256 real block-0 head-0 outputs exactly and takes 939 busy cycles.

Four independent softmax rows were necessary to keep the MAC fed, but the
first wrapper used dynamic score-matrix reads and mapped to 61,545 LUTs. The
accepted streamed wrapper serializes each six-key QK tile directly into four
row engines. It maps to 4 DSPs, 4 RAMB36E2 primitives, 2,057 LUTs, and 5,750
flip-flops. This is a 29.9 times LUT reduction from the rejected wrapper.

`attention_head_pipeline` reuses that group engine across all 16 four-query
groups. It matches all 4,096 real outputs of block-0 head 0 exactly in 15,056
busy cycles after its QKV scratchpad is loaded. The integrated head scratchpad
and execution wrapper takes 19,158 busy cycles including 4,096 scalar QKV
loads and final canvas drain.

## Twelve-head canvas

`attention_multihead_controller` sequences all 12 heads through one head
engine and refuses QKV loads tagged for the wrong head. Its 12-head control
test passes. `attention_multihead_canvas_pipeline` then writes each complete
head into an on-chip 64-token by 768-channel attention canvas. A real-checkpoint
test reads the canvas back and matches all 4,096 head-0 values exactly.

The first canvas write router mapped to 8,449 LUTs and 5 DSPs. Replacing
variable per-bank part selects with one shared token selector and constant bank
lanes reduced the same exact design to 177 LUTs, 452 flip-flops, 64 RAMB18E2
primitives, and zero DSPs. That 47.7 times LUT reduction is retained as a
specific example of the specialization work this project is meant to show.

## Folded output projection and residual

The attention output projection consumes the Q5.12 canvas directly. Its folded
768 by 768 weight uses signed INT8 with one scale per output channel. A 24-bit
unsigned multiplier and right shift 24 convert the 48-bit accumulator to
signed Q13.10. Across all 12 H0 blocks, the complete fixed projection has
0.9353% mean and 1.3705% maximum relative RMS error against the golden
projection when driven by fixed QKV. The following residual boundary has
0.3460% mean and 0.8257%
maximum relative RMS error. Neither boundary saturates on the frozen trace.

`attention_projection_output_tile_scheduler` buffers six output rows as 192
distributed RAM banks, reads the 768-channel attention canvas by head, reuses
the 4 by 6 MAC tile for 24 inner tiles, and serializes its 24 requantizations
through one multiplier. One real six-output tile matches all 384 projected
values exactly. Projection plus residual takes 2,874 busy cycles with
contiguous weight slices. The block controller advances output tiles and a
two-tile real-checkpoint test matches all 768 post-residual values exactly in
5,797 busy cycles with one deliberate gap per weight beat.

The projection weight tile buffer maps to 192 RAM32M16 distributed memories,
with no DSP, block RAM, UltraRAM, or flip-flop cost. The 64 by 768 Q13.10
residual canvas maps to 24 URAM288 primitives, one flip-flop, and no LUTs,
DSPs, or block RAMs. This avoids exceeding the K26 block RAM budget while the
18-bit attention canvas and 48-bank QKV scratchpad are live.

At the provisional and unvalidated 250 MHz target, sequential 12-head attention
including QKV scratch loads is analytically 0.920 ms per block. Scaling the
conservative two-output-tile projection measurement to all 128 output tiles is
1.484 ms per block. Together these implemented attention stages are about
2.404 ms per block, 28.85 ms per 12-block evaluation, or 230.8 ms across eight
evaluations. These are cycle-derived estimates, not board timings, and exclude
QKV projection, LayerNorm, MLP, final vocabulary projection, DMA contention,
and control overhead.

The folded INT8 projection weights occupy 589,824 bytes per block, half of
FP16. They are read once per block because each six-row weight tile is reused
across all 16 token groups. This is 7.08 MB per evaluation and 56.62 MB across
eight evaluations before transport framing.

## One physical attention array

The head and projection schedulers both support an external-array mode.
`attention_block_shared_mac_pipeline` phase-multiplexes their requests into one
4 by 6 by 32 `mixed_precision_mac_tile_pipelined` instance and routes responses
back only to the active phase. A connected reduced-head H0 test executes both
attention and the first six projection outputs through that one instance. It
matches every post-residual output integer exactly in 22,059 busy cycles and
records 704 attention requests followed by 384 projection requests. The test
also asserts that the two clients never request the array together.

This closes the duplicate-array fit problem in the intended architecture. The
reduced-head test zero-fills absent heads only to keep portable simulation time
reasonable. The complete 12-head attention canvas and complete projection
tiles remain independently checked against real checkpoint boundaries.

## QKV projection producer

The accepted QKV format keeps normalized activations in signed Q5.12 and uses
signed INT16 weights with one scale per output channel. A Q28 multiplier and
signed Q5.12 bias convert 48-bit accumulators back to signed Q5.12. Across all
12 H0 blocks, QKV projection has 0.0415% mean and 0.0718% maximum relative RMS
error. Chaining fixed LayerNorm, fixed QKV, rotary, QK, softmax, and
probability-times-V gives 0.1608% mean and 0.2252% maximum attention error.
No QKV output saturation occurs on the frozen trace.

`qkv_projection_output_tile_scheduler` computes six QKV outputs for all 64
tokens. The first version was exact in 1,721 cycles. Streaming consecutive
normalized-canvas responses directly into the MAC reduced it to 969 cycles,
a 43.7% reduction, while retaining exact output integers. Its six-row INT16
weight buffer maps to 384 RAM32M16 distributed memories and no DSPs, block RAM,
UltraRAM, or flip-flops.

`normalized_canvas_uram` stores four-token Q5.12 LayerNorm outputs in 32
channel banks. One read returns a complete 4 by 32 MAC activation tile. It maps
to 32 URAM288 primitives, 32 LUTs, one flip-flop, and no DSP or block RAM.
Together with the 24-URAM residual canvas, this uses 56 of the K26's 64
UltraRAM primitives. Each kind needs eleven six-output tiles to cover 64 head
channels, with four valid lanes in the last tile. The complete block therefore
contains 396 QKV tiles, not 384. At the standalone 969 cycles per tile, their
analytical total is 383,724 cycles, or 1.535 ms at the provisional and
unvalidated 250 MHz target.

`qkv_head_tile_controller` sequences all 33 Q, K, and V channel tiles for a
selected head and supplies checkpoint-global row addresses. The
`qkv_head_output_router` converts each four-token by six-channel tile into
scalar scratchpad writes and correctly suppresses the two invalid lanes in a
tail tile. Its open-source K26 screen maps to 281 LUTs and 470 flip-flops, with
no DSP or memory primitives.

## Rotary staging into the attention head

`qk_unrotated_scratchpad_banked` holds one head of unrotated Q and K in
distributed memory and provides both half-vector values for a rotary pair. It
maps to 384 RAM64M8 primitives, 1,661 LUTs, 1,164 flip-flops, and no block RAM,
UltraRAM, or DSPs. The shared 64 by 32 cosine and sine table maps to two
RAMB36E2 primitives.

`rotary_head_writeback_scheduler` requests one pair every two cycles, uses the
existing exact eight-DSP rotary unit, and serializes the two resulting channel
writes into the final attention scratchpad. All 4,096 block-0 head-0 Q/K
coordinates match exactly in 4,099 busy cycles. Mapping for the scheduler and
rotary arithmetic is 8 DSPs, 497 LUTs, and 455 flip-flops.

The attention scratchpad now has independent Q/K and V write addresses in
addition to independent write enables. This permits rotary Q/K writeback and V
projection writes on the same cycle. A connected fixed-QKV test stages
unrotated Q/K, loads projected V, performs the complete rotary writeback, and
then executes the real attention head. Every one of the 4,096 attention outputs
matches the cascaded fixed reference exactly, and attention execution remains
15,056 cycles.

`qkv_head_staging_pipeline` starts rotary only after the final K tile has
drained from the router, then overlaps it with all remaining V work.
`qkv_attention_head_pipeline` connects the full 33-tile projection, rotary,
scratchpad, and attention scheduler through one physical 192-lane MAC array.
The current real block-0 head-0 test verifies every routed Q, K, and V
coordinate and every one of the 4,096 final attention values exactly in 29,703
cycles. Twelve-head timing must be taken from the exhaustive multihead run
rather than multiplying this isolated-head result because head and canvas
control add their own drain cycles. Vendor timing remains unvalidated.

`qkv_attention_multihead_canvas_pipeline` reuses that head path across up to
12 heads and writes the existing 768-channel attention canvas. Its reduced
two-head integration test advances through both complete heads and reads back
all 64 channels for selected tokens in 83,700 control-model cycles. This test
uses a deterministic external response model, so its cycle count is sequencing
evidence rather than a physical-MAC performance result.

`qkv_attention_projection_block_pipeline` then gives the canvas to the folded
INT8 output projection and Q13.10 residual stage through the same external MAC
interface. A reduced two-head, two-output-tile integration emits all 32
expected block tiles in 89,261 control-model cycles. Real checkpoint numerics
are now continuous through one complete head and independently exact through
the output projection and residual boundaries.

The exhaustive packed multihead run now checks all 49,152 attention values,
covering 12 heads by 64 tokens by 64 channels, against the block-0 hardware
package. Every value matches exactly and the RTL testbench completes in 312,781
cycles. The original outer test expected an obsolete 521,000 through 524,000
cycle window and therefore reported a failure after the testbench had already
printed PASS. The bound is corrected to 311,000 through 315,000 and the exact
run is preserved in `h4-packed-all12-heads-h0-rtl.json`. This is Icarus
simulation evidence, not a board measurement or achieved clock result.

## Packed query-key precision decision

A propagated-logit ablation separates the two attention matrix products. Using
INT8 for both query-key and probability-value reaches only 93.75% masked-token
top-1 agreement over five fresh canvases. Keeping query-key at 18 bits while
quantizing probability-value reaches 95.0%, so probability-value is the more
sensitive product and remains on the existing 18-bit path.

The retained candidate quantizes only query-key. A single captured maximum per
head reaches 96.25% at INT8 and 96.875% at INT9. Dynamic scaling for each
64-element query and key vector with a Q17 reciprocal improves INT8 query-key
to 633 exact top-1 matches over 640 masked positions across 20 independent
canvases, or 98.90625%. The mean logit cosine similarity is 0.9999198 and the
maximum normalized logit RMSE is 0.016318. This passes the 98% aggregate
forward-pass gate. Q17 also fits the reciprocal in 24 bits, allowing each
18-by-24-bit quantization product to use one DSP48E2 instead of the two required
by the earlier Q24 prototype.

Dynamic vector scaling is compatible with the intended scheduler: rotary
staging collects a maximum for every query and key vector, replay quantizes each
vector to signed INT8, and the query and key scales form a Q28 multiplier for
each score. Two query rows can then share a DSP48E2 packed-product lane while
the probability-value phase reuses the existing 18-bit datapath. The next gate
is standalone bit-exact quantizer and packed-QK RTL, followed by complete
multi-step generated-text comparison. This numerical result is not FPGA timing
or end-to-end generation evidence.

The first standalone RTL gate is complete. Two 24-cycle iterative dividers track
query and key scales in parallel, and a 16-lane Q17 quantizer matches real
block-0 rotary vectors plus zero and signed 18-bit edge cases exactly. Separate
UltraScale+ technology maps sum to 1,767 LUT primitives, 604 flip-flops, and 16
DSP48E2s, or 1.51%, 0.26%, and 1.28% of the K26 resources. The superseded Q24
version used 32 DSP48E2s and 2,669 LUT primitives. These are Yosys technology
maps, not Vivado placement or timing results. The dynamic score multiplier and
packed query-key scheduler remain the next integration boundary.

That boundary is now bit exact in a connected group-pair scheduler. An exact
constant-reciprocal score multiplier and signed score requantizer each pass
20,008 randomized and edge-case vectors. The scheduler quantizes eight real
block-0 head-0 query rows, processes all 64 keys through the retained packed
array, and matches all 4,096 Q7.10 scores across the complete head. Loading the next key
tile during the current score postprocess reduces the exact pair schedule from
763 to 501 cycles. All eight connected group pairs complete in 4,008 QK cycles
per head versus the current measured 6,304, a measured 36.42% QK reduction.

At the same external-array boundary, the packed scheduler maps to 26,437 LUT
primitives, 10,570 flip-flops, and 40 DSP48E2s. The current fixed18 scheduler
maps to 18,747 LUTs and 11,988 flip-flops, so the provisional increment is 7,690
LUTs, 40 DSPs, and 1,418 fewer flip-flops. Local scale buffering already removed
6,571 LUTs from the first prototype. The retained array plus packed scheduler is
68.75% of K26 LUT capacity by summing separate Yosys maps. This is not a complete
block map, so the design remains accepted for integration but still subject to
the final capacity gate.

The eight-row to four-row boundary is also complete. A streaming pair buffer
forwards the lower score group directly and stores only the upper group. It maps
to 478 LUTs, 15 flip-flops, and 31 RAM32M16 primitives. The existing parallel
softmax and fixed18 PV scheduler then alternate with packed QK on the same
physical mixed-mode array. Holding the selected mode and operands while its
six-stage pipeline drains is required when changing between INT8 QK and fixed18
PV; without that hold, in-flight narrow reductions are overwritten.

Across all 64 block-0 head-0 query rows, the connected path matches 4,096 QK
scores, 4,096 Q16 probabilities, and 4,096 final Q12 attention values exactly.
QK plus softmax takes 7,272 cycles, and the complete path including fixed18 PV
takes 11,400 cycles. The current fixed18 head path takes 15,056 cycles, so the
new measured head reduction is 3,656 cycles or 24.28%. Replacing all 12 heads
would remove 43,872 block cycles, but that block-level delta remains pending
until the multihead canvas controller uses this path.

The reusable scratchpad-level packed head is now connected as well. It collects
dynamic Q/K scales during the 4,096 scalar writes, waits 26 cycles for the last
two iterative dividers, and then sequences all eight group pairs without test
bench intervention. All 4,096 final attention values match the packed software
reference exactly in 11,416 busy cycles. Against the 15,056-cycle fixed18 head
controller, this saves 3,640 cycles or 24.18% after including controller gaps.

`qkv_attention_head_pipeline_packed_m8` then connects real normalized H0 input,
the unchanged 33-tile QKV projection, rotary writeback, dynamic scaling, packed
QK, softmax, and fixed18 PV. It checks all 12,288 routed Q/K/V values and all
4,096 final attention values exactly in 26,063 cycles. The current fixed18
connected head requires 29,703 cycles, so the complete connected saving is
3,640 cycles or 12.25%.

The array boundary is now externalized without changing those results. The
group-pair engine, scratchpad-level head, and complete QKV-to-attention head
each pass again when QKV projection, packed QK, and fixed18 PV use one externally
owned mixed-mode array. Mode and operands remain held while the six-stage array
drains, and the real connected head remains exact at 26,063 cycles. This removes
the duplicate-array caveat at head scope.

`qkv_attention_multihead_canvas_pipeline_packed_m8` sequences that external
interface across heads. Its two-head zero-data control model completes in
48,567 cycles while issuing 352 narrow and 26,048 wide requests and successfully
reading both head canvases. Connecting the existing output projection gives
49,744 cycles versus 55,968 for the fixed18 control path, an 11.12% reduction.
These control-model cycles use an immediate deterministic response model and
are sequencing evidence, not physical-array performance measurements.

The same interface now reaches `ddit_block_pipeline`, its one shared physical
mixed-mode array, the parameter and image fabrics, the KV260 kernel core, and
the board-facing AXI top. A reduced complete DDiT run with one head, two
attention output tiles, and one MLP output completes in 40,785 cycles versus
44,425 for fixed18. Both paths produce the same zero outputs and identical MLP
request counts. The AXI top selects packed attention by default and passes its
elaboration, register-interface, and kernel-hierarchy tests. Exhaustive 12-head
H0 numerics are now exact.

The refreshed complete UltraScale+ technology map includes the synchronous MLP
activation memory and residual-bank reduction. The board-facing M8 top uses
236,111 CLB-LUT-equivalent primitives, 165 BRAM36-equivalent blocks, 57 URAMs,
951 DSP48E2s, and 153,292 flip-flops. Relative to the documented K26 capacities,
those counts are 201.60% LUT, 114.58% BRAM, 89.06% URAM, 76.20% DSP, and 65.44%
flip-flops. This is a 63,999-LUT-equivalent or 21.33% complete-top reduction
from the first 300,110-LUT map, but the M8 build still does not fit. The KV260
profile therefore needs narrower MLP scheduling, compute folding, and a memory
rebalance that removes at least 21 BRAM36-equivalent blocks. Vivado placement,
timing, power, and board behavior remain open gates.

The selected KV260 M4 profile now has its own complete board-top map: 197,208
CLB-LUT-equivalent primitives, 162.5 BRAM36-equivalent blocks, 43 URAMs, 936
DSP48E2s, and 136,004 flip-flops. This is 16.48% fewer LUT-equivalents than the
M8 top and 34.29% fewer than the first complete map. It still requires a 40.6%
LUT reduction and 18.5 fewer BRAM36-equivalent blocks to reach raw K26 capacity,
before placement headroom. The M4 profile is therefore a measured step toward
fit, not yet a fit claim.

Plain per-output INT8 QKV weights were rejected because their mean QKV and
attention relative RMS errors are 3.37% and 8.44%. Channel equalization with
SmoothQuant alpha 0.25 improves those means to 0.525% and 1.12% with no
activation saturation. When propagated through all 12 blocks on five fresh
half-masked canvases, it produces 156 of 160 matching top-1 tokens, or 97.5%.
That is below the 98% gate, so the current RTL retains INT16 QKV. Equalized INT8
remains a candidate for quantization-aware fine-tuning, not an implementation
assumption.

## Resource-directed MLP storage

The first complete map identified the M8 MLP control and storage path as the
largest non-array consumer at 143,251 CLB-LUT-equivalent primitives. Two
structural changes preserve the arithmetic and request rate. The 64-token by
3,072-channel down activation buffer now uses a one-cycle synchronous banked
memory, which maps to exactly 29 URAMs and one flip-flop with no LUTs in its
standalone 2,048-bit by 768-word configuration. The residual buffer now keeps
only the current and prefetched output-tile parity banks instead of all 128
tiles, a 64-fold depth reduction that preserves the existing ping-pong load.

The current focused M8 MLP map is 80,941 CLB-LUT-equivalent primitives, 47
BRAM36-equivalent blocks, 29 URAMs, 58 DSPs, and 45,442 flip-flops. This is a
62,310-LUT-equivalent or 43.5% reduction from the prior M8 map. The reduced
complete packed DDiT simulation remains bit exact at 40,785 cycles, so the
synchronous memory did not reduce steady-state issue throughput.

An explicit M4 design point uses 44,161 CLB-LUT-equivalent primitives and 15
URAMs. It saves another 36,780 LUT-equivalents but gives up the previously
measured 36.5% MLP cycle reduction from paired-token M8 execution. The refreshed
full map confirms that M8 is the larger-device performance profile. M4 is now
the selected KV260 fit direction. Its full board-top map and reduced exact RTL
validation are complete; full-shape timing and board validation remain open.
