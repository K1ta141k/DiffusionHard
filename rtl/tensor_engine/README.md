# Fixed-shape INT8 tensor tile

`int8_mac_tile.sv` is the H1 compute primitive for the 64-token MDLM graph. The
default configuration contains 4 token lanes, 8 output-channel lanes, and 32
inner-product lanes, for 1,024 signed INT8 MACs per accepted cycle and 32
independent 32-bit accumulators.

The surrounding scheduler will retain the 64 by 768 activation canvas on chip,
load one 8 by 32 weight tile, reuse that tile across sixteen groups of four
tokens, and retain only 64 by 8 accumulators for the current output-channel
tile. This loop order reads every dense weight once per model evaluation without
materializing the 64 by 3,072 accumulator matrix.

`clear_accumulators` marks the first inner tile. `last_k_tile` marks the final
inner tile and produces `valid_out`. The module deliberately excludes the SRAM,
DDR, requantization, and GELU stages so the MAC array can be validated and mapped
independently before H1 integration.

Run the self-checking reduced-lane simulation on macOS:

```bash
iverilog -g2012 -Wall \
  -s tb_int8_mac_tile \
  -o /tmp/tb_int8_mac_tile \
  rtl/tensor_engine/int8_mac_tile.sv \
  rtl/tensor_engine/tb_int8_mac_tile.sv
vvp /tmp/tb_int8_mac_tile
```

`int8_mac_tile_pipelined.sv` is the timing-oriented 768-lane commit candidate.
It fixes the inner dimension at 32, registers the signed products, and reduces
them through five balanced registered adder levels. Control metadata follows
the six-stage product/reduction pipe. A separate result register protects a
completed token group while the accumulator immediately starts the next one.
The array accepts one inner tile per cycle and has six cycles of input-to-
accumulator latency.

```bash
iverilog -g2012 -Wall \
  -s tb_int8_mac_tile_pipelined \
  -o /tmp/tb_int8_mac_tile_pipelined \
  rtl/tensor_engine/int8_mac_tile_pipelined.sv \
  rtl/tensor_engine/tb_int8_mac_tile_pipelined.sv
vvp /tmp/tb_int8_mac_tile_pipelined
```

`mlp_tile_controller.sv` adds the fixed-shape storage and traversal around the
pipelined core. Activations are addressed by token group and 32-channel inner
tile. The current output-channel slice stores all inner tiles, so each weight is
reused by all token groups before the next six output channels are loaded. For
the real graph, its two supported parameterizations are:

| Layer | Activation store | Weight-slice buffer | Compute cycles per slice |
| --- | ---: | ---: | ---: |
| 768 to 3,072 | 49,152 bytes | 4,608 bytes | 384 |
| 3,072 to 768 | 196,608 bytes | 18,432 bytes | 1,536 |

The packed tile arrays express the required logical SRAM banking portably. A
vendor-specific memory wrapper remains needed to prove BRAM/URAM inference.

```bash
iverilog -g2012 -Wall \
  -s tb_mlp_tile_controller \
  -o /tmp/tb_mlp_tile_controller \
  rtl/tensor_engine/int8_mac_tile_pipelined.sv \
  rtl/tensor_engine/mlp_tile_controller.sv \
  rtl/tensor_engine/tb_mlp_tile_controller.sv
vvp /tmp/tb_mlp_tile_controller
```

`mlp_tile_pingpong_controller.sv` adds two full weight-slice banks. The inactive
bank may load during compute, while the active bank remains locked until its
last tagged result exits the MAC pipeline. `mlp_up_pingpong_pipeline.sv` adds
banked multiplier and bias metadata plus the active serialized requantization
and GELU path. Its tag contains bank, output tile, and token group.

At the modeled 12.48 GB/s effective DDR rate, up loading takes 96 cycles for
384 cycles of compute and down loading takes 373 cycles for 1,536 cycles of
compute. The ping-pong buffers are 9,216 bytes for up and 36,864 bytes for down.
The analytical overlap model reports 36.782976 ms saved across twelve blocks
and eight evaluations versus sequential load then compute issue.

`fixed_requantize.sv` converts each 32-bit accumulator with an unsigned fixed
multiplier, symmetric round-to-nearest, signed bias, and output saturation. Its
width and binary shift are parameters. The MLP-up candidate produces signed
Q5.10 values for GELU. Later MLP-down outputs require a wider destination
because the real frozen checkpoint exceeds the Q5.10 range after block 4.

`gelu_q10_lut.sv` implements the checkpoint's tanh-approximate GELU with a
1,024 by 16-bit ROM over -8 to +8 at 1/64 spacing. Inputs below -8 return zero;
positive inputs at or above +8 pass through. The generated ROM is 2,048 bytes.
Exhaustive checking of every Q5.10 input in the lookup interval bounds absolute
error below 0.019. The generator has a `--check` mode so CI detects stale ROM
contents.

The active integrated implementation serializes the 24 result lanes because a
new MLP-up result group arrives once every 24 inner-tile cycles. It uses one
requantizer and one synchronous GELU ROM port, with an active vector and one
pending vector to sustain one lane per cycle. Open-source mapping gives 4 DSPs,
5,210 LUT primitives, 4,830 flip-flops, and one RAMB18. The rejected fully
parallel version used 96 DSPs and 8,309 LUT primitives.

The current serializer factorizes each multiplier into a runtime unsigned Q18
token factor and a frozen unsigned Q20 output factor. One extra DSP forms their
product while the lane is processed. This removes duplicated per-group
multiplier vectors. The six-channel interstage multiplier slice follows the
same queued vector as a protected sideband. With that sideband included, the
complete active postprocessor maps to 5 DSPs, 3,957 LUT primitives, 3,883
flip-flops, and one RAMB18. The reduced ping-pong test uses distinct token
factors for every group and distinct output factors and sidebands in both
weight banks.

`mlp_down_pingpong_pipeline.sv` reuses the tagged ping-pong controller, converts
accumulators to signed Q13.10 with a serialized requantizer, selects the matching
residual tile, and performs a saturating 24-bit add. The serializer maps to 4
DSPs, 7,135 LUT primitives, and 4,902 flip-flops. The parallel residual adder
maps to 1,392 LUT primitives and 593 flip-flops.

`weight_slice_stream_adapter.sv` assembles three continuous 512-bit stream beats
into each 1,536-bit 6 by 32 weight tile. It propagates bank backpressure on the
third beat, numbers inner tiles, checks the final slice marker, and has no
adapter bubble between ready tiles. It remains the continuous-slice adapter for
streaming sources.
The fixed three-beat mapping uses 18 LUT primitives and 1,035 flip-flops. A
general variable-insertion version was rejected after mapping to 20,188 LUTs.

`smoothquant_int8_vector_serial.sv` converts the Q5.10 GELU stream into the
frozen SmoothQuant INT8 representation used by MLP down. The default 24-lane
vector is serialized over 24 cycles with one pending-vector slot. It preserves
tags, rounds positive and negative values symmetrically, saturates to -127
through +127, and maps to one DSP48E2, 2,326 LUT primitives, and 2,168
flip-flops in the open-source K26 screen.

`mlp_interstage_tile_bridge_bram.sv` repacks each 4 by 6 converted vector into
4 by 32 activation tiles for the down controller. Two synchronous 32-bit banks
hold the 2,048-byte staging state. Six channel writes fit between converter
outputs, and a completed tile is read two channels per cycle. The output
assembler is specialized to sixteen fixed channel-pair destinations and the
only three boundary offsets reachable when six-channel chunks cover 3,072
channels. It maps to 2,314 LUT primitives, 2,488 flip-flops, two RAMB18 blocks,
and no DSP.

The rejected `mlp_interstage_tile_bridge.sv` is retained as the wide-register
baseline. It mapped to 142,058 LUT primitives because variable writes flattened
the store and output selection into mux fabric. The first banked rewrite still
used 44,233 LUT primitives until the fixed destinations and boundary cases were
made explicit. The active bridge is 61.4 times smaller in LUT primitives than
the original and 19.1 times smaller than that first banked rewrite.

`mlp_interstage_pipeline.sv` connects both modules. Its cycle test verifies
backpressure, tag order, two independent token groups, and two boundary-crossing
K tiles. Vendor place-and-route and timing validation remain open.

`mlp_up_to_down_activation_pipeline.sv` structurally connects the full up
ping-pong engine to the interstage converter and tile bridge. Interstage
multipliers are captured with each weight bank and travel beside the GELU
vector, so reloading a released bank cannot alter an in-flight conversion. The
default fixed-shape top passes full RTL elaboration.

`unsigned_divider_iterative.sv` is a 26-cycle restoring divider used during
runtime activation-scale setup. `mlp_up_activation_quantizer.sv` uses four
instances across two phases. It transforms signed Q5.12 normalized values with
static Q3.15 SmoothQuant reciprocals, finds four token maxima, generates Q18
token factors, replays the input, and emits 4 by 32 INT8 activation tiles. It
maps to 8 DSPs, 1,858 LUT primitives, and 2,899 flip-flops with no RAM.

`mlp_quantized_up_to_down_pipeline.sv` wires those activation tiles and token
factors directly into the up-to-down top. The default 64-token model shape
passes RTL elaboration.

`layer_norm_q12_group.sv` is the fixed unaffine LayerNorm producer. It consumes
four Q13.10 residual tokens per channel, computes population moments, an
integer square root, and a Q18 inverse standard deviation, then supports
multiple Q5.12 output replays. Its block-0 checkpoint test matches all 3,072
integer outputs exactly. Open-source K26 mapping uses 24 DSPs, 4,420 LUT
primitives, and 2,860 flip-flops.

`layer_norm_mlp_up_activation_frontend.sv` automatically schedules the
statistics pass and the two replays required by the runtime activation
quantizer. Its connected checkpoint test matches every activation byte and
token factor. It maps to 32 DSPs, 6,307 LUT primitives, and 5,784 flip-flops.
`mlp_residual_up_to_down_pipeline.sv` connects this residual-input front end to
the complete MLP up-to-down path and passes default-shape RTL elaboration.

`mlp_shared_up_down_pipeline.sv` disables the two internal MAC instances and
routes both MLP phases through one external tagged array. An owner tag bit
returns delayed responses to the correct phase. Its reduced exact test observes
3,072 up requests followed by 48 down requests through one physical MAC.

`hidden_canvas_group_replay.sv` reads one six-channel hidden tile at a time and
serializes its channels for LayerNorm. `hidden_canvas_mlp_frontend.sv` sequences
the statistics pass and both quantizer replays. The connected reduced test
performs 384 hidden-canvas reads, emits all activation tiles and token factors,
then drives the shared MLP datapath.

`mlp_tile_load_sequencer.sv` accepts a tile command and does not report
completion until all fixed 32-channel weight chunks and the matching metadata
have reached the selected ping-pong bank. Weight and metadata streams stall
independently. `hidden_canvas_residual_load_sequencer.sv` performs the analogous
all-group replay for each MLP-down residual tile.

`mlp_block_controller.sv` automatically sequences all norm2 groups, every MLP
up output tile, interstage drain, and every MLP down output tile. It preloads
the alternate bank while the current tile computes. The fully connected
`hidden_canvas_automatic_mlp_block.sv` uses one start pulse and one external
physical MAC. Its four-token reduced-shape test completes 128 up tiles and two
down tiles in 11,372 total simulated cycles, with 3,120 exact shared-array
requests. This is structural and deterministic evidence, not a full K26 timing
result.

The opt-in `test_hidden_canvas_automatic_mlp_block_h0_rtl.py` gate uses a real
block-0 hidden state and channel-addressed SmoothQuant reciprocal table. It
runs four tokens through all 512 up tiles and the first down tile, observes
12,384 physical-array requests, and matches all 24 post-residual integers in
34,881 simulated cycles. The default suite skips this 32-second gate unless
`DIFFUSION_ACCEL_RUN_REAL_MLP_RTL=1` is set.

Setting `DIFFUSION_ACCEL_RUN_FULL_MLP_RTL=1` promotes the same fixture to the
production shape. It runs 64 real tokens through all 512 up tiles, all 3,072
interstage channels, and all 128 down tiles using one physical INT8 MAC array.
The test observes 393,216 array requests and 2,048 output vectors in 471,939
simulated cycles, then matches all 49,152 post-residual integers exactly. Two
consecutive macOS runs passed in 572.54 and 574.10 host seconds. Set
`DIFFUSION_ACCEL_SHOW_LONG_RTL_OUTPUT=1` to print the internal counters.

`ddit_block_shared_mac.sv` is the H4 cross-precision array boundary. Attention
requests retain signed 18-bit operands and 48-bit accumulators. MLP INT8
operands are widened into the same physical multiplier inputs, and each 48-bit
result lane is narrowed to its exact signed 32-bit MLP value. An owner bit is
carried in the physical tag so an attention result remains correctly routed if
the controller changes to MLP phase before that result emerges.

`ddit_block_pipeline.sv` sequences the connected QKV, rotary, attention,
folded projection, in-place residual canvas, norm2, and automatic MLP phases.
Its opt-in reduced test uses one physical `mixed_precision_mac_tile_pipelined`,
computes one attention head and two projection tiles, then runs four tokens
through 128 up tiles and one down tile. It emits exact zero-reference outputs
in 66,408 simulated cycles. The eleven intentionally omitted attention heads
are zero-padded in the testbench only. Set
`DIFFUSION_ACCEL_RUN_DDIT_BLOCK_RTL=1` to run this roughly one-minute gate.

`mdlm_block_parameter_address_generator.sv` implements the committed block-0
execution-image offsets and record geometry. `axi512_read_burst_master.sv`
issues one tagged AXI4 read at a time, propagates read backpressure, checks AXI
responses and `RLAST`, and counts bytes plus address and data stalls.
`mdlm_block_parameter_dma.sv` connects the two and handles invalid records
without touching AXI.

`fixed_weight_record_adapter.sv` converts one returned record into either a
3,072-bit INT16 QKV tile or a 1,536-bit INT8 tile while retaining it across
compute-side stalls. `mdlm_weight_slice_dma.sv` walks every input tile for one
model output tile and drives the existing weight loader interface. A real H0
test reads the first 24 QKV records from the committed block image and matches
all 24 packed loader words against independently recomputed checkpoint weights.
The transaction master is a correctness-first, one-outstanding design. Vendor
memory integration and multi-outstanding bandwidth tuning remain open.

`fixed_aligned_record_adapter.sv` consumes the aligned QKV, projection, and MLP
metadata records. `compact_table_record_adapter.sv` selects one rotary or
SmoothQuant entry from a cache line shared by sixteen four-byte slots.
`mdlm_parameter_record_dma.sv` provides a checked single-record interface for
either geometry. Real checkpoint tests match one QKV metadata record and one
compact MLP reciprocal against independently recomputed tensor values.

`mdlm_qkv_output_tile_loader.sv` combines metadata and weight fetches behind one
AXI master. A command names the head, Q/K/V kind, and six-channel output tile;
the loader emits the exact metadata followed by 24 input weight tiles on the
ports already consumed by `qkv_head_staging_pipeline.sv`. Its real block-0 test
transfers 9,280 bytes and matches every emitted bit under consumer stalls.

`mdlm_projection_output_tile_loader.sv` is the folded INT8 counterpart. It
fetches one multiplier record plus 24 weight records through one master and
transfers 4,672 bytes per output tile. Real connected tests now drive both the
QKV and projection loaders into their physical MAC schedulers. All 64 by 6
outputs match the block-0 references exactly, so neither testbench injects
metadata or packed weights at the compute boundary.

`mdlm_mlp_output_tile_loader.sv` serves both MLP phases. The up configuration
loads one 444-bit metadata item and 24 weight tiles. The down configuration
loads one 1,344-bit metadata item and 96 weight tiles. Their real tests transfer
4,672 and 18,624 bytes respectively and compare every metadata and weight bit
against independently rebuilt block-0 values.

`axi512_read_arbiter_4.sv` holds a round-robin client grant for a complete AXI
burst and routes response backpressure only to that client. It independently
checks response framing and counts transactions. The contention test covers
all four clients and an injected late-last response.
`mdlm_block_parameter_load_fabric.sv` instantiates all four operator loaders
behind this arbiter, so the production-shape parameter path exposes one
physical 512-bit AXI read port plus per-client byte counters.

The first K26-family Yosys screen exposed five accidental DSP48E2 uses in fixed
record-index multipliers. Model-specific shift-and-add strides remove all five.
The resulting structural screen uses 11,462 flip-flops, 370 carry cells, and
1,633 LUT1 through LUT6 primitives. The top-level pad counts are deliberately
excluded because the fabric's wide compute interfaces were synthesized as chip
pins. This is not a Vivado timing or board-bandwidth result.

`ddit_block_with_parameter_fabric.sv` drives those four clients from explicit
compute-controller request signals. Its reduced connected test performs 4,100
reads and transfers 918,400 bytes through one AXI port, then matches the exact
attention and final zero references in 93,889 simulated cycles. Rotary and
reciprocal table preload remain a separate open integration boundary.

`mdlm_block_constant_preloader.sv` now implements that boundary. It serializes
128 rotary cache lines into 2,048 scratchpad writes and stores 48 reciprocal
cache lines as 768 channel-addressed Q3.15 values. The real block-0 test matches
all values exactly after 176 reads and 11,264 transferred bytes. Arbitration
with the dense-parameter port is implemented by
`ddit_block_with_image_fabric.sv`. Its reduced functional test uses one external
AXI read port for both constant and dense traffic, gates block launch on preload,
and accounts for 4,276 reads and 929,664 bytes exactly. Full-shape real block-0
validation now passes through the same top: 1,584,457 cycles, 38,492 reads,
9,060,096 bytes, 2,048 exact attention vectors, and 2,048 exact final vectors.
`hidden_canvas_norm1_precompute.sv` now closes that boundary with two residual
replays and a 384-tile Q12 buffer. Its real embedding test is bit exact, and a
reduced image-fabric run passes with no external normalized data in 140,301
cycles. The full-shape run also holds the legacy normalized input at zero, so
the block is autonomous from residual Q10 plus its execution image.

`attention_canvas_grouped_scratchpad_banked.sv` stores each four-token group in
one word and lets output projection stream one 32-channel K tile per cycle.
`fixed_requantize_vector_parallel` processes four lanes per cycle for QKV and
attention projection. Real two-tile block-0 projection is bit
exact in 1,413 cycles versus 5,797 for the original schedule. The reduced
autonomous image-fabric block now takes 126,393 cycles versus 140,301. The
grouped canvas maps to 64 RAMB36E2 primitives. The two retained production
requantizers add 24 DSPs against their serial versions. MLP-up and MLP-down
parallel postprocess prototypes remain uninstantiated. MLP-up saved only 18
reduced-block cycles, and MLP-down saved only 18 full-block cycles while adding
12 DSPs.

`axi512_residual_canvas_reader.sv` and `axi512_output_canvas_writer.sv` provide
the DDR edge of the board kernel using padded 128-byte records for each 576-bit
canvas vector. `ddit_block_kv260_kernel_core.sv` sequences residual load,
constant preload, autonomous compute, and output drain behind one AXI memory
master. Residual and constant loading overlap, and the functional test proves
both clients are busy together. `ddit_block_kv260_axi_top.sv` adds a tested
AXI-Lite register shell.
The portable checks cover both DMAs, control-register behavior, and production
top elaboration. Vivado synthesis and KV260 execution remain open.

`rotary_qk_pair_serial.sv` implements the checkpoint's half-vector rotary
convention for one token and one head pair per cycle. It uses signed Q5.12 Q/K
and Q1.15 cosine and sine. A real block-0 test checks 1,536 pair records
exactly. Mapping uses 8 DSPs, 416 LUT primitives, and 376 flip-flops.

`attention_softmax_row_q16.sv` implements stable 64-element softmax with one
score-row store, a synchronous 1,025-entry exponential ROM, one iterative
reciprocal division, and one Q16 probability per output cycle. Its real H0 row
test checks every probability exactly. Mapping uses one DSP, one RAMB36, 388
LUT primitives, and 300 flip-flops.

`attention_head_scratchpad_banked.sv` provides the 16-bank Q, K, and transposed
V layouts. `attention_group_pipeline.sv` connects QK, four streamed softmax
rows, and probability-times-V through one shared MAC. The group matches 256
real checkpoint outputs exactly in 939 cycles. `attention_head_pipeline.sv`
matches a complete 4,096-value head in 15,056 execution cycles.

`attention_multihead_canvas_pipeline.sv` reuses the head engine across 12 heads
and writes `attention_canvas_scratchpad_banked.sv`. The optimized canvas maps to
64 RAMB18E2 primitives, 177 LUTs, 452 flip-flops, and no DSPs. Its earlier
dynamic write router used 8,449 LUTs and 5 DSPs.

`attention_projection_output_tile_scheduler.sv` consumes the 768-channel
canvas, per-output INT8 folded weights, and Q24 requant multipliers. It matches
one complete six-output tile exactly in 2,873 projection cycles.
`attention_projection_residual_output_tile.sv` adds a 24-bank UltraRAM residual
canvas and produces Q13.10 block values in 2,874 cycles. The residual canvas
maps to 24 URAM288 primitives and the six-row weight buffer maps to 192
distributed RAM primitives. `attention_projection_block_pipeline.sv` sequences
all 128 six-output tiles. Its two-tile checkpoint test matches every emitted
post-residual integer exactly.

`mixed_precision_mac_tile_pipelined.sv` reuses one registered 18-bit physical
multiplier tree for fixed QK, Q16 probability by Q5.12 V, and sign-extended
INT8 MLP products. Its test covers all three modes. The 192-lane mapping uses
192 DSPs, 9,671 LUT primitives, and 14,601 flip-flops. A linear four-token
projection is 768 DSPs, 38,684 LUTs, and 58,404 flip-flops, subject to the same
projection and vendor-timing caveats as the INT8 screen.

The standalone wrappers retain an internal-MAC mode for focused tests.
`attention_block_shared_mac_pipeline.sv` selects their external-MAC modes and
phase-multiplexes attention and projection through one physical array. Its H0
test matches post-residual integers exactly and observes 704 attention requests
followed by 384 projection requests with no overlap.

`qkv_projection_output_tile_scheduler.sv` uses the same array shape for signed
Q5.12 activations and per-output INT16 weights. Consecutive normalized-canvas
responses issue one MAC request per cycle. A real H0 six-output tile matches all
384 QKV integers exactly in 969 cycles. `qkv_weight_tile_buffer.sv` maps to 384
distributed RAM primitives. `normalized_canvas_uram.sv` returns one 4 by 32
activation tile per read and maps to 32 URAM288 primitives.

`qk_unrotated_scratchpad_banked.sv` and
`rotary_head_writeback_scheduler.sv` connect projected Q/K to the final head
scratchpad. The transient Q/K store maps to 384 distributed RAM primitives.
The scheduler issues one pair every two cycles and writes all 4,096 rotated Q/K
coordinates in 4,099 cycles. A connected H0 test then runs the complete
attention head from fixed QKV and matches every attention integer exactly.

`qkv_head_tile_controller.sv` sequences the 33 Q/K/V output tiles for one
head. `qkv_head_output_router.sv` serializes each tile into scalar Q, K, or V
writes and suppresses invalid tail lanes. `qkv_head_staging_pipeline.sv`
connects projection to the unrotated scratchpad, starts rotary after the final
K tile drains, and overlaps rotary with V projection. A parallel router
prototype maps to an estimated 874 logic cells, but its extra scratchpad muxes
expose only 20 saved head cycles because the scalar drain already overlaps the
next group computation. The scalar router remains the production choice.

`qkv_projection_output_tile_scheduler_streaming.sv` continuously issues all
sixteen input groups, propagates group tags through the MAC, overlaps four-lane
requantization, and uses a two-entry output FIFO. A real block-0 tile falls from
683 to 428 cycles. The complete connected real head falls from 37,623 to 29,703
cycles while retaining the scalar router and the same scheduler DSP count.

`qkv_attention_head_pipeline.sv` uses separate Q/K and V write addresses and
phase-multiplexes QKV, QK, and PV requests through one external physical MAC
array. Its full 33-tile, rotary, and attention control test emits all 4,096
head values in 29,703 cycles with one 192-lane array instance, down from 47,127
cycles before four-lane requantization and continuous QKV scheduling. A real
block-0 head-0 test also verifies
every routed Q, K, and V coordinate and every final attention value exactly
through that connected path.

`qkv_attention_multihead_canvas_pipeline.sv` reuses that path across up to 12
heads and writes `attention_canvas_scratchpad_banked.sv`. A two-head control
test completes both heads and reads back selected 64-channel canvas rows.
`qkv_attention_projection_block_pipeline.sv` hands the completed canvas to the
folded output projection and residual stage while retaining one external MAC
interface. Its reduced two-head, two-output-tile test emits all 32 expected
post-residual tiles.

The opt-in full-checkpoint gates are:

```bash
DIFFUSION_ACCEL_RUN_LONG_RTL=1 \
  .venv/bin/python -m pytest -q -s \
  tests/test_qkv_attention_multihead_h0_connected_rtl.py

DIFFUSION_ACCEL_RUN_FULL_PROJECTION_RTL=1 \
  .venv/bin/python -m pytest -q -s \
  tests/test_attention_projection_output_tile_h0_rtl.py
```

They compare all 49,152 block-0 canvas integers and all 49,152 folded
projection-plus-residual integers. The recorded runs and limitations are in
`data/results/h3-full-attention-block0-rtl-validation.json`.

```bash
.venv/bin/python rtl/tensor_engine/generate_gelu_q10_lut.py \
  --out rtl/tensor_engine/gelu_q10_lut.hex \
  --check

.venv/bin/python -m pytest -q tests/test_fixed_postprocess_rtl.py
```

Open-source synthesis is a pre-place-and-route resource screen. It does not
establish the final K26 frequency, routing, power, or DSP packing.

The 192-lane quarter-width pipelined screen maps to 192 DSP48E2 primitives,
3,519 LUT primitives, and 6,849 flip-flop primitives. Multiplying those counts
by four projects 768 DSPs, 14,076 LUTs, and 27,396 flip-flops for the active
shape. That projection is useful for architecture selection but is not reported
as a completed full-width synthesis result. The exact full-width Yosys attempt
was stopped after 30 minutes without producing a report.
