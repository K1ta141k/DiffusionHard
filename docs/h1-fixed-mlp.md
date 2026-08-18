# H1 fixed-shape MDLM MLP

Status: in progress. Real-checkpoint integer validation, a registered 768-lane
MAC pipeline, and a functional activation/weight traversal controller are
complete. Portable RTL now integrates ping-pong weight slices, banked requant
metadata, serialized requantization, GELU, MLP-down requantization, and the
saturating residual path. Up and down now share one external physical MAC
interface. A block controller automatically schedules norm2, all up tiles,
the interstage stream, all down tiles, and residual replay. A 512-bit
weight-stream adapter and the complete GELU-to-down INT8 interstage stream are
also functional. A two-pass runtime quantizer now produces up INT8 activation
tiles and Q18 token factors from signed Q5.12 normalized values.
The complete 64-token by 768-output automatic block-0 checkpoint test now
passes. Vendor-specific SRAM inference, an AXI address/burst master, broader
captured and randomized-state validation, and timing closure remain open.

## Fixed workload

Every one of the twelve MDLM blocks contains the same two matrix shapes:

- MLP up: 64 by 768 multiplied by 768 by 3,072;
- MLP down: 64 by 3,072 multiplied by 3,072 by 768.

Together they require 301,989,888 MACs per block and 3,623,878,656 MACs per
model evaluation. This is about 45 percent of the complete model arithmetic.

At the active 768-lane commit point and 250 MHz, the compute-only lower bound is
393,216 cycles or 1.572864 ms per block. All twelve MLPs have a lower bound of
18.874368 ms per model evaluation. The INT8 MLP matrices contain about 56.6 MB
across all blocks, with a weight-streaming lower bound near 4.54 ms at the
modeled 12.48 GB/s effective DDR rate.

## Real block-0 precision sweep

The sweep uses `folded.block_00.norm2_unaffine` and the exported folded weights
from H0. Integer matrix products use exact wide accumulation. Activations use
either one scale per tensor or one scale per token. Weights use one scale per
output channel.

| Design | SmoothQuant alpha | Block-output relative RMS | Cosine similarity |
| --- | ---: | ---: | ---: |
| A8W8, tensor scale | none | 12.9448% | 0.991851 |
| A8W8, token scale | none | 9.9033% | 0.995154 |
| A8W8, token scale | 0.50 | 1.2984% | 0.999922 |
| A8W8, token scale | 0.75 | 1.2537% | 0.999927 |
| A8W16, token scale | none | 9.1851% | 0.995852 |
| A16W8, token scale | none | 4.0017% | 0.999206 |
| A16W16, token scale | none | 0.0434% | 1.000000 |

The current best W8A8 point is token scaling with alpha 0.75. It reduces the
relative RMS error by about 10.3 times versus naive tensor-scale W8A8. This is a
single-block, single-input result and does not yet pass the full-model quality
gate. It establishes SmoothQuant-style channel folding as the first sprint-mode
precision candidate while A16W16 remains the reference fixed-point mode.

The measured W8A8 accumulators required 19 signed bits for MLP up and 18 signed
bits for MLP down. The RTL retains 32-bit accumulators until calibration covers
all blocks and noise states.

The same five-alpha W8A8 search now covers all twelve blocks on the frozen H0
input. Every block selected either 0.50 or 0.75. Best per-block relative RMS
error ranges from 0.5711% to 1.2537%, with a 1.1424% median and 1.0663% mean.
This table can be baked into the fixed-model exporter, but it remains a
single-input calibration result rather than a generation-quality result.

## Tiled compute architecture

The active `int8_mac_tile_pipelined` has:

- four token lanes;
- six output-channel lanes;
- thirty-two inner-product lanes;
- 768 signed INT8 multipliers per accepted cycle;
- twenty-four independent 32-bit accumulators;
- registered products and five registered balanced-reduction levels;
- a six-cycle input-to-accumulator pipeline that accepts one tile per cycle.

The scheduler loop order is output tile, token group, then inner tile. One full
6 by K output-channel slice is loaded, then reused while sixteen groups of four
tokens traverse every 32-channel inner tile. Only the current 4 by 6 partial
sums stay in the compute core. This makes weight bandwidth scale with matrix
size rather than token count without requiring sixteen accumulator banks.

For MLP up, the controller stores a 49,152-byte activation canvas and a
4,608-byte weight slice. Each output slice takes 384 accepted compute cycles.
For MLP down, the corresponding sizes are 196,608 and 18,432 bytes, and each
slice takes 1,536 cycles. The packed RTL arrays describe the required logical
banking, but BRAM/URAM inference and a physical DMA-width adapter have not yet
been proven with vendor tools.

`mlp_tile_pingpong_controller` keeps two complete weight slices. A bank remains
locked from launch until its final tagged result leaves the six-stage MAC pipe.
The other bank can load concurrently. Bank ID, output-tile ID, and token-group
ID travel with every accumulator result, so a following tile can enter the MAC
pipe before the previous tile completes post-processing without mixing weights
or scales. The integrated reduced-shape test overlaps bank-1 loading with bank-0
compute and verifies both result streams.

At 12.48 GB/s effective DDR bandwidth and 250 MHz, an up weight slice plus
requant metadata takes 96 modeled load cycles versus 384 compute cycles. A down
slice takes 373 load cycles versus 1,536 compute cycles. Both have about four
times enough compute time to hide the next load. The two weight buffers occupy
9,216 bytes for up and 36,864 bytes for down. Relative to sequential load then
compute issue, ping-pong scheduling saves 95,789 cycles or 0.383156 ms per
block. Across twelve blocks and eight evaluations, the modeled saving is
36.782976 ms. This excludes AXI setup, bank conflicts, final drain, and vendor
timing.

## Fixed post-processing

The accumulator post-processor now uses a per-lane unsigned multiplier,
symmetric round-to-nearest, a signed fixed-point bias, and saturation. MLP-up
targets signed Q5.10. The GELU stage is a 1,024-entry, 2,048-byte Q5.10 ROM over
-8 to +8 with 1/64 input spacing. Values at or below -8 become zero, while
values at or above +8 pass through. Exhaustive comparison across every Q5.10
input in the lookup interval gives a maximum absolute error below 0.019 against
the checkpoint's tanh-approximate GELU.

The active postprocessor is serialized because an up result group arrives only
once every 24 inner-tile cycles. It consumes one of the 24 accumulator lanes per
cycle with a two-vector active/pending queue, so it sustains the exact result
rate with one requant multiplier and one GELU ROM port. The fixed workload
asserts that inner tiles per group are at least the number of output lanes.

| Postprocessor | DSP | LUT primitives | Flip-flops | BRAM |
| --- | ---: | ---: | ---: | ---: |
| 24 lanes in parallel | 96 | 8,309 | 802 | 0 |
| Serialized, full multiplier vectors | 4 | 5,210 | 4,830 | 1 |
| Serialized, factorized scales plus interstage sideband | 5 | 3,957 | 3,883 | 1 |

The first serialized version saves 92 DSPs and 3,099 LUT primitives at the cost
of 4,028 flip-flops and one RAMB18. A synchronous scalar GELU table alone maps
to one RAMB18, 37 LUT primitives, and 65 flip-flops. The active factorized
version adds one DSP to form a 16-bit token factor by 18-bit output factor each
cycle. It removes the duplicated 24-lane multiplier vectors and reduces the
postprocessor by another 1,253 LUT primitives and 947 flip-flops, including the
protected six-channel interstage multiplier sideband. A rejected shift-register
serializer had increased LUT use from 5,210 to 6,353.

Up token factors use unsigned Q18 in 16 bits and are generated from the runtime
per-token activation scale. Frozen output factors use unsigned Q20 in 18 bits.
Their product is rounded and shifted by eight to form the existing 24-bit
requant multiplier. Across all H0 blocks, the worst relative multiplier error
from factorization is 0.0567%. The ping-pong test proves distinct token groups
and distinct output-factor banks. This replaces about 5.4 KB of duplicated
group multiplier metadata with 1,024 bits of token factors plus two small
output-factor and bias banks.

## MLP-down and residual path

MLP-down uses the same active/pending serialized vector schedule but produces
signed Q13.10 values without GELU. The 24-lane default maps to 4 DSPs, 7,135
LUT primitives, and 4,902 flip-flops. The separate 24-lane saturating residual
adder maps to 1,392 LUT primitives and 593 flip-flops. Its self-checking test
covers positive overflow, negative overflow, ordinary addition, valid, and tag
preservation.

`mlp_down_pingpong_pipeline` connects the down serializer to the shared
ping-pong MAC controller and a residual-tile store indexed by token group and
output tile. A reduced end-to-end test overlaps two weight banks and verifies
bank-specific bias metadata, Q13.10 results, residual selection, output tags,
and completion for both tiles.

The real-checkpoint reference now quantizes `after_attention` itself to Q13.10
and performs the same saturating integer add as RTL. Across all twelve frozen
blocks, residual saturation, down saturation, and final block-output saturation
are all zero. Local block-output relative RMS remains between 0.6793% and
1.4339%, showing that the final fixed residual boundary adds negligible error
on H0.

## Weight stream adapter

`weight_slice_stream_adapter` converts a continuous 512-bit ready/valid stream
into the controller's 1,536-bit 6 by 32 INT8 weight-tile writes. Each tile is
exactly three stream beats. A full up slice uses 72 beats and a full down slice
uses 288 beats. The final beat stalls if the selected ping-pong bank is not
ready; otherwise adjacent tiles have no adapter bubble. `stream_last` is checked
against the final beat of the complete slice and latches a protocol error on a
mismatch.

At 250 MHz, the interface has a 16 GB/s peak before arbitration, which exceeds
the conservative 12.48 GB/s effective model. The current block is a width and
flow-control adapter, not an AXI address generator or cache-coherent DMA master.
It needs 72 uninterrupted beats per up slice and 288 per down slice, versus the
conservative bandwidth model's 96 and 373 load cycles after metadata.

The first general insertion-based assembler mapped to 20,188 LUT primitives.
Specializing the known three-beat layout into two fixed 512-bit registers plus
the live third beat reduces the final adapter to 18 LUT primitives and 1,035
flip-flops, with no DSP or BRAM. Both versions were functionally equivalent;
only the fixed-bank implementation is retained as the active result.

The format is deliberately not applied unchanged to the residual path. Frozen
MLP-down maxima grow from 19.1 in block 0 to 176.1 in block 11, so later blocks
need a wider fixed output or fewer fractional bits. The requantizer is therefore
width-parameterized. A full-block sweep must choose these formats before the
residual adder is frozen.

The RTL-equivalent reference now covers every block using the provisional alpha
table. It includes both W8A8 integer matrix products, the 24-bit requant
multipliers, symmetric rounding, bias, Q5.10 saturation, the exact generated
GELU ROM, and Q13.10 down conversion. Local block-output relative RMS ranges
from 0.6793% to 1.4339%, with a 1.2476% median and 1.1776% mean. Every local
cosine similarity is above 0.99991. The largest requant multiplier needs 22 of
the available 24 bits. Q13.10 down conversion has zero saturation. Q5.10 up
saturation peaks at 0.0107% in block 6, where the frozen input contains a large
negative tail that GELU suppresses.

For block 0 specifically, the full hardware arithmetic reference gives 1.3502%
block-output relative RMS and 0.999917 cosine similarity. The earlier model that
used floating dequantization and floating GELU gave 1.2537%, so fixed
post-processing adds 0.0965 percentage points on this input. These remain local
single-input comparisons, not a composed twelve-block generation validation.

The current best precision result uses per-token activation scales at both
linears, but exact down per-token scaling requires retaining all 64 by 3,072
GELU values before conversion. The streamable hybrid keeps per-token scaling
at MLP up and uses one frozen tensor activation scale at the GELU-to-down
boundary. Across all twelve blocks its local block-output relative RMS ranges
from 0.6987% to 1.6358%, with a 1.4111% median and 1.3533% mean. Fully per-token
scaling gives 1.1776% mean and fully tensor scaling gives 1.4172% mean. The
streamable choice therefore costs 0.1757 percentage points versus the best
local reference while avoiding a 393,216-byte Q5.10 intermediate buffer.

## GELU-to-down interstage

`smoothquant_int8_vector_serial` converts each 24-lane Q5.10 GELU vector to
symmetric INT8 using a frozen per-channel unsigned multiplier, symmetric
rounding, and saturation to -127 through +127. It consumes one lane each cycle
and has the same active plus pending vector capacity as the up postprocessor.
The frozen H0 constants require at most 21 of the available 24 multiplier bits.
Four channels are identically zero on H0 and are encoded with multiplier zero.
Only 59 of 2,359,296 converted values differ from ideal floating-point rounding.

The all-block exporter writes a 28,880,760-byte safetensors artifact containing
108 tensors: static up reciprocals, output factors, and Q10 biases, H0
token-factor and quantizer-multiplier goldens,
interstage multipliers, transformed INT8 down weights, down output multipliers,
and down Q10 biases for every block. H0 token factors are validation vectors,
not runtime constants. The artifact SHA-256 is
`1626ddc9d387adb2b04edd76049f6c4299e15673641d4882aa4e4cdb2f53eaba`.
The complete fixed-input, factorized-up, and streamable-down path has 1.3606%
mean and 1.6292% maximum local block-output relative RMS on H0.

## Runtime up activation quantizer

`mlp_up_activation_quantizer` consumes four normalized tokens per channel in
signed 18-bit Q5.12. A frozen unsigned 18-bit Q3.15 reciprocal applies the
SmoothQuant channel transform. The first 768-channel pass finds four maxima.
Four iterative 26-bit dividers then form the per-token INT8 reciprocals and are
reused to form unsigned 16-bit Q18 token factors. The second pass recomputes the
transformed values and emits one 4 by 32 INT8 activation tile every 32 accepted
channels. Replaying the on-chip hidden-state SRAM avoids a 6 KB transformed
activation buffer.

Across all H0 blocks, the Q5.12 input has zero saturation. The maximum
transformed-value relative RMS error is 0.0719%, at most 1.6724% of activation
codes differ from the float-scale reference in any block, and every difference
is one INT8 code. The maximum token-factor relative error is 0.0627%. After the
factorized up and complete down path, this boundary raises mean local block
error by only 0.0084 percentage points.

Open-source K26 mapping uses 8 DSP48E2 primitives, 1,858 LUT primitives, and
2,899 flip-flops, with no RAM. The fixed schedule takes two 768-channel passes
plus about 52 divider cycles per four-token group. At the provisional 250 MHz
target this is about 0.102 ms per block for all sixteen groups, excluding the
upstream normalized-value read interface and control handoff.

A checkpoint-derived RTL test uses the real block-0 SmoothQuant reciprocals and
the first four H0 `norm2_unaffine` tokens. It compares all four Q18 token factors
and all 24 emitted 4 by 32 INT8 tiles against the Python integer reference. All
3,072 activation bytes and all four factors match exactly.

`mlp_interstage_tile_bridge_bram` receives one 4 by 6 converted vector at a time
and packs it directly into the down engine's 4 by 32 activation tiles.
Six-channel fragments that cross a 32-channel boundary emit the completed tile
and retain the spill channels for the next tile. The 2,048-byte staging store
is organized as two 32-bit synchronous banks. Six writes fit between converted
vectors, and two channels are read per cycle to assemble a completed tile in
16 cycles. This replaces a full 393,216-byte Q5.10 GELU tensor.
`mlp_interstage_pipeline` connects the converter and bridge with ready/valid
backpressure and tagged output-tile and token-group ordering. Standalone and
connected cycle tests cover exact completion, boundary crossing, group
isolation, signed rounding, saturation, and tag retention.

Open-source mapping of the 24-lane converter uses one DSP48E2, 2,326 LUT
primitives, and 2,168 flip-flops. The first bridge stored sixteen wide partial
tiles and used variable indexed writes. It flattened to 142,058 LUT primitives,
17,420 flip-flops, one DSP, and no RAM, so it was rejected. Banking the bytes
into two synchronous memories proved two RAMB18 blocks, but variable writes to
the 1,024-bit output register still used 44,233 LUT primitives. Specializing the
active model's sixteen fixed channel-pair destinations and three legal boundary
cases reduces the final bridge to 2,314 LUT primitives, 2,488 flip-flops, two
RAMB18 blocks, and no DSP. That is a 61.4 times LUT reduction from the first
portable bridge and a 19.1 times reduction from the first BRAM rewrite.

These are open-source resource screens, not vendor timing results. The portable
RTL now infers the intended RAM structure, but Vitis place-and-route still has
to establish frequency, routing, and power.

`mlp_up_to_down_activation_pipeline` is the structural H1 connection from the
ping-pong up engine through factorized requantization, GELU, static interstage
conversion, and 4 by 32 down activation-tile writes. The six frozen interstage
multipliers are captured with the active weight-bank metadata, carried through
the serialized postprocess queue, and expanded across four tokens only at the
converter. This prevents a bank reload from changing multipliers while a GELU
vector is still in flight. The complete default-shape top elaborates with
Icarus, and the component cycle tests prove both bank sidebands, backpressure,
tag order, conversion, and boundary-crossing tile assembly.

`mlp_quantized_up_to_down_pipeline` makes the runtime quantizer the front end of
that connection. Its activation-tile and token-factor outputs are wired
directly into the up engine, eliminating both as external assumptions. The
default 64-token, 768-to-3,072-to-768 top passes full RTL elaboration. The later
automatic block gate now drives all sixteen real H0 token groups through this
complete shape.

`layer_norm_q12_group` closes the next upstream boundary. For four tokens it
accumulates signed sums and unsigned sums of squares across 768 Q13.10
channels, computes rounded population moments with iterative division, takes a
44-bit integer square root, and forms a Q18 inverse standard deviation. It then
replays the residual stream into signed Q5.12 normalized values. The replay
interface permits one statistics pass to feed more than one downstream
consumer.

The fixed-point reference was swept across all 24 block norms and the final
norm in the H0 trace. It has zero observed input saturation, 0.0144% mean
relative RMS error, 0.1025% maximum relative RMS error, and uses at most 32
variance bits, 44 square-root-radicand bits, and 20 inverse-standard-deviation
bits. These are frozen-input results, not a held-out calibration claim.

A real block-0 `after_attention` RTL test compares four tokens and all 768
channels against the integer reference. All 3,072 Q5.12 outputs match exactly.
The standalone open-source K26 mapping uses 24 DSP48E2 primitives, 4,420 LUT
primitives, and 2,860 flip-flops. It uses no BRAM. Timing remains unvalidated.

`layer_norm_mlp_up_activation_frontend` schedules one statistics pass and two
normalized replays. The first replay finds the four SmoothQuant maxima. The
second emits the 24 INT8 activation tiles. A checkpoint-derived connected test
starts at the block-0 residual tensor and checks all 3,072 activation bytes and
all four Q18 token factors exactly. The combined front end maps to 32 DSPs,
6,307 LUT primitives, and 5,784 flip-flops. `mlp_residual_up_to_down_pipeline`
wires this front end into the existing MLP up, GELU, interstage, and down path,
and the complete default-shape top passes RTL elaboration.

## Automatic shared-array MLP sublayer

The up and down ping-pong controllers can now disable their internal MACs and
export tagged requests. `mlp_shared_up_down_pipeline` adds one owner bit to the
tag, routes late responses back to the correct phase, and asserts that up and
down never request the physical array together. Its reduced exact test runs
128 up tiles followed by two down tiles through one `int8_mac_tile_pipelined`
instance. It observes 3,072 up requests and 48 down requests in 3,120 active
compute cycles.

The attention residual UltraRAM canvas is reused in place after folded output
projection. Each post-attention result overwrites its consumed residual tile,
so norm2 and MLP residual replay do not allocate another 64 by 768 hidden
canvas. This keeps the intended normalized and residual canvas total at 56
UltraRAM blocks instead of an estimated 80, which would exceed the K26 total
of 64 before routing and implementation effects.

`hidden_canvas_mlp_frontend` makes three controlled passes over one four-token
group: LayerNorm statistics, SmoothQuant maximum discovery, and quantized
activation emission. The standalone zero-reference test performs 384 canvas
reads, emits one token-factor vector and 24 activation tiles, and finishes in
3,340 busy cycles. `hidden_canvas_mlp_shared_pipeline` connects that front end
to the shared up/down array and completes the reduced structural path in 6,460
active datapath cycles.

`mlp_tile_load_sequencer` defines a concrete preload boundary. A bank is ready
only after every 32-channel weight chunk and the matching metadata have both
been accepted under independent backpressure. The down path additionally uses
`hidden_canvas_residual_load_sequencer` to replay every token group for the
requested output tile. `mlp_block_controller` overlaps the next weight bank
with current compute and automatically orders frontend groups, all up tiles,
interstage drain, and all down tiles.

`hidden_canvas_automatic_mlp_block` joins those controllers to the complete
datapath. With a four-token, 768-channel down input and 12-channel down output,
one start pulse causes 384 norm2 reads, 128 up tiles, two residual-replayed down
tiles, 3,120 physical-array requests, and two exact zero-reference outputs.
The self-checking run finishes in 11,372 total simulated cycles. This is a
reduced deterministic control and connectivity result, not a full-shape model
latency or a real-checkpoint accuracy claim.

The short opt-in captured-checkpoint gate drives the first four real block-0 tokens
through all 512 up output tiles, all 3,072 GELU and interstage channels, and the
first six down output channels. Runtime reciprocals are selected through the
new explicit 768-channel table address. The run issues 12,288 up requests and
96 down requests through one physical array, then matches all 24 post-residual
Q13.10 integers exactly in 34,881 simulated cycles.

The exhaustive mode uses the same generated checkpoint data and actual
4 by 6 by 32 physical INT8 array, with no accumulator mock. It runs all 64
tokens, all 512 up tiles, all 3,072 interstage channels, and all 128 down tiles.
The controller issues 393,216 array requests and emits 2,048 tagged vectors in
471,939 simulated cycles. Every one of the 49,152 post-residual Q13.10 integers
matches the bit-accurate Python reference. Two consecutive macOS Icarus runs
passed in 572.54 and 574.10 host seconds. These are functional simulation
results, not a measured FPGA frequency or latency.

The reduced-lane self-checking Icarus test accumulates two inner tiles, covers
positive and negative products, checks the final-valid pulse, and verifies stall
retention. It passes on macOS. Open-source mapping of the 1,024-lane default is
a resource screen only; vendor place-and-route remains required for frequency,
routing, and power claims.

The first 1,024-lane Yosys UltraScale+ mapping used 1,024 DSP48E2 primitives,
88,778 LUT primitives, and 1,025 flip-flop primitives. That is 82.05% of K26
DSPs and a conservative 75.80% LUT-primitive comparison before SRAM, vector
operations, attention control, DMA, or the sampler. The bare 1,024-lane array is
therefore rejected as the complete-bitstream commit point unless multiplier
packing substantially changes the vendor result. Smaller 512 and 768-lane
points are the active fit candidates.

The 512-lane mapping used 512 DSP48E2 primitives, 44,400 LUT primitives, and
513 flip-flop primitives. This is 41.03% of K26 DSPs and a conservative 37.91%
LUT-primitive comparison. It is a credible fit baseline, but at 250 MHz its
full-model arithmetic floor is about 62.4 ms per evaluation. Eight evaluations
would consume almost the entire 500 ms contract before utilization loss and
control overhead. The 768-lane point is therefore the active latency candidate,
while 512 lanes remains the safer resource fallback.

The 768-lane mapping used 768 DSP48E2 primitives, 66,552 LUT primitives, and
769 flip-flop primitives. This is 61.54% of K26 DSPs and a conservative 56.82%
LUT-primitive comparison. It leaves 480 DSPs and 50,568 LUT primitives before
packing for the rest of the model. Combined with the conservative latency model,
768 lanes is now the provisional K26 commit point. This selection remains
conditional on a pipelined reduction tree and vendor timing closure.

The original resource-screen RTL has a combinational inner-tile reduction. The
active timing-oriented RTL now registers the products and every level of a
balanced reduction tree. Its self-checking test sends two token groups without
an input bubble and confirms that result snapshots and tags remain independent.
The traversal-controller test covers load, automatic group and inner loops,
ordered results, signed arithmetic, and completion. Yosys reports still do not
provide timing, so vendor timing closure remains required before 250 MHz becomes
a measured result.

A quarter-width mapping of the same timing-oriented structure uses one token
lane, six output lanes, and 32 inner lanes. It maps to 192 DSP48E2 primitives,
3,519 LUT primitives, and 6,849 flip-flop primitives. Linear projection to four
token lanes is 768 DSPs, about 14,076 LUTs, and about 27,396 flip-flops. This is
an inference, not a substitute for a full-width map, but it shows that the
registered adder tree does not reproduce the old combinational array's very
large LUT cost. The exact 768-lane Yosys map was stopped after 30 minutes of
CPU time without a report; vendor synthesis remains the authoritative next
resource and timing gate.

Combining the measured LayerNorm-to-activation-quantizer front end,
factorized postprocessor,
interstage converter, and banked tile bridge with the linear projection of the
quarter-width MAC gives a provisional residual-to-down screen of 806 DSPs,
28,980 LUT primitives, 41,719
flip-flops, and three BRAMs before activation and weight SRAM, attention,
vector operations, DMA, or sampling. The MAC portion is projected and the
components are summed rather than mapped as one top, so this is an architecture
budget rather than a complete synthesis result.

## Commands

```bash
.venv/bin/diffusion-accel sweep-fixed-mlp \
  --package-dir data/hardware/mdlm-owt-169m-h0 \
  --block 0 \
  --out data/results/mdlm-block0-fixed-mlp-sweep.json

.venv/bin/diffusion-accel optimize-fixed-mlp-alphas \
  --package-dir data/hardware/mdlm-owt-169m-h0 \
  --out data/results/mdlm-all-block-fixed-mlp-alphas.json

.venv/bin/diffusion-accel export-mlp-interstage \
  --package-dir data/hardware/mdlm-owt-169m-h0 \
  --out data/hardware/mdlm-owt-169m-h0/mlp_interstage_int8.safetensors \
  --manifest data/results/mdlm-mlp-interstage-int8.json

iverilog -g2012 -Wall \
  -s tb_int8_mac_tile \
  -o /tmp/tb_int8_mac_tile \
  rtl/tensor_engine/int8_mac_tile.sv \
  rtl/tensor_engine/tb_int8_mac_tile.sv
vvp /tmp/tb_int8_mac_tile

iverilog -g2012 -Wall \
  -s tb_mlp_tile_controller \
  -o /tmp/tb_mlp_tile_controller \
  rtl/tensor_engine/int8_mac_tile_pipelined.sv \
  rtl/tensor_engine/mlp_tile_controller.sv \
  rtl/tensor_engine/tb_mlp_tile_controller.sv
vvp /tmp/tb_mlp_tile_controller

.venv/bin/python -m pytest -q tests/test_fixed_postprocess_rtl.py

.venv/bin/diffusion-accel sweep-hardware-fixed-mlp \
  --package-dir data/hardware/mdlm-owt-169m-h0 \
  --activation-granularity token \
  --out data/results/mdlm-all-block-hardware-fixed-mlp.json

.venv/bin/diffusion-accel analyze-mlp-pingpong \
  --clock-mhz 250 \
  --effective-ddr-gbps 12.48 \
  --evaluations 8 \
  --out data/results/mdlm-mlp-pingpong-analysis.json
```
