# H4 block execution image

Status: block-0 image, address calculation, one-outstanding 512-bit AXI reader,
fixed weight-record reassembly, and model-specific weight-slice loading are
complete in portable RTL. Metadata consumers, vendor memory integration, and
board traffic measurements remain open.

## Artifact

The optimized block-0 execution image is
`data/hardware/mdlm-owt-169m-h0/block_00_execution.bin`.
It is 9,064,448 bytes and has SHA-256
`1558242167716dff76363693cd60be62192d6dd2b746bfbe6935ef9faa2eab77`.
The independently validated manifest is
`data/results/block-00-execution-image.json`.

The exporter derives every integer record from the pinned FP16 package and the
frozen MLP interstage artifact. It does not serialize tensors in framework
order. Records are emitted in the exact traversal order used by the H4 RTL.
Every section begins on a 4 KiB boundary and has its own SHA-256 digest. Record
padding and section padding are required to be zero.

| Section | Offset | Records | Payload | Stride | Order |
| --- | ---: | ---: | ---: | ---: | --- |
| QKV metadata | 0 | 396 | 32 B | 64 B | head, kind, output tile |
| QKV weights | 28,672 | 9,504 | 384 B | 384 B | head, kind, output tile, input tile |
| Rotary constants | 3,678,208 | 2,048 | 4 B | 4 B | token, half-head pair |
| Projection metadata | 3,686,400 | 128 | 18 B | 64 B | output tile |
| Projection weights | 3,694,592 | 3,072 | 192 B | 192 B | output tile, input tile |
| MLP reciprocal | 4,284,416 | 768 | 3 B | 4 B | hidden channel |
| MLP-up metadata | 4,288,512 | 512 | 56 B | 64 B | output tile |
| MLP-up weights | 4,321,280 | 12,288 | 192 B | 192 B | output tile, input tile |
| MLP-down metadata | 6,680,576 | 128 | 168 B | 192 B | output tile, token lane, output lane |
| MLP-down weights | 6,705,152 | 12,288 | 192 B | 192 B | output tile, input tile |

The section payloads occupy 9,060,096 bytes. Only 4,352 bytes are inter-section
or final alignment padding. At the same padded size, twelve block images would
occupy 108,773,376 bytes. That is an arithmetic projection until all twelve
images are exported and hashed.

## Address generation

`mdlm_block_parameter_address_generator.sv` accepts a 64-bit block base, a
section ID, and a record index. It returns:

- a bounds-checked 64-byte-aligned AXI address;
- the number of 64-byte beats;
- the payload byte count;
- the byte offset for compact four-byte rotary and reciprocal records.

The weight records require three beats for INT8 tiles and six beats for INT16
QKV tiles. Compact tables share one cache line across sixteen records. The RTL
test checks the final valid record of every section, compact-table offsets,
and out-of-range rejection. A Python test independently checks that the RTL
offsets, counts, and payload sizes equal the committed manifest.

## AXI transport and loader boundary

`axi512_read_burst_master.sv` accepts one command at a time and emits an AXI4
INCR read with 64-byte beats. It retains the request tag across the transfer,
propagates stream backpressure, counts bytes plus address and data stalls, and
detects non-OK responses and early or late `RLAST`. Its self-checking test
covers a stalled legal burst and an intentionally malformed early-last burst.

`mdlm_block_parameter_dma.sv` connects that master to the address generator.
An invalid section or out-of-range record completes locally without issuing an
AXI request. For a valid record, the stream carries its compact byte offset and
payload size beside every beat. The connected test checks a six-beat QKV weight
record, a reciprocal packed into byte offset four of a shared cache line, and
an invalid record.

`fixed_weight_record_adapter.sv` reassembles either six beats into one 3,072-bit
INT16 QKV tile or three beats into one 1,536-bit INT8 tile. Completed records
remain stable until the compute-side loader accepts them. It rejects an early
last beat and checks that all beats retain the same tag.

`mdlm_weight_slice_dma.sv` is the first direct compute-loader bridge. One
command calculates all record indices for an output tile, fetches each input
tile, and presents the existing bank, K-tile, and packed-weight interface. Its
QKV test issues 24 bursts and transfers 9,216 bytes while compute-side stalls
are absorbed by the completed-record buffer.

`fixed_aligned_record_adapter.sv` handles the 32-byte QKV metadata, 18-byte
projection metadata, 56-byte MLP-up metadata, and three-beat 168-byte MLP-down
metadata geometries. `compact_table_record_adapter.sv` selects one four-byte
slot from a shared cache line for rotary or SmoothQuant constants. The generic
`mdlm_parameter_record_dma.sv` connects either form to the checked DMA. Real
block-0 tests match QKV metadata and SmoothQuant reciprocal 17 against values
recomputed from the frozen tensor artifacts.

`mdlm_qkv_output_tile_loader.sv` is the first complete model-operator load
unit. It uses one shared AXI master to fetch the metadata record and then all
24 weight records for one requested head, kind, and six-channel output tile.
It directly drives the metadata and weight ports of the existing QKV scheduler.
The real block-0 test transfers 9,280 bytes and observes one exact metadata
transaction plus 24 exact packed INT16 tiles under independent consumer stalls.

`mdlm_projection_output_tile_loader.sv` applies the same one-master sequence to
the folded attention-output projection. It fetches one 18-byte multiplier
record followed by 24 three-beat INT8 weight records. The real block-0 loader
test transfers 4,672 bytes and matches every emitted bit.

Two connected compute tests remove manual parameter injection entirely. The
QKV loader drives `qkv_projection_output_tile_scheduler.sv` and its physical
mixed-precision MAC, producing all 64 by 6 exact Q12 values for block-0 output
tile zero. The projection loader drives
`attention_projection_output_tile_scheduler.sv` and produces all 64 by 6 exact
Q13.10 values. Thus both matrix stages around attention are validated from
committed DDR-image bytes through the real compute pipelines.

`mdlm_mlp_output_tile_loader.sv` parameterizes the same sequence for both MLP
matrices. In up mode it emits one 444-bit metadata item and 24 INT8 weight
tiles after 4,672 transferred bytes. In down mode it emits the three-beat
1,344-bit metadata item and all 96 INT8 weight tiles after 18,624 bytes. Real
block-0 tests independently rebuild the SmoothQuant up weights, down weights,
factors, biases, and interstage multipliers before comparing every emitted bit.

`axi512_read_arbiter_4.sv` gives the four operator loaders one physical read
port. It holds a grant across each complete burst, routes backpressure only to
the active client, uses round-robin selection between bursts, counts completed
transactions, and checks response framing independently of the client master.
Its test covers simultaneous requests, client-specific stalls, fair grant
order, and an injected late-last fault.

`mdlm_block_parameter_load_fabric.sv` packages the QKV, projection, MLP-up, and
MLP-down loaders behind that arbiter. The production-shape top elaborates with
one external 512-bit AXI read interface and exposes per-client byte counters.

The macOS Yosys K26-family screen initially inferred five DSP48E2 blocks for
fixed record-index multiplications. Replacing strides 11, 24, 33, and 96 with
baked shifts and adds removes all five DSPs. The post-rewrite screen reports
11,460 FDREs, two FDSEs, 370 CARRY4s, and 1,633 LUT1 through LUT6 primitives.
It also reports 10,336 OBUFs because this interface-heavy fabric was synthesized
as the chip top. That pad count is not a credible block-level utilization
number. Full details and limitations are in
`data/results/h4-block-parameter-load-fabric-screen.json`.

`ddit_block_with_parameter_fabric.sv` connects the one-port load fabric to the
actual DDiT controller. New untagged request-valid signals launch QKV and
projection loads, while the existing tag-matched ready signals still guard
acceptance. This avoids a circular dependency between loader launch and tag
matching. The reduced connected gate completes in 93,889 simulated cycles,
performs 4,100 AXI reads, transfers 918,400 bytes, and preserves the exact zero
reference through attention and MLP. Detailed counters and scope limitations
are in `data/results/h4-ddit-parameter-fabric-validation.json`.

`mdlm_block_constant_preloader.sv` closes the static-table side of the image.
It reads 128 rotary cache lines and 48 reciprocal cache lines, emits all 2,048
rotary pairs into the existing scratchpad interface, and fills a 768 by 18-bit
reciprocal lookup table. The real block-0 test performs 176 reads and transfers
11,264 bytes. Every rotary and reciprocal value matches the independently
loaded tensor artifacts exactly. Evidence is in
`data/results/h4-block-constant-preload-validation.json`.

`ddit_block_with_image_fabric.sv` now arbitrates that constant preloader and
the four-client dense fabric behind one external 512-bit AXI read port. It
prevents block launch until all constants are resident. The reduced connected
run now includes internal norm1 and completes in 140,301 simulated cycles. It
performs 4,276 reads and transfers 929,664 bytes. The accounting splits exactly
into 176 constant reads and 4,100 dense reads, while attention and final outputs
match the zero reference. Evidence is in
`data/results/h4-ddit-image-fabric-validation.json`.

`hidden_canvas_norm1_precompute.sv` reuses the attention residual-canvas replay
port for a statistics pass and a normalization pass, then stores 384 Q12 tiles
for QKV reuse. All 384 tiles match the real captured embedding exactly. Its
open-source K26-family screen reports 32 RAMB36, 25 DSP48E2, 4,762 LUT
primitives, and 3,463 flip-flops. The RAM allocation is material and requires
whole-design Vivado evidence. The reduced image-fabric test supplies no external
normalized data and still passes. Evidence is in
`data/results/h4-internal-norm1-validation.json`.

The complete real block-0 gate now passes as well. It uses all 12 heads, all
128 projection tiles, all 512 MLP-up tiles, and all 128 MLP-down tiles. The
single image port performs 38,492 reads and transfers 9,060,096 bytes. All
2,048 attention boundary vectors and all 2,048 final block vectors match an
independently recomputed fixed-point reference bit for bit. The run takes
1,584,457 simulated cycles and 2,558.38 seconds of macOS host time. Norm1 is
computed internally from the residual canvas, while the legacy normalized-data
input is held at zero. This is the first autonomous complete block proof from
Q10 residual input and the execution image through final Q10 output. Evidence
is in
`data/results/h4-ddit-full-block0-image-fabric-validation.json`.

At an unverified 250 MHz clock, the current schedule corresponds to 6.337828 ms
per block. Twelve blocks across eight denoising evaluations would consume
608.431488 ms before embedding, final normalization, vocabulary projection,
sampling, or memory stalls. This is a useful negative result: exactness and
full shape are closed, but the sub-500 ms goal requires schedule optimization.

The first schedule optimization is now implemented. The attention canvas stores
four-token groups as one wide word, so attention output projection streams one
32-channel K tile per cycle after the initial memory latency. Four-lane
requantizers replace the serial QKV and attention-projection postprocess loops.
A real two-output-tile block-0 projection remains bit exact and falls from 5,797
to 1,413 cycles. The reduced autonomous block screening run falls from 140,301
to 126,393 cycles with identical traffic and output. The grouped canvas maps to
64 RAMB36E2 primitives, and the two retained four-lane requantizers add 24 DSPs
against their serial versions. Parallel MLP-up and MLP-down postprocess
experiments were both rejected. The MLP-up work was already hidden behind the
MAC interval, and full-shape profiling showed that four-lane MLP-down
requantization saved only 18 block cycles for 12 extra DSPs.

The complete optimized full-shape rerun is bit exact at 1,189,573 cycles,
38,492 reads, and 9,060,096 read bytes. It saves 394,884 cycles, or 24.9224
percent, from the 1,584,457-cycle baseline. That compiled checkpoint includes
the later-rejected QKV vector drain and MLP-down four-lane experiment. Returning
to the retained scalar versions costs 258 cycles, for a 1,189,831-cycle
equivalent checkpoint. Evidence is in
`data/results/h4-grouped-projection-optimization.json`.

A parallel QKV scratchpad drain was also tested and rejected. With four-lane
requantization already active, the scalar production drain completes a real
block-0 head in 37,623 cycles and the vector prototype takes 37,603 cycles.
Only 20 cycles are exposed because the scalar writes overlap the next QKV
group's MAC work. The earlier 47,127-cycle measurement used serial
requantization, so 9,504 of the apparent 9,524-cycle delta belongs to the
requantizer. The vector router maps to an estimated 874 logic cells, and its
lane write muxes add about 2,094 estimated logic cells across the two
scratchpads. A bank-native crossbar was worse at 19,495 logic cells and 9 DSPs.
The scalar production path is retained. Evidence is in
`data/results/h4-qkv-vector-drain-optimization.json`.

The next optimization replaces stop-and-drain QKV scheduling with a continuous
16-group stream. Group tags travel through the six-stage MAC pipeline, four-lane
requantization overlaps following groups, and a two-entry FIFO absorbs scalar
router stalls. A real block-0 output tile falls from 683 to 428 cycles. A full
real head remains bit exact and falls from 37,623 to 29,703 cycles, saving 7,920
cycles without more DSPs. Open-source hierarchy mapping estimates 13 additional
logic cells and 411 fewer flip-flops than the legacy scheduler. Applying this
measured head delta across 12 heads gives a 1,094,791-cycle retained-production
projection, or 420.399744 ms for 12 blocks across eight evaluations at an
unverified 250 MHz. The complete rerun passes at 1,094,773 cycles with all
2,048 attention and 2,048 final vectors bit exact. It compiled the rejected
four-lane MLP-down postprocess, so the retained serial equivalent is 1,094,791
cycles. Traffic remains 38,492 reads and 9,060,096 bytes. Evidence is in
`data/results/h4-qkv-streaming-scheduler-optimization.json`.

The current compute-width experiment targets the remaining MLP bottleneck.
Two signed INT8 token activations that share a weight are offset and packed
into one DSP48E2 multiply. The standalone eight-token tile is bit exact and a
real captured eight-token MLP workload falls from 50,834 to 32,286 cycles. The
wide input, residual, and output adapters are connected to the automatic MLP,
and the reduced DDiT integration passes exactly.

Three dual-mode reduction organizations were screened. Separate attention and
MLP trees used 99,282 LUT primitives. A single 48-lane wide tree used 120,398
and exceeded K26 LUT capacity. A hybrid fabric tree with shared operand
selection reduced this to 79,664 LUTs but remained tight. The retained
organization uses the DSP48E2 PCIN and PCOUT cascade to form four eight-product
attention partials per output, then keeps only the packed MLP reduction in
fabric. It maps to 768 DSP48E2s, 54,082 LUT primitives, 40,878 flip-flops, and
an estimated 28,721 logic cells. This is 46.1766 percent of K26 LUT capacity,
17.4513 percent of flip-flop capacity, and 61.5385 percent of DSP capacity.

The explicit cell uses the documented multiply plus PCIN operating mode with
its multiplier register enabled and output register bypassed. Behavioral RTL
matches 20,000 randomized cell vectors, and the full array matches eight
attention plus eight packed-MLP reference results. The portable hierarchy also
elaborates with the explicit primitive library. These checks do not replace a
Vivado implementation report. Placement of eight-cell cascade groups, timing
at 250 MHz, and the complete full-shape execution-image rerun remain open.
Detailed evidence is in
`data/results/h4-packed-eight-token-mlp-screen.json`, with raw mapping counts in
`data/results/mixed-precision-packed-m8-dsp-cascade-yosys-xcup.json`.

The block-0 checkpoint test goes beyond synthetic traffic. It serves the first
24 QKV records from the committed execution image and independently recomputes
the expected INT16 weights from the pinned FP16 tensor. All 24 packed 3,072-bit
loader words match exactly. This validates the path from frozen model weights
through image byte order, address calculation, AXI transport, and the existing
compute-loader boundary.

The current master deliberately allows only one outstanding transaction. This
is the correctness-first implementation. Multi-outstanding scheduling is a
later bandwidth optimization and requires vendor memory-controller evidence.

## Reproduction

```bash
.venv/bin/diffusion-accel export-block-execution-image \
  --package-dir data/hardware/mdlm-owt-169m-h0 \
  --block 0 \
  --out data/hardware/mdlm-owt-169m-h0/block_00_execution.bin \
  --manifest data/results/block-00-execution-image.json

.venv/bin/diffusion-accel validate-block-execution-image \
  --manifest data/results/block-00-execution-image.json

.venv/bin/python -m pytest -q \
  tests/test_block_image.py \
  tests/test_mdlm_block_parameter_address_generator_rtl.py \
  tests/test_axi512_read_burst_master_rtl.py \
  tests/test_mdlm_block_parameter_dma_rtl.py \
  tests/test_fixed_weight_record_adapter_rtl.py \
  tests/test_mdlm_weight_slice_dma_rtl.py \
  tests/test_mdlm_weight_slice_dma_h0_rtl.py \
  tests/test_parameter_record_adapters_rtl.py \
  tests/test_mdlm_parameter_record_dma_h0_rtl.py \
  tests/test_mdlm_qkv_output_tile_loader_h0_rtl.py \
  tests/test_mdlm_projection_output_tile_loader_h0_rtl.py \
  tests/test_mdlm_qkv_loader_compute_h0_rtl.py \
  tests/test_mdlm_projection_loader_compute_h0_rtl.py \
  tests/test_mdlm_mlp_output_tile_loader_h0_rtl.py \
  tests/test_axi512_read_arbiter_4_rtl.py \
  tests/test_mdlm_block_parameter_load_fabric_rtl.py \
  tests/test_mdlm_block_constant_preloader_h0_rtl.py

DIFFUSION_ACCEL_RUN_DDIT_FABRIC_RTL=1 \
  .venv/bin/python -m pytest -q \
  tests/test_ddit_block_with_parameter_fabric_rtl.py

DIFFUSION_ACCEL_RUN_DDIT_IMAGE_FABRIC_RTL=1 \
  .venv/bin/python -m pytest -q \
  tests/test_ddit_block_with_image_fabric_functional_rtl.py

DIFFUSION_ACCEL_RUN_FULL_BLOCK_RTL=1 \
  .venv/bin/python -m pytest -q \
  tests/test_ddit_block_with_image_fabric_h0_rtl.py
```

The validator must report zero errors. These are macOS-side data-layout,
transport, and RTL-loader results. They are not evidence of achieved DDR
bandwidth on KV260.
