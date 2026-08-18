# v0.1 release status

This document defines the public validation boundary for DiffusionAccel v0.1.
It exists so that unfinished full-chip work cannot be mistaken for the tested
sampler and memory subsystem.

## Stable release surface

- distribution-equivalent candidate-cache factorization;
- cache capacity and traffic analysis;
- native C++ candidate-reveal kernel;
- SystemVerilog candidate-reveal controller;
- Philox4x32 known-answer behavior;
- factorized fixed-point Gumbel conversion;
- streaming noisy argmax with mask exclusion and deterministic tie handling;
- accumulator requantization into the sampling boundary;
- small machine-readable result records under `data/results/`.

The focused validation command is maintained in the root README and in the
GitHub Actions workflow.

### Validation snapshot on 2026-08-18

The focused v0.1 command completed with:

```text
13 passed in 4.11s
```

The unrestricted repository suite completed with:

```text
228 passed, 10 skipped, 13 failed in 685.64s
```

The unrestricted failures are intentionally disclosed. Five evidence checks
reject stale RTL source hashes after the current edits. Seven connected RTL
checks still produce exact pass markers but have cycle counts that differ from
their retained expectations. One automatic MLP integration check also fails in
the experimental path. None of these tests belongs to the focused sampler
release gate.

## Experimental surface

The fixed-shape attention, MLP, parameter-loader, DDiT, and KV260-facing RTL
under `rtl/tensor_engine/` remains available for research and review. It is not
part of the v0.1 stability promise.

The most recent two-lane folded shared-MAC integration exposed missing
ready/valid backpressure between operator schedulers and the shared physical
array. Flow-control propagation is only partially implemented. The current
complete test suite therefore must not be represented as clean until that work
is finished or isolated.

## Last clean complete-top screen

The last recorded open-source map before the partial flow-control edits was:

| Resource | Count | Raw K26 comparison |
| --- | ---: | ---: |
| CLB-LUT-equivalent primitives | 183,645 | 156.80% |
| BRAM36-equivalent blocks | 162.5 | 112.85% |
| UltraRAM | 43 | 67.19% |
| DSP48E2 | 936 | 75.00% |
| Flip-flops | 130,821 | 55.85% |

This map does not fit the raw K26 capacity and predates the current source
state. It is retained as optimization history, not as a release result.

## Missing validation

DiffusionAccel v0.1 has no completed:

- Vivado synthesis or implementation report;
- achieved FPGA clock;
- bitstream;
- physical KV260 run;
- board-measured latency, bandwidth, power, or text output;
- complete-model resource fit;
- end-to-end eight-evaluation sprint-quality result.

Any future release claiming those results must add the underlying reports and
commands, not only a summarized number.
