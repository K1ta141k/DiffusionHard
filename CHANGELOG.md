# Changelog

All notable public changes to DiffusionAccel are documented here.

## 0.1.0

Initial research-prototype release surface:

- real MDLM-OWT tracing and hardware inventory;
- exact no-change model-forward reuse;
- distribution-equivalent 144-byte candidate cache;
- native C++ candidate-reveal kernel;
- tiled output-head and fused candidate-selection references;
- Philox4x32 and factorized fixed-point Gumbel implementations;
- streaming noisy-argmax and reveal-controller RTL;
- trace-driven memory simulation and K26-class analytical replays;
- open-source UltraScale+ technology maps;
- experimental fixed-shape MLP, attention, DDiT, and KV260 interface RTL.

This version does not claim vendor timing closure, complete K26 resource fit,
physical board execution, measured FPGA power, or end-to-end FPGA text output.
