# Contributing to DiffusionAccel

DiffusionAccel is a research prototype. Contributions should make a numerical,
architectural, or implementation claim easier to reproduce and harder to
misinterpret.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,analysis]'
```

Run the focused v0.1 checks listed in the main README before submitting a
change to candidate caching, the RNG stream, noisy argmax, or the reveal
controller. Run the complete suite when changing shared Python or tensor-engine
code:

```bash
python -m pytest -q
```

Some complete-shape RTL tests are opt-in because they take several minutes.
Their environment variables and requirements are documented beside the tests.

## Evidence required for hardware claims

Please label results as mathematical reference, measured host execution,
analytical estimate, open-source technology map, vendor implementation, or
physical board measurement.

A vendor report should include:

- exact FPGA part and board;
- tool name and version;
- clock and I/O constraints;
- synthesis and implementation commands;
- achieved worst slack and frequency;
- LUT, flip-flop, DSP, BRAM, and UltraRAM counts;
- whether counts are primitive totals or occupied resources;
- a machine-readable report when licensing permits.

Do not convert an assumed clock into a measured latency claim. Do not add
resource counts from separately synthesized blocks and call the sum a placed
design.

## Numerical changes

Fixed-point or quantization changes should include:

- a bit-accurate software reference;
- boundary and saturation tests;
- at least one real-checkpoint comparison where applicable;
- the exact acceptance threshold and whether it passed;
- output examples only as illustrations, not quality evidence.

## Repository hygiene

- Do not commit model checkpoints, exported weight images, tool build trees,
  or user data.
- Keep generated result records small and include their configuration and
  provenance.
- Keep unrelated local changes out of a contribution.
- Use concise commit messages that state the verified outcome.

By submitting a contribution, you agree that it is licensed under Apache-2.0.
