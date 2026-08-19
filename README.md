# DiffusionHard

Hardware exploration for masked diffusion language models, centered on the
169.6M-parameter MDLM-OWT checkpoint.

The stable v0.1 result is a streaming sampling subsystem that replaces a full
position-by-vocabulary probability cache with compact candidate state. The
complete DDiT accelerator is experimental and does not yet fit or run on a
KV260.

The Python package and CLI retain the `diffusion-accel` name for compatibility.

## Results

| Result | Value |
| --- | ---: |
| FP16 probability cache | 6,433,024 bytes |
| Candidate cache | 144 bytes |
| State reduction | 44,674x |
| 64-step model evaluations | 64 to 38 |
| Measured MPS forward time | 1,253 ms to 704 ms |
| M5 Pro end-to-end throughput | 90.2 to 254.1 tok/s |
| M5 Pro MLX, 64 steps | 479.4 tok/s |
| M5 Pro MLX TensorOps, 64 steps | 603.0 tok/s |
| M5 Pro MLX speed mode, 32 steps | 693.5 tok/s |
| Fused replay peak SRAM | 2.868 MiB to 0.375 MiB |

[Watch the 12-second measured trace demo](demo/diffusionhard-cache-demo.mp4).

The 64-step MLX baseline uses a validated 8-bit output head. The experimental
M5 TensorOps and 32-step modes have measured quality tradeoffs. See
[`docs/apple-mlx-quantization.md`](docs/apple-mlx-quantization.md). Yosys
resource screens are included, but they are not Vivado timing or board results.

In a common 64-prompt conditioned screen, maximum MDLM was 1.66x faster than
SmolLM-135M 4-bit, but GPT-2-large perplexity was 6.64x worse.

## Quick start

Requirements: Python 3.9+, PyTorch, a C++17 compiler, and Icarus Verilog.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,analysis]'

diffusion-accel validate-candidate-cache \
  --positions 64 \
  --vocabulary-size 50258 \
  --transitions 64 \
  --monte-carlo-trials 500000 \
  --out /tmp/candidate-cache.json

python -m pytest -q \
  tests/test_candidate_cache.py \
  tests/test_candidate_kernel.py \
  tests/test_candidate_rtl.py
```

On Apple Silicon, install `.[apple]` as well to enable the MLX benchmark.

The full v0.1 CI gate covers 13 Python, C++, and RTL tests.

## Repository

- `src/diffusion_accel/`: references, tracing, and analysis
- `hls/`: portable C++ kernels
- `rtl/`: sampling RTL and experimental tensor-engine RTL
- `data/results/`: small result records and open maps
- `docs/`: detailed decisions, evidence, and roadmap

See [`docs/release-status.md`](docs/release-status.md) for the exact validation
boundary and [`docs/milestone-results.md`](docs/milestone-results.md) for the
complete result history.

## Limitations

There is no completed Vivado implementation, achieved FPGA clock, bitstream,
board-measured power, or end-to-end FPGA text output. The last clean full-top
Yosys screen exceeded raw K26 LUT and BRAM capacity.

Model weights are not included. The upstream
[`kuleshov-group/mdlm-owt`](https://huggingface.co/kuleshov-group/mdlm-owt)
checkpoint is a separate Apache-2.0 project.

DiffusionHard is licensed under Apache-2.0.
