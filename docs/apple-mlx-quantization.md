# Apple MLX quantization

The useful M5 Pro path is mixed W8A8, not blanket 4-bit. Cider TensorOps on
QKV and MLP-up reaches 581.1 tok/s at 64 steps, 23.1% above the q8-head MLX
baseline. Adding MLP-down reaches 603.0 tok/s, with a larger quality tradeoff.

| Mode | 64-step tok/s | Random-canvas top-1 |
| --- | ---: | ---: |
| q8 output head | 472.1 | 98.83% |
| balanced W8A8 | 485.1 | 97.95% |
| core W8A8 | 581.1 | 94.63% |
| maximum W8A8 | 603.0 | 90.97% |
| maximum plus q4 head | 610.5 | 86.96% |

The q4 head adds only 1.24% over maximum W8A8. It is not the recommended
mode. The 16-sample conditioned screen is in
`data/results/mdlm-mlx-m5-tensorops-quantization-screen.json`.

## Setup

Cider 0.8.0 requires an M5-family Mac and Python 3.12 or newer. This source
build was verified with MLX 0.32.1 and nanobind 2.13.0.

```bash
python -m pip install -e '.[apple]'
python -m pip install 'nanobind==2.13.0' cmake
git clone https://github.com/Mininglamp-AI/cider.git /tmp/cider
git -C /tmp/cider checkout 4d91fcee9439f7aea17ae6e965271d9536c604a0
CIDER_FORCE_BUILD=1 python -m pip install --no-build-isolation /tmp/cider
```

## Run

```bash
diffusion-accel benchmark-mdlm-mlx \
  --golden-tensors data/hardware/mdlm-owt-169m-h0/golden_tensors.safetensors \
  --output-head-bits 8 \
  --cider-w8a8 core \
  --mps-validation-seeds 32 \
  --seeds 0 1 2 3 4
```

Use `balanced`, `core`, or `maximum`. These are experimental M5-only modes.
The throughput numbers are warm single-request measurements, not serving
throughput or energy measurements.
