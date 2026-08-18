# H0 frozen reference package

Status: complete on macOS and Apple MPS.

The package in `data/hardware/mdlm-owt-169m-h0` freezes the first hardware
graph at batch one, 64 tokens, 12 blocks, 12 heads, hidden size 768, and
vocabulary size 50,258. It was generated from checkpoint revision
`d0958fa851335ece6c15260ce0025f030673c0fb`.

## Numerical validation

| Check | Result |
| --- | ---: |
| Nonzero requested timestep versus forced zero | 0 maximum absolute error |
| Folded graph versus checkpoint | 0.0000762939 maximum absolute error |
| Folded graph mean absolute error | 0.00000811328 |
| Top-1 agreement after excluding mask ID | 100 percent |
| Acceptance threshold | 0.0001 maximum absolute error |

The package includes 240 FP32 golden tensors from both the original graph and
the constant-folded graph. Goldens cover embeddings, both normalizations, QKV,
attention, attention projection, MLP up, GELU, MLP down, every block boundary,
final normalization, logits, and top-1 token IDs.

## Weight images

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `folded_fp16_weights.safetensors` | 324,526,348 | `fe4cfe547e4b0f61a363a220d634ce6a24975244c3117bedbd7ac6918c987c6c` |
| `folded_fp16_weights.bin` | 324,620,288 | `b9365f930b8a515a0b308d8b09d5081f904800261694c6293af1ef9a3b7a779e` |
| `golden_tensors.safetensors` | 106,985,620 | `13a67f6f2f5b887da2bec743e61e68c3d22017dfd841af082dade894f2219270` |

The binary image contains 89 little-endian FP16 tensors in row-major order.
Every tensor begins at a 4,096-byte boundary. `manifest.json` records the exact
offset, length, shape, and order of every tensor for the future AXI loader.

The original 678,522,728-byte checkpoint hash is
`47149e73f7552f39ea9776dbe74d925d25237bcf2ed2e2ec03cdff9d51c82aa4`.

## Regeneration and verification

```bash
.venv/bin/diffusion-accel export-mdlm-hardware-package \
  --out-dir data/hardware/mdlm-owt-169m-h0 \
  --device auto \
  --local-files-only

.venv/bin/diffusion-accel validate-mdlm-hardware-package \
  --package-dir data/hardware/mdlm-owt-169m-h0
```

The independent validator checks all three file hashes, safetensors names and
shapes, binary bounds, non-overlap, 4 KiB alignment, and constant-fold status.
The actual decoded DDPM seed output remains recorded in
`data/traces/mdlm-owt-ddpm-candidate-cache-seed0.jsonl`; the mixed-input decoded
top-1 diagnostic is also stored in the package manifest.
