"""MLX-native execution for the pinned 169.6M-parameter MDLM checkpoint."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable, Sequence

import mlx.core as mx
import mlx.nn as nn


DEFAULT_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--kuleshov-group--mdlm-owt"
    / "snapshots/d0958fa851335ece6c15260ce0025f030673c0fb"
)


class _LayerNorm(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.weight = mx.ones((dimensions,))
        self.folded = False

    def __call__(self, inputs: mx.array) -> mx.array:
        weight = None if self.folded else self.weight
        return mx.fast.layer_norm(inputs, weight, None, 1e-5)


class _EmbeddingLayer(nn.Module):
    def __init__(self, dimensions: int, vocabulary_size: int) -> None:
        super().__init__()
        self.embedding = mx.zeros((vocabulary_size, dimensions))

    def __call__(self, token_ids: mx.array) -> mx.array:
        return self.embedding[token_ids]


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.mlp = [
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        ]
        self.frequency_size = frequency_size

    def __call__(self, timesteps: mx.array) -> mx.array:
        half = self.frequency_size // 2
        frequencies = mx.exp(
            -math.log(10_000.0)
            * mx.arange(half, dtype=mx.float32)
            / half
        )
        arguments = timesteps[:, None].astype(mx.float32) * frequencies[None]
        embedding = mx.concatenate(
            [mx.cos(arguments), mx.sin(arguments)],
            axis=-1,
        )
        hidden = self.mlp[0](embedding)
        hidden = self.mlp[1](hidden)
        return self.mlp[2](hidden)


class _Rotary(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (
            10_000.0
            ** (mx.arange(0, dimensions, 2, dtype=mx.float32) / dimensions)
        )

    def tables(
        self,
        sequence_length: int,
        dtype: mx.Dtype,
    ) -> tuple[mx.array, mx.array]:
        positions = mx.arange(sequence_length, dtype=mx.float32)
        frequencies = positions[:, None] * self.inv_freq[None]
        return mx.cos(frequencies).astype(dtype), mx.sin(frequencies).astype(dtype)


def _rotate(
    inputs: mx.array,
    cosine: mx.array,
    sine: mx.array,
) -> mx.array:
    half = inputs.shape[-1] // 2
    first = inputs[..., :half]
    second = inputs[..., half:]
    cosine = cosine[None, None]
    sine = sine[None, None]
    return mx.concatenate(
        [first * cosine - second * sine, second * cosine + first * sine],
        axis=-1,
    )


class _DDiTBlock(nn.Module):
    def __init__(self, dimensions: int, heads: int, condition_size: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = _LayerNorm(dimensions)
        self.attn_qkv = nn.Linear(dimensions, 3 * dimensions, bias=False)
        self.attn_out = nn.Linear(dimensions, dimensions, bias=False)
        self.norm2 = _LayerNorm(dimensions)
        self.mlp = [
            nn.Linear(dimensions, 4 * dimensions),
            nn.GELU(approx="tanh"),
            nn.Linear(4 * dimensions, dimensions),
        ]
        self.adaLN_modulation = nn.Linear(condition_size, 6 * dimensions)
        self._modulation: tuple[mx.array, ...] | None = None
        self._folded = False

    def prepare(self, condition: mx.array) -> None:
        modulation = self.adaLN_modulation(condition)[:, None]
        self._modulation = tuple(mx.split(modulation, 6, axis=-1))

    def fold_constants(self) -> None:
        """Fold constant AdaLN and residual gates into adjacent linears."""
        if self._modulation is None:
            raise RuntimeError("block condition has not been prepared")
        if self._folded:
            return
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = (value.reshape(-1) for value in self._modulation)

        qkv_weight = self.attn_qkv.weight
        qkv_scale = self.norm1.weight * (1 + scale_attention)
        folded_qkv = nn.Linear(qkv_weight.shape[1], qkv_weight.shape[0], bias=True)
        folded_qkv.weight = qkv_weight * qkv_scale[None]
        folded_qkv.bias = shift_attention @ mx.transpose(qkv_weight)
        self.attn_qkv = folded_qkv
        self.norm1.folded = True

        self.attn_out.weight = self.attn_out.weight * gate_attention[:, None]

        mlp_up_weight = self.mlp[0].weight
        mlp_scale = self.norm2.weight * (1 + scale_mlp)
        folded_mlp_up = nn.Linear(
            mlp_up_weight.shape[1],
            mlp_up_weight.shape[0],
            bias=True,
        )
        folded_mlp_up.weight = mlp_up_weight * mlp_scale[None]
        folded_mlp_up.bias = (
            self.mlp[0].bias + shift_mlp @ mx.transpose(mlp_up_weight)
        )
        self.mlp[0] = folded_mlp_up
        self.norm2.folded = True

        self.mlp[2].weight = self.mlp[2].weight * gate_mlp[:, None]
        self.mlp[2].bias = self.mlp[2].bias * gate_mlp
        self._folded = True
        mx.eval(self.parameters())

    def __call__(
        self,
        inputs: mx.array,
        cosine: mx.array,
        sine: mx.array,
    ) -> mx.array:
        if self._modulation is None:
            raise RuntimeError("block condition has not been prepared")
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self._modulation

        normalized = self.norm1(inputs)
        if not self._folded:
            normalized = normalized * (1 + scale_attention) + shift_attention
        batch, sequence, dimensions = normalized.shape
        head_dimensions = dimensions // self.heads
        qkv = self.attn_qkv(normalized).reshape(
            batch,
            sequence,
            3,
            self.heads,
            head_dimensions,
        )
        query = mx.transpose(qkv[:, :, 0], (0, 2, 1, 3))
        key = mx.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        value = mx.transpose(qkv[:, :, 2], (0, 2, 1, 3))
        query = _rotate(query, cosine, sine)
        key = _rotate(key, cosine, sine)
        attended = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=head_dimensions**-0.5,
        )
        attended = mx.transpose(attended, (0, 2, 1, 3)).reshape(
            batch,
            sequence,
            dimensions,
        )
        attention_output = self.attn_out(attended)
        hidden = inputs + (
            attention_output if self._folded else gate_attention * attention_output
        )

        normalized = self.norm2(hidden)
        if not self._folded:
            normalized = normalized * (1 + scale_mlp) + shift_mlp
        mlp = self.mlp[0](normalized)
        mlp = self.mlp[1](mlp)
        mlp = self.mlp[2](mlp)
        return hidden + (mlp if self._folded else gate_mlp * mlp)


class _FinalLayer(nn.Module):
    def __init__(self, dimensions: int, vocabulary_size: int, condition_size: int) -> None:
        super().__init__()
        self.norm_final = _LayerNorm(dimensions)
        self.linear = nn.Linear(dimensions, vocabulary_size)
        self.adaLN_modulation = nn.Linear(condition_size, 2 * dimensions)
        self._shift: mx.array | None = None
        self._scale: mx.array | None = None
        self._folded = False

    def prepare(self, condition: mx.array) -> None:
        self._shift, self._scale = mx.split(
            self.adaLN_modulation(condition),
            2,
            axis=-1,
        )

    def fold_constants(self) -> None:
        if self._shift is None or self._scale is None:
            raise RuntimeError("final condition has not been prepared")
        if self._folded:
            return
        shift = self._shift.reshape(-1)
        scale = self.norm_final.weight * (1 + self._scale.reshape(-1))
        weight = self.linear.weight
        folded = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
        folded.weight = weight * scale[None]
        folded.bias = self.linear.bias + shift @ mx.transpose(weight)
        self.linear = folded
        self.norm_final.folded = True
        self._folded = True
        mx.eval(self.parameters())

    def hidden(self, inputs: mx.array) -> mx.array:
        if self._shift is None or self._scale is None:
            raise RuntimeError("final condition has not been prepared")
        normalized = self.norm_final(inputs)
        if self._folded:
            return normalized
        return normalized * (1 + self._scale[:, None]) + self._shift[:, None]

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.linear(self.hidden(inputs))


class _Backbone(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 768,
        blocks: int = 12,
        heads: int = 12,
        condition_size: int = 128,
        vocabulary_size: int = 50_258,
    ) -> None:
        super().__init__()
        self.vocab_embed = _EmbeddingLayer(hidden_size, vocabulary_size)
        self.sigma_map = _TimestepEmbedder(condition_size)
        self.rotary_emb = _Rotary(hidden_size // heads)
        self.blocks = [
            _DDiTBlock(hidden_size, heads, condition_size)
            for _ in range(blocks)
        ]
        self.output_layer = _FinalLayer(
            hidden_size,
            vocabulary_size,
            condition_size,
        )
        self._cosine: mx.array | None = None
        self._sine: mx.array | None = None

    def prepare(self, sequence_length: int, dtype: mx.Dtype) -> None:
        condition = nn.silu(self.sigma_map(mx.zeros((1,), dtype=mx.float32)))
        for block in self.blocks:
            block.prepare(condition)
        self.output_layer.prepare(condition)
        self._cosine, self._sine = self.rotary_emb.tables(
            sequence_length,
            dtype,
        )

    def fold_constants(self) -> None:
        for block in self.blocks:
            block.fold_constants()
        self.output_layer.fold_constants()

    def hidden(self, token_ids: mx.array) -> mx.array:
        if self._cosine is None or self._sine is None:
            raise RuntimeError("backbone has not been prepared")
        hidden = self.vocab_embed(token_ids)
        for block in self.blocks:
            hidden = block(hidden, self._cosine, self._sine)
        return hidden

    def selected_logits(
        self,
        token_ids: mx.array,
        positions: mx.array,
    ) -> mx.array:
        hidden = self.hidden(token_ids)
        selected = hidden[0, positions][None]
        return self.output_layer(selected)[0]

    def full_logits(self, token_ids: mx.array) -> mx.array:
        return self.output_layer(self.hidden(token_ids))


class MLXMDLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()

    def prepare(self, sequence_length: int, dtype: mx.Dtype) -> None:
        self.backbone.prepare(sequence_length, dtype)

    def selected_logits(self, token_ids: mx.array, positions: mx.array) -> mx.array:
        return self.backbone.selected_logits(token_ids, positions)

    def full_logits(self, token_ids: mx.array) -> mx.array:
        return self.backbone.full_logits(token_ids)


def load_mlx_mdlm(
    *,
    snapshot: Path = DEFAULT_SNAPSHOT,
    dtype: str = "float32",
    sequence_length: int = 64,
    output_head_bits: int | None = None,
    fold_constants: bool = False,
) -> MLXMDLM:
    """Load the pinned PyTorch safetensors directly into the MLX model."""
    selected_dtype = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }.get(dtype)
    if selected_dtype is None:
        raise ValueError("dtype must be float32, float16, or bfloat16")
    weights = snapshot / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(weights)
    model = MLXMDLM()
    model.load_weights(str(weights), strict=True)
    model.set_dtype(selected_dtype)
    model.prepare(sequence_length, selected_dtype)
    if fold_constants:
        model.backbone.fold_constants()
    if output_head_bits is not None:
        if output_head_bits not in {4, 6, 8}:
            raise ValueError("output_head_bits must be 4, 6, 8, or None")
        nn.quantize(
            model,
            group_size=64,
            bits=output_head_bits,
            class_predicate=lambda path, module: (
                path == "backbone.output_layer.linear"
            ),
        )
    mx.eval(model.parameters())
    return model


def validate_mlx_mdlm(
    model: MLXMDLM,
    *,
    golden_tensors: Path,
) -> dict[str, float]:
    """Compare MLX logits against the frozen PyTorch H0 golden tensor."""
    from safetensors.numpy import load_file

    golden = load_file(golden_tensors)
    token_ids = mx.array(golden["input.ids"])
    reference = mx.array(golden["final.logits"])
    logits = model.full_logits(token_ids)
    difference = logits.astype(mx.float32) - reference.astype(mx.float32)
    top1 = mx.argmax(logits, axis=-1)
    reference_top1 = mx.argmax(reference, axis=-1)
    metrics = {
        "maximum_absolute_logit_error": float(mx.max(mx.abs(difference)).item()),
        "mean_absolute_logit_error": float(mx.mean(mx.abs(difference)).item()),
        "normalized_rmse": float(
            (
                mx.sqrt(mx.mean(mx.square(difference)))
                / mx.sqrt(mx.mean(mx.square(reference.astype(mx.float32))))
            ).item()
        ),
        "top1_agreement": float(mx.mean(top1 == reference_top1).item()),
    }
    return metrics


def validate_mlx_against_mps(
    model: MLXMDLM,
    *,
    sequence_length: int = 64,
    seeds: int = 32,
    seed_offset: int = 1_000,
) -> dict[str, float | int]:
    """Compare MLX and PyTorch/MPS logits on random half-masked canvases."""
    import numpy as np
    import torch

    from .apple_mdlm import _AppleSelectedLogits
    from .mdlm import DEFAULT_MODEL_ID, DEFAULT_REVISION, _load_mdlm_model

    reference_model = _load_mdlm_model(
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_REVISION,
        device="mps",
        local_files_only=True,
    )
    reference_logits = _AppleSelectedLogits(reference_model, device="mps")
    agreements = []
    normalized_errors = []
    mean_errors = []
    maximum_errors = []
    for seed in range(seed_offset, seed_offset + seeds):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        token_ids = torch.randint(
            0,
            50_257,
            (1, sequence_length),
            generator=generator,
        )
        token_ids[0, torch.rand(sequence_length, generator=generator) < 0.5] = (
            50_257
        )
        positions = torch.arange(sequence_length)
        with torch.inference_mode():
            reference = reference_logits(
                token_ids.to("mps"),
                positions.to("mps"),
            ).float().cpu()
        logits = model.selected_logits(
            mx.array(token_ids.numpy()),
            mx.arange(sequence_length),
        )
        mx.eval(logits)
        candidate = torch.from_numpy(np.array(logits, dtype="float32"))
        difference = candidate - reference
        agreements.append(
            float(
                candidate.argmax(dim=-1)
                .eq(reference.argmax(dim=-1))
                .float()
                .mean()
                .item()
            )
        )
        normalized_errors.append(
            float(
                (
                    difference.square().mean().sqrt()
                    / reference.square().mean().sqrt()
                ).item()
            )
        )
        mean_errors.append(float(difference.abs().mean().item()))
        maximum_errors.append(float(difference.abs().max().item()))
    return {
        "seeds": seeds,
        "positions": seeds * sequence_length,
        "top1_agreement": statistics.mean(agreements),
        "minimum_per_seed_top1_agreement": min(agreements),
        "mean_normalized_rmse": statistics.mean(normalized_errors),
        "mean_absolute_logit_error": statistics.mean(mean_errors),
        "maximum_absolute_logit_error": max(maximum_errors),
    }


def _reveal_groups(
    *,
    positions: int,
    steps: int,
    seed: int,
    sampling_epsilon: float = 1e-5,
) -> list[list[int]]:
    if positions <= 0 or steps <= 0:
        raise ValueError("positions and steps must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not 0.0 <= sampling_epsilon < 1.0:
        raise ValueError("sampling_epsilon must be in [0, 1)")
    generator = random.Random(seed)
    groups = [[] for _ in range(steps + 1)]
    reveal_mass = 1.0 - sampling_epsilon
    for position in range(positions):
        draw = generator.random()
        transition = (
            min(steps - 1, int(draw * steps / reveal_mass))
            if draw < reveal_mass
            else steps
        )
        groups[transition].append(position)
    return groups


def run_mlx_event_sampler(
    model: MLXMDLM,
    *,
    canvas_tokens: int = 64,
    prefix_token_ids: Sequence[int] = (),
    steps: int = 64,
    seed: int = 0,
    logits_function: Callable[[mx.array, mx.array], mx.array] | None = None,
    synchronize_each_event: bool = False,
) -> tuple[mx.array, dict[str, float | int]]:
    """Run distribution-equivalent event-driven sampling entirely in MLX."""
    mask_token_id = 50_257
    prefix_tokens = len(prefix_token_ids)
    if prefix_tokens:
        token_ids = mx.concatenate(
            [
                mx.array([prefix_token_ids], dtype=mx.int32),
                mx.full((1, canvas_tokens), mask_token_id, dtype=mx.int32),
            ],
            axis=1,
        )
    else:
        token_ids = mx.full((1, canvas_tokens), mask_token_id, dtype=mx.int32)
    groups = _reveal_groups(
        positions=canvas_tokens,
        steps=steps,
        seed=seed,
    )
    key = mx.random.key(seed)
    model_evaluations = 0
    selected_logits = logits_function or model.selected_logits
    mx.synchronize()
    started = time.perf_counter()
    for transition_positions in groups[:-1]:
        if not transition_positions:
            continue
        positions = mx.array(
            [prefix_tokens + position for position in transition_positions],
            dtype=mx.int32,
        )
        logits = selected_logits(token_ids, positions)
        key, sampling_key = mx.random.split(key)
        candidates = mx.random.categorical(
            logits[..., :-1],
            key=sampling_key,
        ).astype(mx.int32)
        token_ids = token_ids.at[0, positions].add(candidates - mask_token_id)
        if synchronize_each_event:
            mx.eval(token_ids)
        model_evaluations += 1

    if groups[-1]:
        positions = mx.array(
            [prefix_tokens + position for position in groups[-1]],
            dtype=mx.int32,
        )
        logits = selected_logits(token_ids, positions)
        candidates = mx.argmax(logits[..., :-1], axis=-1).astype(mx.int32)
        token_ids = token_ids.at[0, positions].add(candidates - mask_token_id)
        if synchronize_each_event:
            mx.eval(token_ids)
        model_evaluations += 1

    mx.eval(token_ids)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    return token_ids, {
        "wall_latency_ms": elapsed * 1e3,
        "output_tokens_per_second": canvas_tokens / elapsed,
        "model_evaluations": model_evaluations,
        "selected_output_rows": canvas_tokens,
        "prefix_tokens": prefix_tokens,
    }


def compile_mlx_event_sampler(
    model: MLXMDLM,
    *,
    canvas_tokens: int = 64,
    steps: int = 64,
    reveal_seed: int = 0,
    prefix_tokens: int = 0,
) -> tuple[Callable[[mx.array, mx.array], mx.array], dict[str, int]]:
    """Compile one complete sampler graph for a fixed reveal schedule."""
    mask_token_id = 50_257
    groups = _reveal_groups(
        positions=canvas_tokens,
        steps=steps,
        seed=reveal_seed,
    )

    def sample(token_ids: mx.array, key: mx.array) -> mx.array:
        for transition_positions in groups[:-1]:
            if not transition_positions:
                continue
            positions = mx.array(
                [prefix_tokens + position for position in transition_positions],
                dtype=mx.int32,
            )
            logits = model.selected_logits(token_ids, positions)
            key, sampling_key = mx.random.split(key)
            candidates = mx.random.categorical(
                logits[..., :-1],
                key=sampling_key,
            ).astype(mx.int32)
            token_ids = token_ids.at[0, positions].add(candidates - mask_token_id)

        if groups[-1]:
            positions = mx.array(
                [prefix_tokens + position for position in groups[-1]],
                dtype=mx.int32,
            )
            logits = model.selected_logits(token_ids, positions)
            candidates = mx.argmax(logits[..., :-1], axis=-1).astype(mx.int32)
            token_ids = token_ids.at[0, positions].add(candidates - mask_token_id)
        return token_ids

    return mx.compile(sample), {
        "model_evaluations": sum(bool(group) for group in groups),
        "selected_output_rows": canvas_tokens,
    }


def run_compiled_mlx_event_sampler(
    sampler: Callable[[mx.array, mx.array], mx.array],
    *,
    canvas_tokens: int = 64,
    prefix_token_ids: Sequence[int] = (),
    sampling_seed: int = 0,
) -> tuple[mx.array, dict[str, float | int]]:
    """Time a warmed, schedule-specialized complete sampler graph."""
    mask_token_id = 50_257
    if prefix_token_ids:
        token_ids = mx.concatenate(
            [
                mx.array([prefix_token_ids], dtype=mx.int32),
                mx.full((1, canvas_tokens), mask_token_id, dtype=mx.int32),
            ],
            axis=1,
        )
    else:
        token_ids = mx.full((1, canvas_tokens), mask_token_id, dtype=mx.int32)
    key = mx.random.key(sampling_seed)
    mx.synchronize()
    started = time.perf_counter()
    token_ids = sampler(token_ids, key)
    mx.eval(token_ids)
    mx.synchronize()
    elapsed = time.perf_counter() - started
    return token_ids, {
        "wall_latency_ms": elapsed * 1e3,
        "output_tokens_per_second": canvas_tokens / elapsed,
        "prefix_tokens": len(prefix_token_ids),
    }


def benchmark_mlx_mdlm(
    *,
    snapshot: Path = DEFAULT_SNAPSHOT,
    golden_tensors: Path,
    dtype: str = "float32",
    canvas_tokens: int = 64,
    steps: int = 64,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    output_head_bits: int | None = None,
    fold_constants: bool = False,
    compile_sampler: bool = False,
    mps_validation_seeds: int = 0,
) -> dict[str, Any]:
    """Validate and benchmark the MLX event sampler with warmed seed shapes."""
    if canvas_tokens <= 0 or steps <= 0:
        raise ValueError("canvas_tokens and steps must be positive")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    model = load_mlx_mdlm(
        snapshot=snapshot,
        dtype=dtype,
        sequence_length=canvas_tokens,
        output_head_bits=output_head_bits,
        fold_constants=fold_constants,
    )
    validation = validate_mlx_mdlm(model, golden_tensors=golden_tensors)
    selected_logits = mx.compile(model.selected_logits)
    samples = []
    for seed in seeds:
        compilation_warmup_ms = 0.0
        if compile_sampler:
            sampler, sampler_metadata = compile_mlx_event_sampler(
                model,
                canvas_tokens=canvas_tokens,
                steps=steps,
                reveal_seed=seed,
            )
            started = time.perf_counter()
            run_compiled_mlx_event_sampler(
                sampler,
                canvas_tokens=canvas_tokens,
                sampling_seed=seed,
            )
            compilation_warmup_ms = (time.perf_counter() - started) * 1e3
            output, metrics = run_compiled_mlx_event_sampler(
                sampler,
                canvas_tokens=canvas_tokens,
                sampling_seed=seed,
            )
            metrics.update(sampler_metadata)
        else:
            run_mlx_event_sampler(
                model,
                canvas_tokens=canvas_tokens,
                steps=steps,
                seed=seed,
                logits_function=selected_logits,
            )
            output, metrics = run_mlx_event_sampler(
                model,
                canvas_tokens=canvas_tokens,
                steps=steps,
                seed=seed,
                logits_function=selected_logits,
            )
        samples.append(
            {
                "seed": seed,
                "compilation_warmup_ms": compilation_warmup_ms,
                **metrics,
                "generated_token_ids": output[0].tolist(),
            }
        )
    latencies = [float(sample["wall_latency_ms"]) for sample in samples]
    median_latency = statistics.median(latencies)
    result = {
        "backend": "mlx",
        "mlx_version": mx.__version__,
        "dtype": dtype,
        "output_head_bits": output_head_bits,
        "fold_constants": fold_constants,
        "compiled_scope": (
            "full-sampler-schedule-specialized"
            if compile_sampler
            else "selected-logits"
        ),
        "canvas_tokens": canvas_tokens,
        "steps": steps,
        "seeds": list(seeds),
        "validation": validation,
        "samples": samples,
        "summary": {
            "median_wall_latency_ms": median_latency,
            "median_output_tokens_per_second": (
                canvas_tokens * 1e3 / median_latency
            ),
            "minimum_output_tokens_per_second": min(
                float(sample["output_tokens_per_second"])
                for sample in samples
            ),
            "maximum_output_tokens_per_second": max(
                float(sample["output_tokens_per_second"])
                for sample in samples
            ),
        },
    }
    if mps_validation_seeds:
        result["mps_random_canvas_validation"] = validate_mlx_against_mps(
            model,
            sequence_length=canvas_tokens,
            seeds=mps_validation_seeds,
        )
    return result


def write_benchmark(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
