"""Measure whether full-attention MDLM prefix K/V survives a suffix edit."""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Callable, Dict, List

from .mdlm import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    _load_mdlm_model,
    _resolve_device,
    _synchronize,
    prefix_isolation,
)


DEFAULT_PREFIX_TEXT = (
    "Efficient language model hardware should keep reusable context close to "
    "the compute units."
)


def tensor_drift_metrics(reference: Any, changed: Any) -> Dict[str, object]:
    """Return stable scalar drift metrics for two same-shaped tensors."""
    import torch

    if reference.shape != changed.shape:
        raise ValueError("drift tensors must have the same shape")
    reference = reference.detach().float().cpu()
    changed = changed.detach().float().cpu()
    difference = changed - reference
    reference_rms = reference.square().mean().sqrt()
    rmse = difference.square().mean().sqrt()
    denominator = reference.flatten().norm() * changed.flatten().norm()
    cosine = torch.tensor(1.0)
    if float(denominator.item()) > 0:
        cosine = torch.dot(reference.flatten(), changed.flatten()) / denominator
        cosine = cosine.clamp(-1.0, 1.0)
    return {
        "exact_equal": bool(torch.equal(reference, changed)),
        "cosine_similarity": float(cosine.item()),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "max_absolute_error": float(difference.abs().max().item()),
        "normalized_rmse": float(
            (rmse / reference_rms).item()
            if float(reference_rms.item()) > 0
            else math.inf
        ),
        "fraction_changed_gt_1e_6": float(
            difference.abs().gt(1e-6).float().mean().item()
        ),
    }


def _extract_logits(output: Any) -> Any:
    return output.logits if hasattr(output, "logits") else output


def _capture_prefix_qkv(
    model: Any,
    input_ids: Any,
    timestep: Any,
    *,
    prefix_tokens: int,
    heads: int,
) -> tuple[Any, List[Any]]:
    captures: List[Any] = []
    handles = []

    def capture_projection(_module: Any, _inputs: Any, output: Any) -> None:
        batch, sequence, three_hidden = output.shape
        head_dimension = three_hidden // (3 * heads)
        qkv = output.reshape(batch, sequence, 3, heads, head_dimension)
        # Copy only K/V for the fixed prefix. The projection is captured before
        # rotary embedding, which is deterministic and position-local.
        captures.append(qkv[:, :prefix_tokens, 1:3].detach().clone())

    for block in model.backbone.blocks:
        handles.append(block.attn_qkv.register_forward_hook(capture_projection))
    try:
        output = model(input_ids=input_ids, timesteps=timestep)
    finally:
        for handle in handles:
            handle.remove()
    if len(captures) != len(model.backbone.blocks):
        raise RuntimeError("failed to capture one K/V projection per MDLM block")
    return _extract_logits(output), captures


def _benchmark(
    operation: Callable[[], Any],
    *,
    device: str,
    repeats: int,
) -> tuple[Any, float, List[float]]:
    latencies = []
    output = None
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        output = operation()
        _synchronize(device)
        latencies.append(time.perf_counter() - started)
    return output, statistics.median(latencies), latencies


def _timed_forward(
    model: Any,
    input_ids: Any,
    timestep: Any,
    device: str,
    repeats: int,
) -> tuple[Any, float, List[float]]:
    output, median, samples = _benchmark(
        lambda: model(input_ids=input_ids, timesteps=timestep),
        device=device,
        repeats=repeats,
    )
    return _extract_logits(output), median, samples


def _top1_ids(logits: Any, mask_token_id: int) -> Any:
    logits = logits.detach().float().clone()
    logits[..., mask_token_id] = -math.inf
    return logits.argmax(dim=-1)


def _top1_agreement(left: Any, right: Any) -> float:
    return float(left.eq(right).float().mean().item())


def _apply_rotary(vector: Any, positions: Any, inverse_frequency: Any) -> Any:
    """Apply the checkpoint's rotary convention to Q or K."""
    import torch

    frequencies = torch.einsum(
        "i,j->ij",
        positions.to(dtype=inverse_frequency.dtype),
        inverse_frequency,
    )
    cosine = frequencies.cos()[None, :, None, :]
    sine = frequencies.sin()[None, :, None, :]
    half = vector.shape[-1] // 2
    first = vector[..., :half]
    second = vector[..., half:]
    return torch.cat(
        [first * cosine - second * sine, second * cosine + first * sine],
        dim=-1,
    )


def _cached_suffix_forward(
    model: Any,
    suffix_ids: Any,
    timestep: Any,
    *,
    prefix_tokens: int,
    prefix_kv: List[Any],
    return_kv: bool = False,
) -> Any:
    """Run only the active suffix using exact prefix-isolated cached K/V."""
    import torch
    import torch.nn.functional as functional

    backbone = model.backbone
    sigma = timestep
    if not model.config.time_conditioning:
        sigma = torch.zeros_like(sigma)
    conditioning = functional.silu(backbone.sigma_map(sigma))
    hidden = backbone.vocab_embed(suffix_ids)
    active_tokens = suffix_ids.shape[1]
    positions = torch.arange(
        prefix_tokens,
        prefix_tokens + active_tokens,
        device=suffix_ids.device,
    )
    inverse_frequency = backbone.rotary_emb.inv_freq
    heads = int(model.config.n_heads)
    active_kv: List[Any] = []

    for layer_index, block in enumerate(backbone.blocks):
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = block.adaLN_modulation(conditioning)[:, None].chunk(6, dim=2)

        residual = hidden
        attention_input = block.norm1(hidden)
        attention_input = attention_input * (1 + scale_attention) + shift_attention
        qkv = block.attn_qkv(attention_input)
        head_dimension = qkv.shape[-1] // (3 * heads)
        qkv = qkv.reshape(
            qkv.shape[0],
            qkv.shape[1],
            3,
            heads,
            head_dimension,
        )
        query = _apply_rotary(qkv[:, :, 0], positions, inverse_frequency)
        active_key = _apply_rotary(qkv[:, :, 1], positions, inverse_frequency)
        active_value = qkv[:, :, 2]
        if return_kv:
            active_kv.append(
                torch.stack([active_key, active_value], dim=2).detach().clone()
            )
        cached_key = prefix_kv[layer_index][:, :, 0]
        cached_value = prefix_kv[layer_index][:, :, 1]
        key = torch.cat([cached_key, active_key], dim=1)
        value = torch.cat([cached_value, active_value], dim=1)
        attention = functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        attention = attention.transpose(1, 2).reshape(
            hidden.shape[0],
            active_tokens,
            -1,
        )
        hidden = residual + gate_attention * block.attn_out(attention)

        residual = hidden
        mlp_input = block.norm2(hidden)
        mlp_input = mlp_input * (1 + scale_mlp) + shift_mlp
        hidden = residual + gate_mlp * block.mlp(mlp_input)

    if return_kv:
        return active_kv
    return backbone.output_layer(hidden, conditioning)


def _materialize_cached_block(
    model: Any,
    block_ids: Any,
    timestep: Any,
    *,
    prefix_tokens: int,
    prefix_kv: List[Any],
) -> List[Any]:
    """Finalize one block and append its per-layer K/V to session state."""
    import torch

    block_kv = _cached_suffix_forward(
        model,
        block_ids,
        timestep,
        prefix_tokens=prefix_tokens,
        prefix_kv=prefix_kv,
        return_kv=True,
    )
    combined = [
        torch.cat([cached, appended], dim=1)
        for cached, appended in zip(prefix_kv, block_kv)
    ]
    return combined


def _rotate_prefix_cache(model: Any, captures: List[Any]) -> List[Any]:
    """Convert captured pre-rotary prefix K/V into reusable attention state."""
    import torch

    prefix_tokens = captures[0].shape[1]
    positions = torch.arange(prefix_tokens, device=captures[0].device)
    inverse_frequency = model.backbone.rotary_emb.inv_freq
    rotated = []
    for capture in captures:
        key = _apply_rotary(capture[:, :, 0], positions, inverse_frequency)
        value = capture[:, :, 1]
        rotated.append(torch.stack([key, value], dim=2))
    return rotated


def analyze_mdlm_prefix_drift(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: str = "auto",
    prefix_text: str = DEFAULT_PREFIX_TEXT,
    suffix_tokens: int = 32,
    changed_suffix_tokens: int = 1,
    warmup: int = 1,
    timing_repeats: int = 5,
    seed: int = 0,
    local_files_only: bool = False,
) -> Dict[str, object]:
    """Run two real MDLM forwards and quantify fixed-prefix K/V drift."""
    if not prefix_text.strip():
        raise ValueError("prefix_text must not be empty")
    if suffix_tokens <= 0:
        raise ValueError("suffix_tokens must be positive")
    if changed_suffix_tokens <= 0 or changed_suffix_tokens > suffix_tokens:
        raise ValueError("changed_suffix_tokens must be in [1, suffix_tokens]")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if timing_repeats <= 0:
        raise ValueError("timing_repeats must be positive")

    import torch
    from transformers import AutoTokenizer

    resolved_device = _resolve_device(device)
    torch.manual_seed(seed)
    model = _load_mdlm_model(
        model_id=model_id,
        revision=revision,
        device=resolved_device,
        local_files_only=local_files_only,
    )
    import flash_attn

    if not getattr(flash_attn, "_diffusion_accel_compat", False):
        raise RuntimeError(
            "prefix isolation currently requires the macOS attention compatibility path"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        local_files_only=local_files_only,
    )
    prefix_ids_list = tokenizer.encode(prefix_text, add_special_tokens=False)
    if not prefix_ids_list:
        raise ValueError("prefix_text produced no tokens")

    config = model.config
    total_tokens = len(prefix_ids_list) + suffix_tokens
    if total_tokens > int(config.model_length):
        raise ValueError(
            "prefix and suffix exceed checkpoint context length %d"
            % config.model_length
        )
    mask_token_id = int(config.vocab_size) - 1
    prefix_ids = torch.tensor(prefix_ids_list, dtype=torch.long)
    baseline_ids = torch.cat(
        [
            prefix_ids,
            torch.full((suffix_tokens,), mask_token_id, dtype=torch.long),
        ]
    ).unsqueeze(0)
    changed_ids = baseline_ids.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    replacement_ids = torch.randint(
        0,
        mask_token_id,
        (changed_suffix_tokens,),
        generator=generator,
        dtype=torch.long,
    )
    changed_positions = torch.randperm(
        suffix_tokens,
        generator=generator,
    )[:changed_suffix_tokens].sort().values
    changed_ids[
        0,
        len(prefix_ids_list) + changed_positions,
    ] = replacement_ids
    baseline_ids = baseline_ids.to(resolved_device)
    changed_ids = changed_ids.to(resolved_device)
    timestep = torch.ones(1, device=resolved_device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(input_ids=baseline_ids, timesteps=timestep)
        _synchronize(resolved_device)
        baseline_logits, baseline_elapsed, baseline_latency_samples = _timed_forward(
            model,
            baseline_ids,
            timestep,
            resolved_device,
            timing_repeats,
        )
        changed_logits, changed_elapsed, changed_latency_samples = _timed_forward(
            model,
            changed_ids,
            timestep,
            resolved_device,
            timing_repeats,
        )
        _, baseline_kv = _capture_prefix_qkv(
            model,
            baseline_ids,
            timestep,
            prefix_tokens=len(prefix_ids_list),
            heads=int(config.n_heads),
        )
        _, changed_kv = _capture_prefix_qkv(
            model,
            changed_ids,
            timestep,
            prefix_tokens=len(prefix_ids_list),
            heads=int(config.n_heads),
        )
        _synchronize(resolved_device)

        with prefix_isolation(len(prefix_ids_list)):
            # Warm the masked attention path separately. MPS compiles a new
            # kernel shape the first time this mask is used.
            model(input_ids=changed_ids, timesteps=timestep)
            _synchronize(resolved_device)
            isolated_logits, isolated_elapsed, isolated_latency_samples = _timed_forward(
                model,
                changed_ids,
                timestep,
                resolved_device,
                timing_repeats,
            )
            _, isolated_baseline_kv = _capture_prefix_qkv(
                model,
                baseline_ids,
                timestep,
                prefix_tokens=len(prefix_ids_list),
                heads=int(config.n_heads),
            )
            _, isolated_changed_kv = _capture_prefix_qkv(
                model,
                changed_ids,
                timestep,
                prefix_tokens=len(prefix_ids_list),
                heads=int(config.n_heads),
            )
        prefix_only_ids = baseline_ids[:, : len(prefix_ids_list)]
        model(input_ids=prefix_only_ids, timesteps=timestep)
        _synchronize(resolved_device)
        _, prefix_prefill_elapsed, prefix_prefill_latency_samples = _timed_forward(
            model,
            prefix_only_ids,
            timestep,
            resolved_device,
            timing_repeats,
        )
        _, prefix_captures = _capture_prefix_qkv(
            model,
            prefix_only_ids,
            timestep,
            prefix_tokens=len(prefix_ids_list),
            heads=int(config.n_heads),
        )
        rotated_prefix_cache = _rotate_prefix_cache(model, prefix_captures)
        _cached_suffix_forward(
            model,
            changed_ids[:, len(prefix_ids_list) :],
            timestep,
            prefix_tokens=len(prefix_ids_list),
            prefix_kv=rotated_prefix_cache,
        )
        (
            cached_suffix_logits,
            cached_suffix_elapsed,
            cached_suffix_latency_samples,
        ) = _benchmark(
            lambda: _cached_suffix_forward(
                model,
                changed_ids[:, len(prefix_ids_list) :],
                timestep,
                prefix_tokens=len(prefix_ids_list),
                prefix_kv=rotated_prefix_cache,
            ),
            device=resolved_device,
            repeats=timing_repeats,
        )

    layers = []
    first_drifting_layer = None
    for layer_index, (reference, changed) in enumerate(
        zip(baseline_kv, changed_kv)
    ):
        key_metrics = tensor_drift_metrics(reference[:, :, 0], changed[:, :, 0])
        value_metrics = tensor_drift_metrics(
            reference[:, :, 1], changed[:, :, 1]
        )
        exact = bool(key_metrics["exact_equal"] and value_metrics["exact_equal"])
        if not exact and first_drifting_layer is None:
            first_drifting_layer = layer_index
        layers.append(
            {
                "layer": layer_index,
                "prefix_kv_exact_equal": exact,
                "key": key_metrics,
                "value": value_metrics,
            }
        )

    prefix_tokens = len(prefix_ids_list)
    baseline_top1 = _top1_ids(baseline_logits, mask_token_id)
    changed_top1 = _top1_ids(changed_logits, mask_token_id)
    baseline_suffix_top1 = baseline_top1[0, prefix_tokens:]
    changed_suffix_top1 = changed_top1[0, prefix_tokens:]
    changed_absolute_positions = [
        prefix_tokens + int(position) for position in changed_positions.tolist()
    ]
    isolated_prefix_exact = all(
        bool(torch.equal(reference, changed))
        for reference, changed in zip(isolated_baseline_kv, isolated_changed_kv)
    )
    isolated_suffix_logits = isolated_logits[:, prefix_tokens:]
    isolated_suffix_top1 = _top1_ids(isolated_suffix_logits, mask_token_id)
    cached_suffix_top1 = _top1_ids(cached_suffix_logits, mask_token_id)
    cached_logit_drift = tensor_drift_metrics(
        isolated_suffix_logits,
        cached_suffix_logits,
    )

    return {
        "model_id": model_id,
        "revision": revision,
        "device": resolved_device,
        "attention_policy": "full-bidirectional",
        "capture_point": "pre-rotary-key-value-projection",
        "seed": seed,
        "prefix_text": prefix_text,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "changed_suffix_tokens": changed_suffix_tokens,
        "changed_absolute_positions": changed_absolute_positions,
        "replacement_token_ids": replacement_ids.tolist(),
        "replacement_text": tokenizer.decode(replacement_ids.tolist()),
        "baseline_forward_latency_ms": baseline_elapsed * 1e3,
        "baseline_forward_latency_samples_ms": [
            sample * 1e3 for sample in baseline_latency_samples
        ],
        "changed_forward_latency_ms": changed_elapsed * 1e3,
        "changed_forward_latency_samples_ms": [
            sample * 1e3 for sample in changed_latency_samples
        ],
        "prefix_logits_top1_agreement": _top1_agreement(
            baseline_top1[0, :prefix_tokens],
            changed_top1[0, :prefix_tokens],
        ),
        "suffix_logits_top1_agreement": _top1_agreement(
            baseline_suffix_top1,
            changed_suffix_top1,
        ),
        "baseline_suffix_top1_text": tokenizer.decode(
            baseline_suffix_top1.detach().cpu().tolist()
        ),
        "changed_suffix_top1_text": tokenizer.decode(
            changed_suffix_top1.detach().cpu().tolist()
        ),
        "first_drifting_layer": first_drifting_layer,
        "all_prefix_kv_exact_equal": first_drifting_layer is None,
        "vanilla_prefix_kv_reuse_is_exact": first_drifting_layer is None,
        "layers": layers,
        "prefix_isolated_cache_experiment": {
            "latency_measurement": "single-sample-steady-state-after-path-warmup",
            "all_prefix_kv_exact_equal_after_suffix_edit": isolated_prefix_exact,
            "full_recompute_latency_ms": isolated_elapsed * 1e3,
            "full_recompute_latency_samples_ms": [
                sample * 1e3 for sample in isolated_latency_samples
            ],
            "one_time_prefix_prefill_latency_ms": prefix_prefill_elapsed * 1e3,
            "one_time_prefix_prefill_latency_samples_ms": [
                sample * 1e3 for sample in prefix_prefill_latency_samples
            ],
            "cached_suffix_latency_ms": cached_suffix_elapsed * 1e3,
            "cached_suffix_latency_samples_ms": [
                sample * 1e3 for sample in cached_suffix_latency_samples
            ],
            "suffix_top1_agreement": _top1_agreement(
                isolated_suffix_top1,
                cached_suffix_top1,
            ),
            "suffix_logit_drift": cached_logit_drift,
            "cached_suffix_top1_text": tokenizer.decode(
                cached_suffix_top1[0].detach().cpu().tolist()
            ),
            "full_recompute_suffix_top1_text": tokenizer.decode(
                isolated_suffix_top1[0].detach().cpu().tolist()
            ),
        },
    }
