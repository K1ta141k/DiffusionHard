"""Apple Silicon execution path specialized for the pinned MDLM checkpoint."""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Optional, Sequence

from .mdlm import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    _load_mdlm_model,
    _resolve_device,
    _run_ddpm_candidate_cache_sampler,
    _synchronize,
    legacy_attention_offsets,
)


def _reveal_steps_from_uniforms(
    uniforms: Any,
    *,
    steps: int,
    sampling_epsilon: float,
) -> Any:
    """Map independent reveal draws to their ancestral DDPM transitions."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 <= sampling_epsilon < 1.0:
        raise ValueError("sampling_epsilon must be in [0, 1)")
    import torch

    uniforms = uniforms.detach().cpu()
    reveal_steps = torch.full(uniforms.shape, steps, dtype=torch.long)
    reveal_mass = 1.0 - sampling_epsilon
    reveal = uniforms < reveal_mass
    reveal_steps[reveal] = torch.floor(
        uniforms[reveal] * steps / reveal_mass
    ).long().clamp(max=steps - 1)
    return reveal_steps


def _sample_categorical_from_logits(
    logits: Any,
    *,
    mask_token_id: int,
    uniforms: Optional[Any] = None,
) -> Any:
    """Sample the official exponential race without materializing softmax."""
    import torch

    scores = logits.float().clone()
    scores[..., mask_token_id] = -torch.inf
    if uniforms is None:
        uniforms = torch.rand_like(scores)
    if uniforms.shape != scores.shape:
        raise ValueError("uniforms must have the same shape as logits")
    exponential = 1e-10 - torch.log(uniforms + 1e-10)
    return (scores - torch.log(exponential)).argmax(dim=-1)


def _run_event_driven_candidate_sampler(
    logits_function: Callable[[Any, Any], Any],
    input_ids: Any,
    *,
    mask_token_id: int,
    steps: int,
    device: str,
    reveal_seed: int,
    sampling_epsilon: float = 1e-5,
    reveal_uniforms: Optional[Any] = None,
) -> tuple[Any, dict[str, object]]:
    """Generate by executing only transitions that reveal at least one token.

    For the non-time-conditioned MDLM checkpoint, empty transitions have no
    effect on the model distribution. Candidate tokens for positions that do
    not reveal are also discarded after the next state change. This permits
    delaying evaluation until a reveal event and projecting only those rows.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("event-driven sampler currently requires batch size one")
    if reveal_seed < 0:
        raise ValueError("reveal_seed must be non-negative")
    import torch

    positions = input_ids.shape[1]
    if reveal_uniforms is None:
        generator = torch.Generator(device="cpu").manual_seed(reveal_seed)
        reveal_uniforms = torch.rand(positions, generator=generator)
    elif reveal_uniforms.numel() != positions:
        raise ValueError("reveal_uniforms must contain one value per position")
    reveal_steps = _reveal_steps_from_uniforms(
        reveal_uniforms.reshape(positions),
        steps=steps,
        sampling_epsilon=sampling_epsilon,
    )

    changed_tokens: list[int] = []
    active_tokens: list[int] = []
    active_count = positions
    model_evaluations = 0
    projected_rows = 0

    _synchronize(device)
    started = time.perf_counter()
    for transition_id in range(steps):
        if active_count == 0:
            break
        active_tokens.append(active_count)
        selected_cpu = torch.nonzero(
            reveal_steps.eq(transition_id), as_tuple=False
        ).squeeze(1)
        changed_count = int(selected_cpu.numel())
        changed_tokens.append(changed_count)
        if changed_count:
            selected = selected_cpu.to(device)
            logits = logits_function(input_ids, selected)
            candidates = _sample_categorical_from_logits(
                logits,
                mask_token_id=mask_token_id,
            )
            input_ids[0].index_copy_(0, selected, candidates)
            model_evaluations += 1
            projected_rows += changed_count
            active_count -= changed_count

    final_cpu = torch.nonzero(reveal_steps.eq(steps), as_tuple=False).squeeze(1)
    final_count = int(final_cpu.numel())
    if final_count:
        selected = final_cpu.to(device)
        logits = logits_function(input_ids, selected).float().clone()
        logits[..., mask_token_id] = -torch.inf
        input_ids[0].index_copy_(0, selected, logits.argmax(dim=-1))
        model_evaluations += 1
        projected_rows += final_count
        active_count -= final_count

    _synchronize(device)
    wall_latency_s = time.perf_counter() - started
    full_head_rows = model_evaluations * positions
    return input_ids, {
        "sampling_transitions": len(active_tokens),
        "model_evaluations": model_evaluations,
        "empty_transitions_elided": sum(count == 0 for count in changed_tokens),
        "transition_active_tokens": active_tokens,
        "transition_changed_tokens": changed_tokens,
        "selected_output_rows": projected_rows,
        "full_output_rows_equivalent": full_head_rows,
        "output_row_reduction": (
            1.0 - projected_rows / full_head_rows if full_head_rows else 0.0
        ),
        "wall_latency_ms": wall_latency_s * 1e3,
        "output_tokens_per_second": positions / wall_latency_s,
        "all_tokens_committed": active_count == 0,
        "correctness_class": "distribution-equivalent",
    }


class _AppleSelectedLogits:
    """Run the full transformer while projecting only selected positions."""

    def __init__(self, model: Any, *, device: str) -> None:
        import torch
        import torch.nn.functional as functional

        if bool(model.config.time_conditioning):
            raise ValueError("event-driven sampling requires time_conditioning=false")
        self.model = model
        self.device = device
        with torch.inference_mode():
            sigma = torch.zeros(1, device=device)
            self.condition = functional.silu(model.backbone.sigma_map(sigma))
            final = model.backbone.output_layer
            shift, scale = final.adaLN_modulation(self.condition).chunk(2, dim=-1)
            self.final_shift = shift
            self.final_scale = scale

    def __call__(self, input_ids: Any, selected_positions: Any) -> Any:
        import torch.nn.functional as functional

        backbone = self.model.backbone
        hidden = backbone.vocab_embed(input_ids)
        rotary_cos_sin = backbone.rotary_emb(hidden)
        for block in backbone.blocks:
            hidden = block(
                hidden,
                rotary_cos_sin,
                self.condition,
                seqlens=None,
            )

        selected = hidden[0].index_select(0, selected_positions)
        final = backbone.output_layer
        selected = functional.layer_norm(
            selected.float(),
            [final.norm_final.dim],
        )
        selected = selected * final.norm_final.weight[None, :]
        selected = selected * (1 + self.final_scale) + self.final_shift
        return final.linear(selected)


def _extract_logits(output: Any) -> Any:
    return output.logits if hasattr(output, "logits") else output


def benchmark_mdlm_apple(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: str = "auto",
    canvas_tokens: int = 64,
    steps: int = 64,
    seeds: Sequence[int] = (0, 1, 2),
    local_files_only: bool = False,
) -> dict[str, object]:
    """Benchmark the reference and event-driven MPS sampling paths."""
    if canvas_tokens <= 0 or steps <= 0:
        raise ValueError("canvas_tokens and steps must be positive")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    import torch

    resolved_device = _resolve_device(device)
    if resolved_device != "mps":
        raise ValueError("Apple benchmark requires the MPS device")
    model = _load_mdlm_model(
        model_id=model_id,
        revision=revision,
        device=resolved_device,
        local_files_only=local_files_only,
    )
    if canvas_tokens > int(model.config.model_length):
        raise ValueError("canvas_tokens exceeds the checkpoint context length")
    mask_token_id = int(model.config.vocab_size) - 1
    selected_logits = _AppleSelectedLogits(model, device=resolved_device)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            local_files_only=local_files_only,
        )
    except OSError:
        tokenizer = None

    def blank() -> Any:
        return torch.full(
            (1, canvas_tokens),
            mask_token_id,
            dtype=torch.long,
            device=resolved_device,
        )

    with torch.inference_mode():
        model(input_ids=blank(), timesteps=torch.ones(1, device=resolved_device))
        _synchronize(resolved_device)

        validation_ids = blank()
        validation_ids[:, ::2] = torch.arange(
            (canvas_tokens + 1) // 2,
            device=resolved_device,
        )
        validation_positions = torch.nonzero(
            validation_ids[0].eq(mask_token_id), as_tuple=False
        ).squeeze(1)
        full = _extract_logits(
            model(
                input_ids=validation_ids,
                timesteps=torch.ones(1, device=resolved_device),
            )
        )[0].index_select(0, validation_positions)
        selected = selected_logits(validation_ids, validation_positions)
        _synchronize(resolved_device)
        maximum_logit_error = float((full - selected).abs().max().item())
        selected_top1_agreement = float(
            full.argmax(dim=-1).eq(selected.argmax(dim=-1)).float().mean().item()
        )

    samples = []
    for seed in seeds:
        def baseline_once() -> tuple[Any, dict[str, object]]:
            torch.manual_seed(seed)
            _synchronize(resolved_device)
            started = time.perf_counter()
            output, measurements, metadata = _run_ddpm_candidate_cache_sampler(
                model,
                blank(),
                mask_token_id=mask_token_id,
                steps=steps,
                device=resolved_device,
            )
            _synchronize(resolved_device)
            wall_s = time.perf_counter() - started
            return output, {
                **metadata,
                "wall_latency_ms": wall_s * 1e3,
                "output_tokens_per_second": canvas_tokens / wall_s,
                "measured_forward_latency_ms": sum(
                    measurement.elapsed_s for measurement in measurements
                )
                * 1e3,
            }

        def optimized_once() -> tuple[Any, dict[str, object]]:
            torch.manual_seed(seed)
            return _run_event_driven_candidate_sampler(
                selected_logits,
                blank(),
                mask_token_id=mask_token_id,
                steps=steps,
                device=resolved_device,
                reveal_seed=seed,
            )

        with torch.inference_mode():
            with legacy_attention_offsets():
                baseline_once()
                legacy_output, legacy = baseline_once()
            baseline_once()
            baseline_output, baseline = baseline_once()
            optimized_once()
            optimized_output, optimized = optimized_once()
        samples.append(
            {
                "seed": seed,
                "legacy_synchronized_attention": legacy,
                "sync_free_attention": baseline,
                "optimized": optimized,
                "sync_free_matches_legacy": bool(
                    legacy_output.eq(baseline_output).all().item()
                ),
                "legacy_generated_token_ids": (
                    legacy_output[0].detach().cpu().tolist()
                ),
                "optimized_generated_token_ids": (
                    optimized_output[0].detach().cpu().tolist()
                ),
                "legacy_generated_text": (
                    tokenizer.decode(legacy_output[0].detach().cpu().tolist())
                    if tokenizer is not None
                    else None
                ),
                "optimized_generated_text": (
                    tokenizer.decode(optimized_output[0].detach().cpu().tolist())
                    if tokenizer is not None
                    else None
                ),
            }
        )

    legacy_latencies = [
        float(sample["legacy_synchronized_attention"]["wall_latency_ms"])
        for sample in samples
    ]
    sync_free_latencies = [
        float(sample["sync_free_attention"]["wall_latency_ms"])
        for sample in samples
    ]
    optimized_latencies = [
        float(sample["optimized"]["wall_latency_ms"]) for sample in samples
    ]
    legacy_median = statistics.median(legacy_latencies)
    sync_free_median = statistics.median(sync_free_latencies)
    optimized_median = statistics.median(optimized_latencies)
    return {
        "model_id": model_id,
        "revision": revision,
        "device": resolved_device,
        "torch_version": torch.__version__,
        "canvas_tokens": canvas_tokens,
        "steps": steps,
        "seeds": list(seeds),
        "optimization": (
            "single-sequence-attention-without-host-sync; event-driven-reveal; "
            "selected-row-output-head; logits-space-exponential-race"
        ),
        "selected_head_validation": {
            "maximum_absolute_logit_error": maximum_logit_error,
            "top1_agreement": selected_top1_agreement,
        },
        "samples": samples,
        "summary": {
            "legacy_median_wall_latency_ms": legacy_median,
            "sync_free_attention_median_wall_latency_ms": sync_free_median,
            "optimized_median_wall_latency_ms": optimized_median,
            "legacy_median_output_tokens_per_second": (
                canvas_tokens * 1e3 / legacy_median
            ),
            "sync_free_attention_median_output_tokens_per_second": (
                canvas_tokens * 1e3 / sync_free_median
            ),
            "optimized_median_output_tokens_per_second": (
                canvas_tokens * 1e3 / optimized_median
            ),
            "attention_sync_removal_speedup": legacy_median / sync_free_median,
            "event_sampler_speedup": sync_free_median / optimized_median,
            "total_median_speedup": legacy_median / optimized_median,
            "sync_free_attention_exact_outputs": sum(
                bool(sample["sync_free_matches_legacy"])
                for sample in samples
            ),
        },
        "scope": (
            "local MPS benchmark; distribution-equivalent sampling; outputs "
            "are not expected to match pathwise because RNG streams differ"
        ),
    }
