"""macOS-compatible MDLM smoke sampling and trace lowering.

The published MDLM checkpoint imports NVIDIA FlashAttention unconditionally.
For tracing on Apple Silicon we install a small API-compatible shim backed by
PyTorch scaled-dot-product attention. The checkpoint weights and model graph
remain unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
import importlib.machinery
import math
import sys
import time
import types
from typing import Any, List, Optional

from .ir import DiffusionStep, Operation, TensorAccess, WorkloadTrace
from .trace import SCHEMA_VERSION

DEFAULT_MODEL_ID = "kuleshov-group/mdlm-owt"
DEFAULT_REVISION = "d0958fa851335ece6c15260ce0025f030673c0fb"

_PREFIX_ISOLATION_TOKENS: contextvars.ContextVar[int] = contextvars.ContextVar(
    "mdlm_prefix_isolation_tokens",
    default=0,
)
_BLOCK_ISOLATION_ENDS: contextvars.ContextVar[tuple[int, ...]] = (
    contextvars.ContextVar(
        "mdlm_block_isolation_ends",
        default=(),
    )
)


@contextmanager
def prefix_isolation(prefix_tokens: int):
    """Prevent fixed-prefix queries from attending to the active suffix.

    This affects the macOS scaled-dot-product-attention compatibility path used
    by the research harness. It does not modify checkpoint weights.
    """
    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be non-negative")
    token = _PREFIX_ISOLATION_TOKENS.set(prefix_tokens)
    try:
        yield
    finally:
        _PREFIX_ISOLATION_TOKENS.reset(token)


@contextmanager
def block_isolation(block_ends: List[int]):
    """Apply block-causal attention to completed block end positions.

    Each completed block may attend to itself and every earlier block. Tokens
    after the final listed boundary form the active bidirectional block.
    """
    normalized = tuple(int(end) for end in block_ends)
    if any(end <= 0 for end in normalized):
        raise ValueError("block end positions must be positive")
    if any(left >= right for left, right in zip(normalized, normalized[1:])):
        raise ValueError("block end positions must be strictly increasing")
    token = _BLOCK_ISOLATION_ENDS.set(normalized)
    try:
        yield
    finally:
        _BLOCK_ISOLATION_ENDS.reset(token)


@dataclass(frozen=True)
class MDLMSpec:
    """Architecture fields needed to lower an MDLM run into the trace IR."""

    model_id: str
    revision: str
    hidden_size: int
    layers: int
    heads: int
    vocab_size: int
    parameter_count: int
    model_weight_bytes: int
    bytes_per_element: int = 4


@dataclass(frozen=True)
class MDLMStepMeasurement:
    step_id: int
    active_tokens: int
    changed_tokens: int
    elapsed_s: float
    sigma: float


def _tensor_access(
    name: str,
    size_bytes: int,
    access: str,
    lifetime: str,
    category: str,
) -> TensorAccess:
    return TensorAccess(  # type: ignore[arg-type]
        name=name,
        size_bytes=size_bytes,
        access=access,
        lifetime=lifetime,
        category=category,
    )


def lower_mdlm_measurements(
    spec: MDLMSpec,
    measurements: List[MDLMStepMeasurement],
    *,
    canvas_tokens: int,
    batch_size: int = 1,
    device: str = "unknown",
    sampler: str = "confidence-unmask-smoke",
) -> WorkloadTrace:
    """Build a hardware-neutral trace from a measured denoising run."""
    if canvas_tokens <= 0 or batch_size <= 0:
        raise ValueError("canvas_tokens and batch_size must be positive")
    if not measurements:
        raise ValueError("at least one step measurement is required")

    element_bytes = spec.bytes_per_element
    hidden_bytes = batch_size * canvas_tokens * spec.hidden_size * element_bytes
    qkv_bytes = 3 * hidden_bytes
    attention_score_bytes = (
        batch_size * spec.heads * canvas_tokens * canvas_tokens * element_bytes
    )
    logits_bytes = batch_size * canvas_tokens * spec.vocab_size * element_bytes
    embedding_weight_bytes = spec.vocab_size * spec.hidden_size * element_bytes
    output_weight_bytes = spec.vocab_size * spec.hidden_size * element_bytes
    block_weight_bytes = max(
        0,
        spec.model_weight_bytes - embedding_weight_bytes - output_weight_bytes,
    ) // spec.layers

    steps: List[DiffusionStep] = []
    for measurement in measurements:
        if measurement.active_tokens < 0 or measurement.changed_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if measurement.changed_tokens > measurement.active_tokens:
            raise ValueError("changed_tokens cannot exceed active_tokens")

        operations: List[Operation] = [
            Operation(
                name="token_embedding",
                flops=0,
                reads=[
                    _tensor_access(
                        "token_embedding_weights",
                        embedding_weight_bytes,
                        "read",
                        "request",
                        "weight",
                    )
                ],
                writes=[
                    _tensor_access(
                        "canvas_hidden_a",
                        hidden_bytes,
                        "write",
                        "canvas",
                        "canvas_state",
                    )
                ],
                parallelism=batch_size * canvas_tokens,
            )
        ]

        for layer in range(spec.layers):
            source = "canvas_hidden_a" if layer % 2 == 0 else "canvas_hidden_b"
            target = "canvas_hidden_b" if layer % 2 == 0 else "canvas_hidden_a"
            # QKV + attention output + 4x MLP projections + bidirectional
            # attention. Multiply and add count as two FLOPs.
            layer_flops = (
                24 * batch_size * canvas_tokens * spec.hidden_size**2
                + 4
                * batch_size
                * canvas_tokens**2
                * spec.hidden_size
            )
            operations.append(
                Operation(
                    name="transformer_block_%02d" % layer,
                    flops=layer_flops,
                    reads=[
                        _tensor_access(
                            "block_%02d_weights" % layer,
                            block_weight_bytes,
                            "read",
                            "request",
                            "weight",
                        ),
                        _tensor_access(
                            source,
                            hidden_bytes,
                            "read",
                            "canvas",
                            "canvas_state",
                        ),
                    ],
                    writes=[
                        _tensor_access(
                            target,
                            hidden_bytes,
                            "write",
                            "canvas",
                            "canvas_state",
                        ),
                        _tensor_access(
                            "step_%03d_block_%02d_qkv"
                            % (measurement.step_id, layer),
                            qkv_bytes,
                            "write",
                            "operation",
                            "activation",
                        ),
                        _tensor_access(
                            "step_%03d_block_%02d_attention"
                            % (measurement.step_id, layer),
                            attention_score_bytes,
                            "write",
                            "operation",
                            "activation",
                        ),
                    ],
                    parallelism=batch_size * canvas_tokens,
                    metadata={"layer": layer, "attention": "bidirectional"},
                )
            )

        final_hidden = (
            "canvas_hidden_a" if spec.layers % 2 == 0 else "canvas_hidden_b"
        )
        operations.extend(
            [
                Operation(
                    name="full_vocabulary_projection",
                    flops=(
                        2
                        * batch_size
                        * canvas_tokens
                        * spec.hidden_size
                        * spec.vocab_size
                    ),
                    reads=[
                        _tensor_access(
                            final_hidden,
                            hidden_bytes,
                            "read",
                            "canvas",
                            "canvas_state",
                        ),
                        _tensor_access(
                            "output_projection_weights",
                            output_weight_bytes,
                            "read",
                            "request",
                            "weight",
                        ),
                    ],
                    writes=[
                        _tensor_access(
                            "step_%03d_logits" % measurement.step_id,
                            logits_bytes,
                            "write",
                            "step",
                            "logit",
                        )
                    ],
                    parallelism=batch_size * canvas_tokens,
                    metadata={
                        "active_tokens": measurement.active_tokens,
                        "computed_tokens": batch_size * canvas_tokens,
                        "optimization_candidate": "masked-only-output-head",
                    },
                ),
                Operation(
                    name="confidence_unmask",
                    flops=4
                    * measurement.active_tokens
                    * spec.vocab_size,
                    reads=[
                        _tensor_access(
                            "step_%03d_logits" % measurement.step_id,
                            logits_bytes,
                            "read",
                            "step",
                            "logit",
                        )
                    ],
                    writes=[
                        _tensor_access(
                            "active_token_bitmap",
                            math.ceil(canvas_tokens / 8),
                            "write",
                            "canvas",
                            "metadata",
                        )
                    ],
                    parallelism=max(1, measurement.active_tokens),
                    metadata={"committed_tokens": measurement.changed_tokens},
                ),
            ]
        )
        steps.append(
            DiffusionStep(
                canvas_id=0,
                step_id=measurement.step_id,
                active_tokens=measurement.active_tokens,
                changed_tokens=measurement.changed_tokens,
                operations=operations,
                metadata={
                    "measured_step_latency_ms": measurement.elapsed_s * 1e3,
                    "sigma": measurement.sigma,
                    "device": device,
                },
            )
        )

    return WorkloadTrace(
        schema_version=SCHEMA_VERSION,
        workload_name="mdlm-owt-real",
        steps=steps,
        metadata={
            "model_id": spec.model_id,
            "revision": spec.revision,
            "parameter_count": spec.parameter_count,
            "model_weight_bytes": spec.model_weight_bytes,
            "weight_bits": spec.bytes_per_element * 8,
            "layers": spec.layers,
            "heads": spec.heads,
            "hidden_size": spec.hidden_size,
            "vocab_size": spec.vocab_size,
            "canvas_tokens": canvas_tokens,
            "batch_size": batch_size,
            "device": device,
            "sampler": sampler,
            "attention_backend": "torch-scaled-dot-product-attention",
            "provenance": "measured-forward-pass-and-derived-operation-counts",
        },
    )


def install_flash_attention_compat() -> None:
    """Provide the two FlashAttention APIs used by the published MDLM code."""
    if "flash_attn" in sys.modules:
        return
    try:
        import flash_attn  # noqa: F401

        return
    except ImportError:
        pass

    import torch
    import torch.nn.functional as functional

    flash_module = types.ModuleType("flash_attn")
    flash_module.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None
    )
    layers_module = types.ModuleType("flash_attn.layers")
    layers_module.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn.layers", loader=None, is_package=True
    )
    rotary_module = types.ModuleType("flash_attn.layers.rotary")
    rotary_module.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn.layers.rotary", loader=None
    )
    interface_module = types.ModuleType("flash_attn.flash_attn_interface")
    interface_module.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn.flash_attn_interface", loader=None
    )
    flash_module._diffusion_accel_compat = True  # type: ignore[attr-defined]

    def apply_rotary_emb_qkv_(qkv: Any, cos: Any, sin: Any) -> Any:
        if qkv.ndim != 5:
            raise ValueError("MDLM compatibility path expects [B, S, 3, H, D]")
        q_and_k = qkv[:, :, :2]
        half = q_and_k.shape[-1] // 2
        cos_view = cos[None, :, None, None, :]
        sin_view = sin[None, :, None, None, :]
        first = q_and_k[..., :half].clone()
        second = q_and_k[..., half:].clone()
        q_and_k[..., :half] = first * cos_view - second * sin_view
        q_and_k[..., half:] = second * cos_view + first * sin_view
        return qkv

    def flash_attn_varlen_qkvpacked_func(
        qkv: Any,
        cu_seqlens: Any,
        max_seqlen: int,
        dropout_p: float,
        causal: bool = False,
    ) -> Any:
        del max_seqlen
        outputs = []
        for index in range(cu_seqlens.numel() - 1):
            start = int(cu_seqlens[index].item())
            end = int(cu_seqlens[index + 1].item())
            query, key, value = qkv[start:end].unbind(dim=1)
            attention_bias = None
            sequence_tokens = end - start
            block_ends = _BLOCK_ISOLATION_ENDS.get()
            if not block_ends:
                isolated_prefix = _PREFIX_ISOLATION_TOKENS.get()
                block_ends = (isolated_prefix,) if isolated_prefix else ()
            block_ends = tuple(
                min(block_end, sequence_tokens)
                for block_end in block_ends
                if block_end > 0
            )
            if block_ends:
                attention_bias = torch.zeros(
                    (sequence_tokens, sequence_tokens),
                    dtype=query.dtype,
                    device=query.device,
                )
                block_start = 0
                for block_end in block_ends:
                    attention_bias[
                        block_start:block_end,
                        block_end:,
                    ] = -torch.inf
                    block_start = block_end
            output = functional.scaled_dot_product_attention(
                query.transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0),
                attn_mask=attention_bias,
                dropout_p=dropout_p,
                is_causal=causal,
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)

    rotary_module.apply_rotary_emb_qkv_ = apply_rotary_emb_qkv_  # type: ignore[attr-defined]
    interface_module.flash_attn_varlen_qkvpacked_func = (  # type: ignore[attr-defined]
        flash_attn_varlen_qkvpacked_func
    )
    flash_module.layers = layers_module  # type: ignore[attr-defined]
    flash_module.flash_attn_interface = interface_module  # type: ignore[attr-defined]
    layers_module.rotary = rotary_module  # type: ignore[attr-defined]
    for module in (
        flash_module,
        layers_module,
        rotary_module,
        interface_module,
    ):
        sys.modules[module.__name__] = module


def _resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _synchronize(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _patch_mdlm_eager_kernels(model: Any, device: str) -> None:
    """Replace CUDA-fuser TorchScript helpers when running on non-CUDA devices."""
    if device.startswith("cuda"):
        return
    module = sys.modules[model.__class__.__module__]

    def modulate_eager(x: Any, shift: Any, scale: Any) -> Any:
        return x * (1 + scale) + shift

    def residual_eager(
        x: Any,
        bias: Any,
        scale: Any,
        residual: Any,
        probability: float,
    ) -> Any:
        return module.bias_dropout_add_scale(
            x,
            bias,
            scale,
            residual,
            probability,
            False,
        )

    module.modulate_fused = modulate_eager
    module.bias_dropout_add_scale_fused_inference = residual_eager


def _load_mdlm_model(
    *,
    model_id: str,
    revision: str,
    device: str,
    local_files_only: bool,
) -> Any:
    install_flash_attention_compat()
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        local_files_only=local_files_only,
    ).eval()
    _patch_mdlm_eager_kernels(model, device)
    return model.to(device)


def _fake_quantize_weights_int8(
    model: Any,
    *,
    preserve_output_head: bool = False,
    only_output_head: bool = False,
) -> int:
    """Apply per-output-channel symmetric INT8 fake quantization in place."""
    import torch

    quantized_parameters = 0
    if preserve_output_head and only_output_head:
        raise ValueError("cannot preserve and exclusively quantize the output head")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim < 2 or not parameter.is_floating_point():
                continue
            is_output_head = name.startswith("backbone.output_layer")
            if preserve_output_head and is_output_head:
                continue
            if only_output_head and not is_output_head:
                continue
            reduce_dimensions = tuple(range(1, parameter.ndim))
            maximum = parameter.abs().amax(dim=reduce_dimensions, keepdim=True)
            scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
            parameter.copy_(
                torch.round(parameter / scale).clamp(-127, 127) * scale
            )
            quantized_parameters += parameter.numel()
    return quantized_parameters


def validate_mdlm_int8(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: str = "auto",
    canvas_tokens: int = 64,
    seed: int = 0,
    preserve_output_head: bool = False,
    only_output_head: bool = False,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Compare FP32 and fake-quantized INT8 outputs on a masked canvas."""
    if canvas_tokens <= 0:
        raise ValueError("canvas_tokens must be positive")
    if preserve_output_head and only_output_head:
        raise ValueError("cannot preserve and exclusively quantize the output head")
    import torch
    import torch.nn.functional as functional

    resolved_device = _resolve_device(device)
    torch.manual_seed(seed)
    model = _load_mdlm_model(
        model_id=model_id,
        revision=revision,
        device=resolved_device,
        local_files_only=local_files_only,
    )
    config = model.config
    if canvas_tokens > int(config.model_length):
        raise ValueError(
            "canvas_tokens exceeds checkpoint context length %d"
            % config.model_length
        )

    mask_token_id = int(config.vocab_size) - 1
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(
        0,
        mask_token_id,
        (1, canvas_tokens),
        generator=generator,
        dtype=torch.long,
    )
    input_ids[:, ::2] = mask_token_id
    input_ids = input_ids.to(resolved_device)
    timestep = torch.ones(1, device=resolved_device)

    with torch.inference_mode():
        baseline_output = model(input_ids=input_ids, timesteps=timestep)
        _synchronize(resolved_device)
        baseline = (
            baseline_output.logits
            if hasattr(baseline_output, "logits")
            else baseline_output
        )[0, ::2].float()
        quantized_parameters = _fake_quantize_weights_int8(
            model,
            preserve_output_head=preserve_output_head,
            only_output_head=only_output_head,
        )
        quantized_output = model(input_ids=input_ids, timesteps=timestep)
        _synchronize(resolved_device)
        quantized = (
            quantized_output.logits
            if hasattr(quantized_output, "logits")
            else quantized_output
        )[0, ::2].float()

        difference = quantized - baseline
        baseline_for_top1 = baseline.clone()
        quantized_for_top1 = quantized.clone()
        baseline_for_top1[:, mask_token_id] = -torch.inf
        quantized_for_top1[:, mask_token_id] = -torch.inf
        top1_agreement = (
            baseline_for_top1.argmax(dim=-1)
            .eq(quantized_for_top1.argmax(dim=-1))
            .float()
            .mean()
        )
        cosine = functional.cosine_similarity(
            baseline.flatten(), quantized.flatten(), dim=0
        )
        rmse = difference.square().mean().sqrt()
        baseline_rms = baseline.square().mean().sqrt()
        baseline_log_probabilities = torch.log_softmax(
            baseline_for_top1,
            dim=-1,
        )
        quantized_log_probabilities = torch.log_softmax(
            quantized_for_top1,
            dim=-1,
        )
        baseline_probabilities = baseline_log_probabilities.exp()
        quantized_probabilities = quantized_log_probabilities.exp()
        total_variation_per_position = 0.5 * (
            baseline_probabilities - quantized_probabilities
        ).abs().sum(dim=-1)
        log_probability_delta = torch.where(
            baseline_probabilities > 0,
            baseline_log_probabilities - quantized_log_probabilities,
            torch.zeros_like(baseline_log_probabilities),
        )
        kl_per_position = (
            baseline_probabilities * log_probability_delta
        ).sum(dim=-1)

    return {
        "model_id": model_id,
        "revision": revision,
        "device": resolved_device,
        "canvas_tokens": canvas_tokens,
        "masked_tokens_compared": (canvas_tokens + 1) // 2,
        "quantization": "per-output-channel-symmetric-int8-fake-quantization",
        "preserve_output_head_fp32": preserve_output_head,
        "only_output_head_int8": only_output_head,
        "quantized_parameter_count": quantized_parameters,
        "top1_agreement": float(top1_agreement.item()),
        "cosine_similarity": float(cosine.item()),
        "mean_absolute_logit_error": float(difference.abs().mean().item()),
        "normalized_rmse": float((rmse / baseline_rms).item()),
        "mean_total_variation": float(total_variation_per_position.mean().item()),
        "maximum_total_variation": float(total_variation_per_position.max().item()),
        "mean_kl_divergence_nats": float(kl_per_position.mean().item()),
        "maximum_kl_divergence_nats": float(kl_per_position.max().item()),
    }


def validate_mdlm_output_head_int8_generation(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: str = "auto",
    canvas_tokens: int = 64,
    steps: int = 64,
    seeds: Optional[List[int]] = None,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Compare full candidate-cached FP32 and output-head INT8 generations."""
    if canvas_tokens <= 0 or steps <= 0:
        raise ValueError("canvas_tokens and steps must be positive")
    selected_seeds = [0, 1, 2, 3, 4] if seeds is None else list(seeds)
    if not selected_seeds:
        raise ValueError("at least one seed is required")

    import torch

    resolved_device = _resolve_device(device)
    model = _load_mdlm_model(
        model_id=model_id,
        revision=revision,
        device=resolved_device,
        local_files_only=local_files_only,
    )
    config = model.config
    if canvas_tokens > int(config.model_length):
        raise ValueError(
            "canvas_tokens exceeds checkpoint context length %d"
            % config.model_length
        )
    mask_token_id = int(config.vocab_size) - 1

    def run(seed: int) -> tuple[Any, dict[str, object]]:
        torch.manual_seed(seed)
        input_ids = torch.full(
            (1, canvas_tokens),
            mask_token_id,
            dtype=torch.long,
            device=resolved_device,
        )
        generated, _, metadata = _run_ddpm_candidate_cache_sampler(
            model,
            input_ids,
            mask_token_id=mask_token_id,
            steps=steps,
            device=resolved_device,
        )
        return generated.detach().cpu(), metadata

    baseline = []
    with torch.inference_mode():
        for seed in selected_seeds:
            baseline.append(run(seed))
        quantized_parameters = _fake_quantize_weights_int8(
            model,
            only_output_head=True,
        )
        quantized = [run(seed) for seed in selected_seeds]

    samples = []
    exact_generations = 0
    matching_tokens = 0
    for seed, (baseline_run, quantized_run) in zip(
        selected_seeds,
        zip(baseline, quantized),
    ):
        baseline_ids, baseline_metadata = baseline_run
        quantized_ids, quantized_metadata = quantized_run
        token_matches = int(baseline_ids.eq(quantized_ids).sum().item())
        exact = token_matches == canvas_tokens
        exact_generations += int(exact)
        matching_tokens += token_matches
        samples.append(
            {
                "seed": seed,
                "exact_generated_tokens": exact,
                "token_agreement": token_matches / canvas_tokens,
                "matching_tokens": token_matches,
                "transition_active_tokens_exact": (
                    baseline_metadata["transition_active_tokens"]
                    == quantized_metadata["transition_active_tokens"]
                ),
                "transition_changed_tokens_exact": (
                    baseline_metadata["transition_changed_tokens"]
                    == quantized_metadata["transition_changed_tokens"]
                ),
                "baseline_model_evaluations": baseline_metadata[
                    "model_evaluations"
                ],
                "quantized_model_evaluations": quantized_metadata[
                    "model_evaluations"
                ],
                "baseline_candidate_cache_hits": baseline_metadata[
                    "candidate_cache_hits"
                ],
                "quantized_candidate_cache_hits": quantized_metadata[
                    "candidate_cache_hits"
                ],
                "baseline_token_ids": baseline_ids[0].tolist(),
                "quantized_token_ids": quantized_ids[0].tolist(),
            }
        )
    return {
        "model_id": model_id,
        "revision": revision,
        "device": resolved_device,
        "canvas_tokens": canvas_tokens,
        "steps": steps,
        "sampler": "candidate-cached-ddpm",
        "quantization": "output-head-only-per-channel-symmetric-int8",
        "quantized_parameter_count": quantized_parameters,
        "seeds": selected_seeds,
        "samples": samples,
        "aggregate": {
            "exact_generations": exact_generations,
            "total_generations": len(selected_seeds),
            "exact_generation_rate": exact_generations / len(selected_seeds),
            "token_agreement": (
                matching_tokens / (len(selected_seeds) * canvas_tokens)
            ),
            "passed": exact_generations == len(selected_seeds),
        },
        "scope": (
            "same-seed-pathwise-generation-gate; small-unprompted-sample; "
            "fake-quantized-weights"
        ),
    }


def _sample_categorical(probabilities: Any) -> Any:
    """Official MDLM exponential-race categorical sampler."""
    import torch

    exponential = 1e-10 - torch.log(torch.rand_like(probabilities) + 1e-10)
    return (probabilities / exponential).argmax(dim=-1)


def _mdlm_log_probabilities(logits: Any, mask_token_id: int) -> Any:
    """Apply the checkpoint's SUBS output parameterization for masked tokens."""
    import torch

    logits = logits.float()
    logits[:, :, mask_token_id] = -torch.inf
    return torch.log_softmax(logits, dim=-1)


def _run_confidence_smoke_sampler(
    model: Any,
    input_ids: Any,
    *,
    mask_token_id: int,
    steps: int,
    device: str,
) -> tuple[Any, List[MDLMStepMeasurement]]:
    import torch

    measurements: List[MDLMStepMeasurement] = []
    for step_id in range(steps):
        active_mask = input_ids.eq(mask_token_id)
        active_count = int(active_mask.sum().item())
        if active_count == 0:
            break
        sigma = 1.0 - (step_id / steps)
        timestep = torch.full((1,), sigma, device=device)
        _synchronize(device)
        started = time.perf_counter()
        output = model(input_ids=input_ids, timesteps=timestep)
        _synchronize(device)
        elapsed = time.perf_counter() - started
        logits = output.logits if hasattr(output, "logits") else output
        active_logits = logits[0, active_mask[0]].float()
        active_logits[:, mask_token_id] = -torch.inf
        probabilities = torch.softmax(active_logits, dim=-1)
        confidence, predicted = probabilities.max(dim=-1)

        remaining_steps = steps - step_id
        commit_count = min(
            active_count,
            max(1, math.ceil(active_count / remaining_steps)),
        )
        selected_active = confidence.topk(commit_count).indices
        active_positions = active_mask[0].nonzero(as_tuple=False).squeeze(1)
        selected_positions = active_positions[selected_active]
        input_ids[0, selected_positions] = predicted[selected_active]
        measurements.append(
            MDLMStepMeasurement(
                step_id=step_id,
                active_tokens=active_count,
                changed_tokens=commit_count,
                elapsed_s=elapsed,
                sigma=sigma,
            )
        )
    return input_ids, measurements


def _run_ddpm_sampler(
    model: Any,
    input_ids: Any,
    *,
    mask_token_id: int,
    steps: int,
    device: str,
    sampling_epsilon: float = 1e-5,
) -> tuple[Any, List[MDLMStepMeasurement]]:
    """Run the official ancestral DDPM equations with a log-linear schedule."""
    import torch

    noise_epsilon = 1e-3
    timesteps = torch.linspace(1.0, sampling_epsilon, steps + 1, device=device)
    measurements: List[MDLMStepMeasurement] = []
    for step_id in range(steps):
        active_count = int(input_ids.eq(mask_token_id).sum().item())
        if active_count == 0:
            break
        time_t = timesteps[step_id].reshape(1)
        time_s = timesteps[step_id + 1].reshape(1)
        sigma_t = -torch.log1p(-(1 - noise_epsilon) * time_t)
        sigma_s = -torch.log1p(-(1 - noise_epsilon) * time_s)

        _synchronize(device)
        started = time.perf_counter()
        output = model(input_ids=input_ids, timesteps=sigma_t)
        _synchronize(device)
        elapsed = time.perf_counter() - started
        logits = output.logits if hasattr(output, "logits") else output
        probabilities_x0 = _mdlm_log_probabilities(
            logits,
            mask_token_id,
        ).exp()
        move_t = (1 - torch.exp(-sigma_t))[:, None, None]
        move_s = (1 - torch.exp(-sigma_s))[:, None, None]
        transition = probabilities_x0 * (move_t - move_s)
        transition[:, :, mask_token_id] = move_s[:, :, 0]
        proposal = _sample_categorical(transition)
        next_ids = torch.where(
            input_ids.eq(mask_token_id),
            proposal,
            input_ids,
        )
        changed_count = int(input_ids.ne(next_ids).sum().item())
        input_ids = next_ids
        measurements.append(
            MDLMStepMeasurement(
                step_id=step_id,
                active_tokens=active_count,
                changed_tokens=changed_count,
                elapsed_s=elapsed,
                sigma=float(sigma_t.item()),
            )
        )

    # The upstream sampler performs a final denoising pass to remove any masks.
    active_mask = input_ids.eq(mask_token_id)
    active_count = int(active_mask.sum().item())
    if active_count:
        sigma = -torch.log1p(
            -(1 - noise_epsilon)
            * torch.tensor([sampling_epsilon], device=device)
        )
        _synchronize(device)
        started = time.perf_counter()
        output = model(input_ids=input_ids, timesteps=sigma)
        _synchronize(device)
        elapsed = time.perf_counter() - started
        logits = output.logits if hasattr(output, "logits") else output
        predicted = _mdlm_log_probabilities(logits, mask_token_id).argmax(dim=-1)
        input_ids = torch.where(active_mask, predicted, input_ids)
        measurements.append(
            MDLMStepMeasurement(
                step_id=len(measurements),
                active_tokens=active_count,
                changed_tokens=active_count,
                elapsed_s=elapsed,
                sigma=float(sigma.item()),
            )
        )
    return input_ids, measurements


def _run_ddpm_cache_sampler(
    model: Any,
    input_ids: Any,
    *,
    mask_token_id: int,
    steps: int,
    device: str,
    sampling_epsilon: float = 1e-5,
) -> tuple[Any, List[MDLMStepMeasurement], dict[str, object]]:
    """Run upstream DDPM-cache behavior for a non-time-conditioned model."""
    import torch

    noise_epsilon = 1e-3
    timesteps = torch.linspace(1.0, sampling_epsilon, steps + 1, device=device)
    measurements: List[MDLMStepMeasurement] = []
    probability_cache = None
    cache_hits = 0
    transition_active_tokens: List[int] = []
    transition_changed_tokens: List[int] = []

    for transition_id in range(steps):
        active_count = int(input_ids.eq(mask_token_id).sum().item())
        if active_count == 0:
            break
        transition_active_tokens.append(active_count)
        time_t = timesteps[transition_id].reshape(1)
        time_s = timesteps[transition_id + 1].reshape(1)
        sigma_t = -torch.log1p(-(1 - noise_epsilon) * time_t)

        evaluated = probability_cache is None
        elapsed = 0.0
        if evaluated:
            _synchronize(device)
            started = time.perf_counter()
            output = model(input_ids=input_ids, timesteps=sigma_t)
            _synchronize(device)
            elapsed = time.perf_counter() - started
            logits = output.logits if hasattr(output, "logits") else output
            probability_cache = _mdlm_log_probabilities(
                logits,
                mask_token_id,
            ).exp()
        else:
            cache_hits += 1

        # For the log-linear schedule this is algebraically the same as the
        # upstream cache update's t and t-dt probabilities.
        move_t = ((1 - noise_epsilon) * time_t)[:, None, None]
        move_s = ((1 - noise_epsilon) * time_s)[:, None, None]
        transition = probability_cache * (move_t - move_s)
        transition[:, :, mask_token_id] = move_s[:, :, 0]
        proposal = _sample_categorical(transition)
        next_ids = torch.where(
            input_ids.eq(mask_token_id),
            proposal,
            input_ids,
        )
        changed_count = int(input_ids.ne(next_ids).sum().item())
        transition_changed_tokens.append(changed_count)
        if evaluated:
            measurements.append(
                MDLMStepMeasurement(
                    step_id=transition_id,
                    active_tokens=active_count,
                    changed_tokens=changed_count,
                    elapsed_s=elapsed,
                    sigma=float(sigma_t.item()),
                )
            )
        if changed_count:
            probability_cache = None
        input_ids = next_ids

    active_mask = input_ids.eq(mask_token_id)
    active_count = int(active_mask.sum().item())
    if active_count:
        sigma = -torch.log1p(
            -(1 - noise_epsilon)
            * torch.tensor([sampling_epsilon], device=device)
        )
        _synchronize(device)
        started = time.perf_counter()
        output = model(input_ids=input_ids, timesteps=sigma)
        _synchronize(device)
        elapsed = time.perf_counter() - started
        logits = output.logits if hasattr(output, "logits") else output
        predicted = _mdlm_log_probabilities(logits, mask_token_id).argmax(dim=-1)
        input_ids = torch.where(active_mask, predicted, input_ids)
        measurements.append(
            MDLMStepMeasurement(
                step_id=len(transition_active_tokens),
                active_tokens=active_count,
                changed_tokens=active_count,
                elapsed_s=elapsed,
                sigma=float(sigma.item()),
            )
        )

    metadata: dict[str, object] = {
        "sampling_transitions": len(transition_active_tokens),
        "model_evaluations": len(measurements),
        "probability_cache_hits": cache_hits,
        "transition_active_tokens": transition_active_tokens,
        "transition_changed_tokens": transition_changed_tokens,
        "cache_hit_sampling_overhead_modeled": False,
    }
    return input_ids, measurements, metadata


def _run_ddpm_candidate_cache_sampler(
    model: Any,
    input_ids: Any,
    *,
    mask_token_id: int,
    steps: int,
    device: str,
    sampling_epsilon: float = 1e-5,
) -> tuple[Any, List[MDLMStepMeasurement], dict[str, object]]:
    """Run DDPM with compact distribution-equivalent candidate state."""
    import torch

    noise_epsilon = 1e-3
    timesteps = torch.linspace(1.0, sampling_epsilon, steps + 1, device=device)
    measurements: List[MDLMStepMeasurement] = []
    candidate_cache = None
    cache_hits = 0
    transition_active_tokens: List[int] = []
    transition_changed_tokens: List[int] = []

    for transition_id in range(steps):
        active_mask = input_ids.eq(mask_token_id)
        active_count = int(active_mask.sum().item())
        if active_count == 0:
            break
        transition_active_tokens.append(active_count)
        time_t = timesteps[transition_id].reshape(1)
        time_s = timesteps[transition_id + 1].reshape(1)
        sigma_t = -torch.log1p(-(1 - noise_epsilon) * time_t)

        evaluated = candidate_cache is None
        elapsed = 0.0
        if evaluated:
            _synchronize(device)
            started = time.perf_counter()
            output = model(input_ids=input_ids, timesteps=sigma_t)
            _synchronize(device)
            elapsed = time.perf_counter() - started
            logits = output.logits if hasattr(output, "logits") else output
            probabilities = _mdlm_log_probabilities(
                logits,
                mask_token_id,
            ).exp()
            candidate_cache = _sample_categorical(probabilities)
        else:
            cache_hits += 1

        move_t = ((1 - noise_epsilon) * time_t).reshape(1, 1)
        move_s = ((1 - noise_epsilon) * time_s).reshape(1, 1)
        reveal_probability = (move_t - move_s) / move_t
        reveal = torch.rand(input_ids.shape, device=device) < reveal_probability
        proposal = torch.where(
            reveal,
            candidate_cache,
            torch.full_like(candidate_cache, mask_token_id),
        )
        next_ids = torch.where(active_mask, proposal, input_ids)
        changed_count = int(input_ids.ne(next_ids).sum().item())
        transition_changed_tokens.append(changed_count)
        if evaluated:
            measurements.append(
                MDLMStepMeasurement(
                    step_id=transition_id,
                    active_tokens=active_count,
                    changed_tokens=changed_count,
                    elapsed_s=elapsed,
                    sigma=float(sigma_t.item()),
                )
            )
        if changed_count:
            candidate_cache = None
        input_ids = next_ids

    active_mask = input_ids.eq(mask_token_id)
    active_count = int(active_mask.sum().item())
    if active_count:
        sigma = -torch.log1p(
            -(1 - noise_epsilon)
            * torch.tensor([sampling_epsilon], device=device)
        )
        _synchronize(device)
        started = time.perf_counter()
        output = model(input_ids=input_ids, timesteps=sigma)
        _synchronize(device)
        elapsed = time.perf_counter() - started
        logits = output.logits if hasattr(output, "logits") else output
        predicted = _mdlm_log_probabilities(logits, mask_token_id).argmax(dim=-1)
        input_ids = torch.where(active_mask, predicted, input_ids)
        measurements.append(
            MDLMStepMeasurement(
                step_id=len(transition_active_tokens),
                active_tokens=active_count,
                changed_tokens=active_count,
                elapsed_s=elapsed,
                sigma=float(sigma.item()),
            )
        )

    positions = input_ids.numel()
    candidate_id_bytes = max(1, math.ceil(mask_token_id.bit_length() / 8))
    metadata: dict[str, object] = {
        "sampling_transitions": len(transition_active_tokens),
        "model_evaluations": len(measurements),
        "probability_cache_hits": cache_hits,
        "candidate_cache_hits": cache_hits,
        "transition_active_tokens": transition_active_tokens,
        "transition_changed_tokens": transition_changed_tokens,
        "cache_hit_sampling_overhead_modeled": False,
        "cache_correctness_class": "distribution-equivalent",
        "hardware_candidate_cache_bytes": (
            positions * candidate_id_bytes + 2 * math.ceil(positions / 8)
        ),
        "candidate_id_bytes": candidate_id_bytes,
    }
    return input_ids, measurements, metadata


def run_mdlm_trace(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: str = "auto",
    canvas_tokens: int = 64,
    steps: int = 64,
    warmup: int = 1,
    seed: int = 0,
    sampler: str = "ddpm",
    local_files_only: bool = False,
) -> WorkloadTrace:
    """Load the official checkpoint, run denoising, and return a real trace."""
    if canvas_tokens <= 0 or steps <= 0 or warmup < 0:
        raise ValueError("canvas_tokens/steps must be positive and warmup non-negative")
    if sampler not in {
        "ddpm",
        "ddpm-cache",
        "ddpm-candidate-cache",
        "confidence-smoke",
    }:
        raise ValueError(
            "sampler must be ddpm, ddpm-cache, ddpm-candidate-cache, "
            "or confidence-smoke"
        )

    import torch

    resolved_device = _resolve_device(device)
    torch.manual_seed(seed)
    model = _load_mdlm_model(
        model_id=model_id,
        revision=revision,
        device=resolved_device,
        local_files_only=local_files_only,
    )

    config = model.config
    if canvas_tokens > int(config.model_length):
        raise ValueError(
            "canvas_tokens exceeds checkpoint context length %d"
            % config.model_length
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_weight_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    spec = MDLMSpec(
        model_id=model_id,
        revision=revision,
        hidden_size=int(config.hidden_dim),
        layers=int(config.n_blocks),
        heads=int(config.n_heads),
        vocab_size=int(config.vocab_size),
        parameter_count=parameter_count,
        model_weight_bytes=model_weight_bytes,
        bytes_per_element=next(model.parameters()).element_size(),
    )

    mask_token_id = int(config.vocab_size) - 1
    input_ids = torch.full(
        (1, canvas_tokens),
        mask_token_id,
        dtype=torch.long,
        device=resolved_device,
    )

    with torch.inference_mode():
        for _ in range(warmup):
            model(input_ids=input_ids, timesteps=torch.ones(1, device=resolved_device))
        _synchronize(resolved_device)

        sampler_metadata: dict[str, object] = {}
        if sampler == "ddpm":
            input_ids, measurements = _run_ddpm_sampler(
                model,
                input_ids,
                mask_token_id=mask_token_id,
                steps=steps,
                device=resolved_device,
            )
            sampler_metadata = {
                "sampling_transitions": len(measurements),
                "model_evaluations": len(measurements),
                "probability_cache_hits": 0,
            }
        elif sampler == "ddpm-cache":
            input_ids, measurements, sampler_metadata = _run_ddpm_cache_sampler(
                model,
                input_ids,
                mask_token_id=mask_token_id,
                steps=steps,
                device=resolved_device,
            )
        elif sampler == "ddpm-candidate-cache":
            input_ids, measurements, sampler_metadata = (
                _run_ddpm_candidate_cache_sampler(
                    model,
                    input_ids,
                    mask_token_id=mask_token_id,
                    steps=steps,
                    device=resolved_device,
                )
            )
        else:
            input_ids, measurements = _run_confidence_smoke_sampler(
                model,
                input_ids,
                mask_token_id=mask_token_id,
                steps=steps,
                device=resolved_device,
            )
            sampler_metadata = {
                "sampling_transitions": len(measurements),
                "model_evaluations": len(measurements),
                "probability_cache_hits": 0,
            }

    trace = lower_mdlm_measurements(
        spec,
        measurements,
        canvas_tokens=canvas_tokens,
        device=resolved_device,
        sampler={
            "ddpm": "official-ddpm",
            "ddpm-cache": "official-ddpm-cache",
            "ddpm-candidate-cache": "distribution-equivalent-candidate-cache",
            "confidence-smoke": "confidence-unmask-smoke",
        }[sampler],
    )
    trace.metadata.update(sampler_metadata)
    decoded_ids = input_ids[0].detach().cpu().tolist()
    trace.metadata["generated_token_ids"] = decoded_ids
    trace.metadata["all_tokens_committed"] = mask_token_id not in decoded_ids
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            local_files_only=local_files_only,
        )
        trace.metadata["generated_text"] = tokenizer.decode(decoded_ids)
    except OSError:
        trace.metadata["generated_text"] = None
    return trace
