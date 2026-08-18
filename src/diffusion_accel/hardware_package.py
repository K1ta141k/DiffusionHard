"""Export the pinned MDLM checkpoint as a fixed-shape hardware package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .mdlm import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    _load_mdlm_model,
    _resolve_device,
    _synchronize,
)
from .model_specialization import (
    fold_normalized_input_affine,
    fold_output_gate,
)


PACKAGE_SCHEMA_VERSION = "0.1"


def deterministic_hardware_input(
    canvas_tokens: int, vocabulary_size: int, mask_token_id: int
) -> List[int]:
    """Create a stable mixed masked and visible 64-token hardware input."""
    if canvas_tokens <= 0:
        raise ValueError("canvas_tokens must be positive")
    if vocabulary_size <= 1:
        raise ValueError("vocabulary_size must be greater than one")
    if mask_token_id != vocabulary_size - 1:
        raise ValueError("MDLM hardware package expects the final ID as mask")
    result = []
    for position in range(canvas_tokens):
        visible_id = (position * 7919 + 17) % mask_token_id
        result.append(mask_token_id if position % 3 else visible_id)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_manifest(tensors: Dict[str, Any]) -> Dict[str, object]:
    entries: Dict[str, object] = {}
    total_bytes = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        size_bytes = tensor.numel() * tensor.element_size()
        total_bytes += size_bytes
        entries[name] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "bytes": size_bytes,
        }
    return {
        "tensor_count": len(entries),
        "tensor_bytes": total_bytes,
        "tensors": entries,
    }


def _hardware_weight_order(names: Iterable[str]) -> List[str]:
    def key(name: str) -> Tuple[int, str]:
        if name == "embedding.weight":
            return (0, name)
        if name.startswith("block_"):
            return (1, name)
        if name.startswith("output."):
            return (2, name)
        return (3, name)

    return sorted(names, key=key)


def _write_aligned_weight_binary(
    tensors: Dict[str, Any], path: Path, alignment: int = 4096
) -> Dict[str, object]:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("binary alignment must be a positive power of two")
    entries: Dict[str, object] = {}
    offset = 0
    with path.open("wb") as stream:
        for name in _hardware_weight_order(tensors):
            tensor = tensors[name]
            padding = (-offset) % alignment
            if padding:
                stream.write(bytes(padding))
                offset += padding
            payload = tensor.numpy().tobytes(order="C")
            entries[name] = {
                "offset_bytes": offset,
                "length_bytes": len(payload),
                "dtype": "little-endian-float16",
                "shape": list(tensor.shape),
                "order": "row-major",
            }
            stream.write(payload)
            offset += len(payload)
        final_padding = (-offset) % alignment
        if final_padding:
            stream.write(bytes(final_padding))
            offset += final_padding
    return {
        "alignment_bytes": alignment,
        "total_bytes": offset,
        "tensor_count": len(entries),
        "tensors": entries,
    }


def _cpu_contiguous(tensor: Any, dtype: Optional[Any] = None) -> Any:
    detached = tensor.detach()
    if dtype is not None:
        detached = detached.to(dtype=dtype)
    return detached.cpu().contiguous()


def _constant_condition(model: Any, device: str) -> Any:
    import torch
    import torch.nn.functional as functional

    zeros = torch.zeros(1, device=device)
    return functional.silu(model.backbone.sigma_map(zeros))


def _block_constants(block: Any, condition: Any) -> Tuple[Any, ...]:
    modulation = block.adaLN_modulation(condition)[0]
    return tuple(modulation.chunk(6, dim=0))


def _fold_block(block: Any, condition: Any) -> Dict[str, Any]:
    (
        shift_msa,
        scale_msa,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
    ) = _block_constants(block, condition)
    qkv_weight, qkv_bias = fold_normalized_input_affine(
        block.attn_qkv.weight,
        block.attn_qkv.bias,
        block.norm1.weight,
        shift_msa,
        scale_msa,
    )
    attention_weight, attention_bias = fold_output_gate(
        block.attn_out.weight, block.attn_out.bias, gate_msa
    )
    mlp_up_weight, mlp_up_bias = fold_normalized_input_affine(
        block.mlp[0].weight,
        block.mlp[0].bias,
        block.norm2.weight,
        shift_mlp,
        scale_mlp,
    )
    mlp_down_weight, mlp_down_bias = fold_output_gate(
        block.mlp[2].weight, block.mlp[2].bias, gate_mlp
    )
    if attention_bias is not None:
        raise ValueError("pinned MDLM attention output must not have a bias")
    if mlp_down_bias is None:
        raise ValueError("pinned MDLM MLP down projection must have a bias")
    return {
        "qkv.weight": qkv_weight,
        "qkv.bias": qkv_bias,
        "attention_out.weight": attention_weight,
        "mlp_up.weight": mlp_up_weight,
        "mlp_up.bias": mlp_up_bias,
        "mlp_down.weight": mlp_down_weight,
        "mlp_down.bias": mlp_down_bias,
    }


def _fold_output_layer(layer: Any, condition: Any) -> Dict[str, Any]:
    shift, scale = layer.adaLN_modulation(condition)[0].chunk(2, dim=0)
    weight, bias = fold_normalized_input_affine(
        layer.linear.weight,
        layer.linear.bias,
        layer.norm_final.weight,
        shift,
        scale,
    )
    return {"weight": weight, "bias": bias}


def folded_mdlm_forward(
    model: Any,
    input_ids: Any,
    device: str,
    captured: Optional[Dict[str, Any]] = None,
) -> Any:
    """Run the constant-folded fixed-shape graph for equivalence validation."""
    import torch
    import torch.nn.functional as functional

    backbone = model.backbone
    condition = _constant_condition(model, device)
    x = backbone.vocab_embed(input_ids)
    if captured is not None:
        captured["folded.embedding"] = _cpu_contiguous(x)
    rotary_cos, rotary_sin = backbone.rotary_emb(x)
    heads = int(model.config.n_heads)
    head_size = int(model.config.hidden_dim) // heads

    for index, block in enumerate(backbone.blocks):
        prefix = "folded.block_%02d" % index
        folded = _fold_block(block, condition)
        normalized = functional.layer_norm(x.float(), [int(model.config.hidden_dim)])
        if captured is not None:
            captured[prefix + ".norm1_unaffine"] = _cpu_contiguous(normalized)
        qkv = functional.linear(
            normalized, folded["qkv.weight"], folded["qkv.bias"]
        )
        if captured is not None:
            captured[prefix + ".qkv"] = _cpu_contiguous(qkv)
        qkv = qkv.view(qkv.shape[0], qkv.shape[1], 3, heads, head_size)

        half = head_size // 2
        cos = rotary_cos[0, :, 0, 0, :half][None, :, None, :]
        sin = rotary_sin[0, :, 0, 0, :half][None, :, None, :]
        query, key, value = qkv.unbind(dim=2)

        def rotate(tensor: Any) -> Any:
            first = tensor[..., :half]
            second = tensor[..., half:]
            return torch.cat(
                (first * cos - second * sin, second * cos + first * sin),
                dim=-1,
            )

        query = rotate(query)
        key = rotate(key)
        attention = functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        attention = attention.reshape(attention.shape[0], attention.shape[1], -1)
        if captured is not None:
            captured[prefix + ".attention"] = _cpu_contiguous(attention)
        attention_projection = functional.linear(
            attention, folded["attention_out.weight"], None
        )
        x = x + attention_projection
        if captured is not None:
            captured[prefix + ".attention_projection"] = _cpu_contiguous(
                attention_projection
            )
            captured[prefix + ".after_attention"] = _cpu_contiguous(x)

        normalized = functional.layer_norm(x.float(), [int(model.config.hidden_dim)])
        if captured is not None:
            captured[prefix + ".norm2_unaffine"] = _cpu_contiguous(normalized)
        mlp = functional.linear(
            normalized, folded["mlp_up.weight"], folded["mlp_up.bias"]
        )
        if captured is not None:
            captured[prefix + ".mlp_up"] = _cpu_contiguous(mlp)
        mlp = functional.gelu(mlp, approximate="tanh")
        if captured is not None:
            captured[prefix + ".gelu"] = _cpu_contiguous(mlp)
        mlp = functional.linear(
            mlp, folded["mlp_down.weight"], folded["mlp_down.bias"]
        )
        if captured is not None:
            captured[prefix + ".mlp_down"] = _cpu_contiguous(mlp)
        x = x + mlp
        if captured is not None:
            captured[prefix + ".output"] = _cpu_contiguous(x)

    output = _fold_output_layer(backbone.output_layer, condition)
    normalized = functional.layer_norm(x.float(), [int(model.config.hidden_dim)])
    if captured is not None:
        captured["folded.final.norm_unaffine"] = _cpu_contiguous(normalized)
    logits = functional.linear(normalized, output["weight"], output["bias"])
    if captured is not None:
        captured["folded.final.logits"] = _cpu_contiguous(logits)
    return logits


def _capture_reference_tensors(model: Any) -> Tuple[Dict[str, Any], List[Any]]:
    captured: Dict[str, Any] = {}
    handles: List[Any] = []

    def register(name: str, module: Any) -> None:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not hasattr(output, "detach"):
                raise TypeError("hardware golden hook expects one tensor output")
            captured[name] = _cpu_contiguous(output)

        handles.append(module.register_forward_hook(hook))

    register("embedding", model.backbone.vocab_embed)
    register("condition.timestep_mlp", model.backbone.sigma_map)
    for index, block in enumerate(model.backbone.blocks):
        prefix = "block_%02d" % index
        register(prefix + ".norm1", block.norm1)
        register(prefix + ".adaln", block.adaLN_modulation)
        register(prefix + ".qkv", block.attn_qkv)
        register(prefix + ".attention_out", block.attn_out)
        register(prefix + ".norm2", block.norm2)
        register(prefix + ".mlp_up", block.mlp[0])
        register(prefix + ".gelu", block.mlp[1])
        register(prefix + ".mlp_down", block.mlp[2])
        register(prefix + ".output", block)
    register("final.norm", model.backbone.output_layer.norm_final)
    register("final.adaln", model.backbone.output_layer.adaLN_modulation)
    register("final.logits", model.backbone.output_layer.linear)
    return captured, handles


def _export_folded_weights(model: Any, condition: Any) -> Dict[str, Any]:
    import torch

    weights: Dict[str, Any] = {
        "embedding.weight": _cpu_contiguous(
            model.backbone.vocab_embed.embedding, torch.float16
        )
    }
    for index, block in enumerate(model.backbone.blocks):
        for suffix, tensor in _fold_block(block, condition).items():
            weights["block_%02d.%s" % (index, suffix)] = _cpu_contiguous(
                tensor, torch.float16
            )
    output = _fold_output_layer(model.backbone.output_layer, condition)
    weights["output.weight"] = _cpu_contiguous(output["weight"], torch.float16)
    weights["output.bias"] = _cpu_contiguous(output["bias"], torch.float16)

    dummy = torch.empty(
        1,
        int(model.config.model_length),
        int(model.config.hidden_dim),
        device=condition.device,
    )
    cos, sin = model.backbone.rotary_emb(dummy)
    half = int(model.config.hidden_dim) // int(model.config.n_heads) // 2
    weights["rotary.cos"] = _cpu_contiguous(
        cos[0, :64, 0, 0, :half], torch.float16
    )
    weights["rotary.sin"] = _cpu_contiguous(
        sin[0, :64, 0, 0, :half], torch.float16
    )
    return weights


def _checkpoint_path(model: Any) -> Optional[Path]:
    del model
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    snapshot = (
        hub
        / "models--kuleshov-group--mdlm-owt"
        / "snapshots"
        / DEFAULT_REVISION
        / "model.safetensors"
    )
    return snapshot if snapshot.exists() else None


def export_mdlm_hardware_package(
    out_dir: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    device: str = "auto",
    canvas_tokens: int = 64,
    local_files_only: bool = False,
) -> Dict[str, object]:
    """Export real goldens and FP16 constant-folded weights for hardware."""
    import torch
    from safetensors.torch import save_file

    if canvas_tokens != 64:
        raise ValueError("the first hardware package is frozen to 64 tokens")
    resolved_device = _resolve_device(device)
    model = _load_mdlm_model(
        model_id=model_id,
        revision=revision,
        device=resolved_device,
        local_files_only=local_files_only,
    )
    if bool(model.config.time_conditioning):
        raise ValueError("constant-folded package requires disabled time conditioning")
    mask_token_id = int(model.config.vocab_size) - 1
    input_values = deterministic_hardware_input(
        canvas_tokens, int(model.config.vocab_size), mask_token_id
    )
    input_ids = torch.tensor(
        [input_values], dtype=torch.long, device=resolved_device
    )
    timestep = torch.tensor([0.73125], device=resolved_device)
    zero_timestep = torch.zeros_like(timestep)

    captured, handles = _capture_reference_tensors(model)
    with torch.no_grad():
        reference_logits = model(input_ids=input_ids, timesteps=timestep)
        _synchronize(resolved_device)
    for handle in handles:
        handle.remove()

    folded_captured: Dict[str, Any] = {}
    with torch.no_grad():
        zero_logits = model(input_ids=input_ids, timesteps=zero_timestep)
        folded_logits = folded_mdlm_forward(
            model, input_ids, resolved_device, folded_captured
        )
        _synchronize(resolved_device)

    time_difference = (reference_logits - zero_logits).abs()
    fold_difference = (reference_logits - folded_logits).abs()
    reference_scores = reference_logits.clone()
    folded_scores = folded_logits.clone()
    reference_scores[..., mask_token_id] = -torch.inf
    folded_scores[..., mask_token_id] = -torch.inf
    reference_top1 = reference_scores.argmax(dim=-1)
    folded_top1 = folded_scores.argmax(dim=-1)
    top1_agreement = float(
        reference_top1.eq(folded_top1).float().mean().item()
    )
    fold_max_abs_error = float(fold_difference.max().item())
    time_max_abs_error = float(time_difference.max().item())

    goldens: Dict[str, Any] = dict(captured)
    goldens.update(folded_captured)
    goldens["input.ids"] = _cpu_contiguous(input_ids, torch.int32)
    goldens["input.timestep"] = _cpu_contiguous(timestep, torch.float32)
    goldens["reference.top1"] = _cpu_contiguous(reference_top1, torch.int32)
    goldens["folded.top1"] = _cpu_contiguous(folded_top1, torch.int32)

    condition = _constant_condition(model, resolved_device)
    weights = _export_folded_weights(model, condition)
    out_dir.mkdir(parents=True, exist_ok=True)
    golden_path = out_dir / "golden_tensors.safetensors"
    weight_path = out_dir / "folded_fp16_weights.safetensors"
    binary_path = out_dir / "folded_fp16_weights.bin"
    save_file(goldens, golden_path)
    save_file(weights, weight_path)
    binary_layout = _write_aligned_weight_binary(weights, binary_path)

    checkpoint = _checkpoint_path(model)
    file_records: Dict[str, object] = {
        golden_path.name: {
            "bytes": golden_path.stat().st_size,
            "sha256": _sha256(golden_path),
        },
        weight_path.name: {
            "bytes": weight_path.stat().st_size,
            "sha256": _sha256(weight_path),
        },
        binary_path.name: {
            "bytes": binary_path.stat().st_size,
            "sha256": _sha256(binary_path),
        },
    }
    source_checkpoint: Dict[str, object] = {
        "model_id": model_id,
        "revision": revision,
    }
    if checkpoint is not None:
        source_checkpoint.update(
            {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
            }
        )

    decoded_reference_top1: Optional[str] = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2", local_files_only=local_files_only
        )
        decoded_reference_top1 = tokenizer.decode(
            reference_top1[0].detach().cpu().tolist()
        )
    except OSError:
        pass

    manifest: Dict[str, object] = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "source_checkpoint": source_checkpoint,
        "fixed_graph": {
            "canvas_tokens": canvas_tokens,
            "batch_size": 1,
            "hidden_size": int(model.config.hidden_dim),
            "blocks": int(model.config.n_blocks),
            "heads": int(model.config.n_heads),
            "vocabulary_size": int(model.config.vocab_size),
            "mask_token_id": mask_token_id,
            "time_conditioning": bool(model.config.time_conditioning),
        },
        "input": {
            "construction": "position*7919+17 modulo mask ID; positions not divisible by three are masked",
            "visible_tokens": sum(value != mask_token_id for value in input_values),
            "masked_tokens": sum(value == mask_token_id for value in input_values),
            "requested_timestep": float(timestep.item()),
            "reference_top1_token_ids": reference_top1[0]
            .detach()
            .cpu()
            .tolist(),
            "reference_top1_decoded_text": decoded_reference_top1,
        },
        "constant_fold_validation": {
            "requested_vs_zero_timestep_max_abs_error": time_max_abs_error,
            "folded_vs_reference_max_abs_error": fold_max_abs_error,
            "folded_vs_reference_mean_abs_error": float(
                fold_difference.mean().item()
            ),
            "top1_agreement": top1_agreement,
            "acceptance_max_abs_error": 1e-4,
            "passed": (
                time_max_abs_error == 0.0
                and fold_max_abs_error <= 1e-4
                and top1_agreement == 1.0
            ),
        },
        "goldens": _tensor_manifest(goldens),
        "weights": _tensor_manifest(weights),
        "weight_binary_layout": binary_layout,
        "files": file_records,
        "device_used_for_reference": resolved_device,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _validate_binary_layout(manifest: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    binary = manifest["weight_binary_layout"]
    alignment = int(binary["alignment_bytes"])
    total_bytes = int(binary["total_bytes"])
    previous_end = 0
    weight_tensors = manifest["weights"]["tensors"]
    for name in _hardware_weight_order(binary["tensors"]):
        entry = binary["tensors"][name]
        offset = int(entry["offset_bytes"])
        length = int(entry["length_bytes"])
        if offset % alignment:
            errors.append("%s offset is not aligned" % name)
        if offset < previous_end:
            errors.append("%s overlaps the previous tensor" % name)
        if length != int(weight_tensors[name]["bytes"]):
            errors.append("%s binary length does not match tensor bytes" % name)
        if offset + length > total_bytes:
            errors.append("%s exceeds the declared binary size" % name)
        previous_end = offset + length
    return errors


def validate_mdlm_hardware_package(package_dir: Path) -> Dict[str, object]:
    """Independently verify package hashes, tensor metadata, and DDR layout."""
    from safetensors import safe_open

    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: List[str] = []
    verified_files: Dict[str, object] = {}
    for name, expected in manifest["files"].items():
        path = package_dir / name
        if not path.exists():
            errors.append("missing file %s" % name)
            continue
        actual_bytes = path.stat().st_size
        actual_sha256 = _sha256(path)
        if actual_bytes != int(expected["bytes"]):
            errors.append("%s byte size mismatch" % name)
        if actual_sha256 != str(expected["sha256"]):
            errors.append("%s SHA-256 mismatch" % name)
        verified_files[name] = {
            "bytes": actual_bytes,
            "sha256": actual_sha256,
        }

    for filename, section in (
        ("golden_tensors.safetensors", "goldens"),
        ("folded_fp16_weights.safetensors", "weights"),
    ):
        with safe_open(package_dir / filename, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            expected_tensors = manifest[section]["tensors"]
            if set(keys) != set(expected_tensors):
                errors.append("%s tensor names do not match manifest" % filename)
                continue
            for name in keys:
                shape = list(handle.get_slice(name).get_shape())
                if shape != expected_tensors[name]["shape"]:
                    errors.append("%s shape mismatch for %s" % (filename, name))

    errors.extend(_validate_binary_layout(manifest))
    if not bool(manifest["constant_fold_validation"]["passed"]):
        errors.append("constant-fold validation did not pass")
    return {
        "schema_version": manifest["schema_version"],
        "package_dir": str(package_dir),
        "passed": not errors,
        "errors": errors,
        "verified_files": verified_files,
        "golden_tensor_count": int(manifest["goldens"]["tensor_count"]),
        "weight_tensor_count": int(manifest["weights"]["tensor_count"]),
        "binary_alignment_bytes": int(
            manifest["weight_binary_layout"]["alignment_bytes"]
        ),
    }
