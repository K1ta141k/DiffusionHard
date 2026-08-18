"""Frozen model manifest and hardware inventory calculations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    revision: str
    parameter_count: int
    model_length: int
    first_hardware_canvas: int
    vocabulary_size: int
    hidden_size: int
    conditioning_size: int
    transformer_blocks: int
    attention_heads: int
    head_size: int
    mlp_hidden_size: int
    time_conditioning: bool

    def parameter_breakdown(self) -> Dict[str, int]:
        hidden = self.hidden_size
        condition = self.conditioning_size
        mlp = self.mlp_hidden_size
        vocabulary = self.vocabulary_size
        per_block = (
            2 * hidden
            + hidden * 3 * hidden
            + hidden * hidden
            + hidden * mlp
            + mlp
            + mlp * hidden
            + hidden
            + condition * 6 * hidden
            + 6 * hidden
        )
        return {
            "token_embedding": vocabulary * hidden,
            "timestep_embedding": (
                256 * condition
                + condition
                + condition * condition
                + condition
            ),
            "transformer_blocks": per_block * self.transformer_blocks,
            "final_normalization": hidden,
            "vocabulary_projection": hidden * vocabulary + vocabulary,
            "final_adaptive_normalization": (
                condition * 2 * hidden + 2 * hidden
            ),
        }

    def mac_breakdown(
        self, canvas_tokens: Optional[int] = None
    ) -> Dict[str, int]:
        tokens = canvas_tokens or self.first_hardware_canvas
        hidden = self.hidden_size
        condition = self.conditioning_size
        blocks = self.transformer_blocks
        heads = self.attention_heads
        head = self.head_size
        mlp = self.mlp_hidden_size
        return {
            "timestep_embedding": 256 * condition + condition * condition,
            "qkv_projection": blocks * tokens * hidden * 3 * hidden,
            "attention_qk_and_av": (
                blocks * 2 * heads * tokens * tokens * head
            ),
            "attention_output_projection": (
                blocks * tokens * hidden * hidden
            ),
            "mlp_up_and_down": blocks * 2 * tokens * hidden * mlp,
            "block_adaptive_normalization": blocks * condition * 6 * hidden,
            "final_adaptive_normalization": condition * 2 * hidden,
            "vocabulary_projection": tokens * hidden * self.vocabulary_size,
        }

    def hardware_manifest(self) -> Dict[str, object]:
        parameters = self.parameter_breakdown()
        computed_parameters = sum(parameters.values())
        if computed_parameters != self.parameter_count:
            raise ValueError(
                "declared parameter count does not match architecture: "
                f"{self.parameter_count} != {computed_parameters}"
            )
        if self.hidden_size != self.attention_heads * self.head_size:
            raise ValueError("attention heads and head size do not match hidden size")
        tokens = self.first_hardware_canvas
        macs = self.mac_breakdown(tokens)
        return {
            "model": {
                "name": self.name,
                "model_id": self.model_id,
                "revision": self.revision,
                "parameter_count": self.parameter_count,
                "model_length": self.model_length,
                "first_hardware_canvas": tokens,
                "vocabulary_size": self.vocabulary_size,
                "hidden_size": self.hidden_size,
                "conditioning_size": self.conditioning_size,
                "transformer_blocks": self.transformer_blocks,
                "attention_heads": self.attention_heads,
                "head_size": self.head_size,
                "mlp_hidden_size": self.mlp_hidden_size,
                "time_conditioning": self.time_conditioning,
            },
            "parameter_breakdown": parameters,
            "weight_storage_bytes": {
                "fp32": self.parameter_count * 4,
                "fp16": self.parameter_count * 2,
                "int8": self.parameter_count,
            },
            "fp16_working_sets": {
                "activation_canvas": tokens * self.hidden_size * 2,
                "activation_ping_pong": tokens * self.hidden_size * 2 * 2,
                "qkv": tokens * 3 * self.hidden_size * 2,
                "mlp_intermediate": tokens * self.mlp_hidden_size * 2,
                "attention_scores": (
                    self.attention_heads * tokens * tokens * 2
                ),
            },
            "forward_macs": {
                "breakdown": macs,
                "total": sum(macs.values()),
                "two_ops_per_mac_flops": 2 * sum(macs.values()),
            },
            "scope": (
                "architecture-derived inventory; excludes vector-operation "
                "FLOPs and assumes one 64-token, batch-one forward"
            ),
        }


def load_model_spec(path: Path) -> ModelSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelSpec(
        name=str(data["name"]),
        model_id=str(data["model_id"]),
        revision=str(data["revision"]),
        parameter_count=int(data["parameter_count"]),
        model_length=int(data["model_length"]),
        first_hardware_canvas=int(data["first_hardware_canvas"]),
        vocabulary_size=int(data["vocabulary_size"]),
        hidden_size=int(data["hidden_size"]),
        conditioning_size=int(data["conditioning_size"]),
        transformer_blocks=int(data["transformer_blocks"]),
        attention_heads=int(data["attention_heads"]),
        head_size=int(data["head_size"]),
        mlp_hidden_size=int(data["mlp_hidden_size"]),
        time_conditioning=bool(data["time_conditioning"]),
    )
