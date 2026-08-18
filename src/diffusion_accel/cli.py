"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .config import load_hardware
from .candidate_cache import (
    analyze_candidate_reveal_kernel,
    validate_candidate_cache_equivalence,
)
from .candidate_producer import (
    analyze_output_head_lane_sweep,
    validate_streaming_candidate_producer,
)
from .block_image import (
    export_block_execution_image,
    validate_block_execution_image,
)
from .conditioned import evaluate_conditioned_mdlm_ddpm
from .hardware_package import (
    export_mdlm_hardware_package,
    validate_mdlm_hardware_package,
)
from .fixed_mlp import (
    export_mlp_interstage_package,
    optimize_fixed_mlp_alphas,
    sweep_hardware_fixed_mlp,
    sweep_fixed_mlp,
    validate_fixed_mlp,
    validate_hardware_fixed_mlp,
)
from .fixed_norm import sweep_fixed_layer_norm
from .attention_int8 import (
    screen_packed_int8_attention,
    screen_packed_int8_attention_heldout,
    screen_packed_int8_attention_logits,
)
from .apple_mdlm import benchmark_mdlm_apple
from .fixed_attention import (
    sweep_fixed_attention,
    sweep_fixed_rotary,
    sweep_qkv_weight_precision,
)
from .mdlm import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    run_mdlm_trace,
    validate_mdlm_int8,
    validate_mdlm_output_head_int8_generation,
)
from .model_spec import load_model_spec
from .mlp_pipeline import analyze_mlp_pingpong
from .multiturn import evaluate_two_turn_mdlm_session
from .output_head import validate_tiled_output_head
from .optimizations import (
    fused_streaming_candidate_head,
    masked_output_head,
    quantize_weights,
)
from .prefix_drift import DEFAULT_PREFIX_TEXT, analyze_mdlm_prefix_drift
from .quality import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_REVISION,
    evaluate_mdlm_prefix_isolation_quality,
)
from .qkv_int8 import screen_qkv_int8_logits
from .realtime_target import analyze_realtime_target
from .rng import validate_rng_and_gumbel
from .rng_hardware import analyze_rng_hardware
from .session_cache import (
    ConversationTurn,
    SessionCacheConfig,
    analyze_session_cache,
)
from .session_lifecycle import (
    SessionRequest,
    analyze_and_trace_session_lifecycle,
)
from .rtl_synthesis import synthesize_candidate_reveal, synthesize_rtl
from .simulator import AllHBMPolicy, CanvasSRAMPolicy, MemoryPolicy, simulate
from .trace import read_trace, synthetic_trace, write_trace


def _policy(name: str) -> MemoryPolicy:
    if name == "all-hbm":
        return AllHBMPolicy()
    if name == "canvas-sram":
        return CanvasSRAMPolicy()
    raise ValueError("unknown policy %s" % name)


def _conversation_turn(value: str) -> ConversationTurn:
    try:
        user_tokens, answer_tokens = value.split(":", maxsplit=1)
        return ConversationTurn(int(user_tokens), int(answer_tokens))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "turn must be USER_TOKENS:ANSWER_TOKENS, for example 128:256"
        ) from error


def _session_request(value: str) -> SessionRequest:
    try:
        session_id, arrival_s, user_tokens, answer_tokens, evaluations = (
            value.split(":", maxsplit=4)
        )
        return SessionRequest(
            session_id=session_id,
            arrival_s=float(arrival_s),
            user_tokens=int(user_tokens),
            answer_tokens=int(answer_tokens),
            model_evaluations=int(evaluations),
        )
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "request must be SESSION:ARRIVAL_S:USER_TOKENS:ANSWER_TOKENS:EVALS"
        ) from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diffusion-accel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate-synthetic", help="write a synthetic diffusion trace"
    )
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--steps", type=int, default=10)
    generate.add_argument("--layers", type=int, default=12)
    generate.add_argument("--canvas-tokens", type=int, default=128)
    generate.add_argument("--hidden-size", type=int, default=768)
    generate.add_argument("--vocab-size", type=int, default=32768)

    inspect_model = subparsers.add_parser(
        "inspect-model-spec",
        help="validate a frozen model architecture and emit its hardware inventory",
    )
    inspect_model.add_argument("--model", type=Path, required=True)
    inspect_model.add_argument("--out", type=Path)
    generate.add_argument("--model-weight-bytes", type=int, default=250_000_000)

    realtime = subparsers.add_parser(
        "analyze-realtime-target",
        help="evaluate model-specific FPGA design points against a latency contract",
    )
    realtime.add_argument("--target", type=Path, required=True)
    realtime.add_argument("--project-root", type=Path, default=Path.cwd())
    realtime.add_argument("--out", type=Path)

    export_hardware = subparsers.add_parser(
        "export-mdlm-hardware-package",
        help="export real 64-token goldens and constant-folded FP16 weights",
    )
    export_hardware.add_argument("--out-dir", type=Path, required=True)
    export_hardware.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    export_hardware.add_argument("--revision", default=DEFAULT_REVISION)
    export_hardware.add_argument("--device", default="auto")
    export_hardware.add_argument("--local-files-only", action="store_true")

    validate_hardware = subparsers.add_parser(
        "validate-mdlm-hardware-package",
        help="verify H0 hashes, safetensors metadata, and aligned DDR layout",
    )
    validate_hardware.add_argument("--package-dir", type=Path, required=True)

    fixed_mlp = subparsers.add_parser(
        "validate-fixed-mlp",
        help="validate one real folded MDLM MLP with integer arithmetic",
    )
    fixed_mlp.add_argument("--package-dir", type=Path, required=True)
    fixed_mlp.add_argument("--block", type=int, default=0)
    fixed_mlp.add_argument("--activation-bits", type=int, choices=[8, 16], default=8)
    fixed_mlp.add_argument("--weight-bits", type=int, choices=[8, 16], default=8)
    fixed_mlp.add_argument(
        "--activation-granularity", choices=["tensor", "token"], default="token"
    )
    fixed_mlp.add_argument("--smoothquant-alpha", type=float)
    fixed_mlp.add_argument("--mac-lanes", type=int, default=1024)
    fixed_mlp.add_argument("--clock-mhz", type=float, default=250.0)
    fixed_mlp.add_argument("--out", type=Path)

    hardware_fixed_mlp = subparsers.add_parser(
        "validate-hardware-fixed-mlp",
        help="validate real MDLM MLP through RTL-equivalent requant and GELU",
    )
    hardware_fixed_mlp.add_argument("--package-dir", type=Path, required=True)
    hardware_fixed_mlp.add_argument("--block", type=int, default=0)
    hardware_fixed_mlp.add_argument(
        "--activation-granularity", choices=["tensor", "token"], default="token"
    )
    hardware_fixed_mlp.add_argument(
        "--down-activation-granularity", choices=["tensor", "token"]
    )
    hardware_fixed_mlp.add_argument("--smoothquant-alpha", type=float, default=0.75)
    hardware_fixed_mlp.add_argument("--out", type=Path)

    hardware_fixed_mlp_sweep = subparsers.add_parser(
        "sweep-hardware-fixed-mlp",
        help="run RTL-equivalent fixed MLP validation across all 12 blocks",
    )
    hardware_fixed_mlp_sweep.add_argument(
        "--package-dir", type=Path, required=True
    )
    hardware_fixed_mlp_sweep.add_argument(
        "--activation-granularity", choices=["tensor", "token"], default="token"
    )
    hardware_fixed_mlp_sweep.add_argument(
        "--down-activation-granularity", choices=["tensor", "token"]
    )
    hardware_fixed_mlp_sweep.add_argument("--out", type=Path)

    interstage_export = subparsers.add_parser(
        "export-mlp-interstage",
        help="freeze Q5.10-to-INT8 SmoothQuant constants for every MLP block",
    )
    interstage_export.add_argument("--package-dir", type=Path, required=True)
    interstage_export.add_argument("--out", type=Path, required=True)
    interstage_export.add_argument("--manifest", type=Path)

    block_image_export = subparsers.add_parser(
        "export-block-execution-image",
        help="pack one optimized DDiT block into execution-order DDR records",
    )
    block_image_export.add_argument("--package-dir", type=Path, required=True)
    block_image_export.add_argument("--out", type=Path, required=True)
    block_image_export.add_argument("--block", type=int, default=0)
    block_image_export.add_argument("--manifest", type=Path)

    block_image_validate = subparsers.add_parser(
        "validate-block-execution-image",
        help="verify an execution-order block image, layout, padding, and hashes",
    )
    block_image_validate.add_argument("--manifest", type=Path, required=True)

    fixed_layer_norm = subparsers.add_parser(
        "sweep-fixed-layernorm",
        help="validate Q13.10 to Q5.12 LayerNorm at all H0 boundaries",
    )
    fixed_layer_norm.add_argument("--package-dir", type=Path, required=True)
    fixed_layer_norm.add_argument("--out", type=Path)

    fixed_rotary = subparsers.add_parser(
        "sweep-fixed-rotary",
        help="validate Q5.12 rotary Q/K transforms at all H0 blocks",
    )
    fixed_rotary.add_argument("--package-dir", type=Path, required=True)
    fixed_rotary.add_argument("--out", type=Path)

    fixed_attention = subparsers.add_parser(
        "sweep-fixed-attention",
        help="validate fixed attention, output projection, and residual",
    )
    fixed_attention.add_argument("--package-dir", type=Path, required=True)
    fixed_attention.add_argument("--out", type=Path)

    qkv_precision = subparsers.add_parser(
        "sweep-qkv-weight-precision",
        help="compare INT8 and INT16 QKV weights at all H0 boundaries",
    )
    qkv_precision.add_argument("--package-dir", type=Path, required=True)
    qkv_precision.add_argument(
        "--weight-bits", type=int, nargs="+", choices=[8, 16], default=[8, 16]
    )
    qkv_precision.add_argument(
        "--smoothquant-alphas", type=float, nargs="+", default=[]
    )
    qkv_precision.add_argument("--out", type=Path)

    qkv_logits = subparsers.add_parser(
        "screen-qkv-int8-logits",
        help="propagate calibrated SmoothQuant INT8 QKV through all blocks",
    )
    qkv_logits.add_argument("--package-dir", type=Path, required=True)
    qkv_logits.add_argument("--seeds", type=int, nargs="+")
    qkv_logits.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    qkv_logits.add_argument("--smoothquant-alpha", type=float, default=0.25)
    qkv_logits.add_argument("--out", type=Path)

    int8_attention = subparsers.add_parser(
        "screen-packed-int8-attention",
        help="screen paired-token INT8 QK and PV on all captured H0 blocks",
    )
    int8_attention.add_argument("--package-dir", type=Path, required=True)
    int8_attention.add_argument("--out", type=Path)

    int8_attention_heldout = subparsers.add_parser(
        "screen-packed-int8-attention-heldout",
        help="apply captured INT8 attention calibration to fresh canvases",
    )
    int8_attention_heldout.add_argument(
        "--package-dir", type=Path, required=True
    )
    int8_attention_heldout.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    int8_attention_heldout.add_argument("--seeds", type=int, nargs="+")
    int8_attention_heldout.add_argument("--out", type=Path)

    int8_attention_logits = subparsers.add_parser(
        "screen-packed-int8-attention-logits",
        help="propagate packed INT8 attention error through all twelve blocks",
    )
    int8_attention_logits.add_argument(
        "--package-dir", type=Path, required=True
    )
    int8_attention_logits.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    int8_attention_logits.add_argument("--seeds", type=int, nargs="+")
    int8_attention_logits.add_argument("--preserve-qk18", action="store_true")
    int8_attention_logits.add_argument("--preserve-pv18", action="store_true")
    int8_attention_logits.add_argument(
        "--qk-bits", type=int, choices=[8, 9], default=8
    )
    int8_attention_logits.add_argument(
        "--qk-scale-mode",
        choices=["calibrated-head", "dynamic-vector"],
        default="calibrated-head",
    )
    int8_attention_logits.add_argument(
        "--qk-multiplier-fraction-bits",
        type=int,
        choices=[17, 24],
        default=24,
    )
    int8_attention_logits.add_argument(
        "--pv-probability-levels", type=int, choices=[127, 255], default=127
    )
    int8_attention_logits.add_argument(
        "--pv-probability-sum-correction", action="store_true"
    )
    int8_attention_logits.add_argument(
        "--pv-value-bits", type=int, choices=[8, 9], default=8
    )
    int8_attention_logits.add_argument("--out", type=Path)

    mlp_pingpong = subparsers.add_parser(
        "analyze-mlp-pingpong",
        help="model fixed MDLM MLP weight-load overlap and buffer sizes",
    )
    mlp_pingpong.add_argument("--clock-mhz", type=float, default=250.0)
    mlp_pingpong.add_argument("--effective-ddr-gbps", type=float, default=12.48)
    mlp_pingpong.add_argument("--evaluations", type=int, default=8)
    mlp_pingpong.add_argument("--out", type=Path)

    fixed_mlp_sweep = subparsers.add_parser(
        "sweep-fixed-mlp",
        help="sweep frozen W8A8, mixed, and W16A16 block design points",
    )
    fixed_mlp_sweep.add_argument("--package-dir", type=Path, required=True)
    fixed_mlp_sweep.add_argument("--block", type=int, default=0)
    fixed_mlp_sweep.add_argument("--mac-lanes", type=int, default=1024)
    fixed_mlp_sweep.add_argument("--clock-mhz", type=float, default=250.0)
    fixed_mlp_sweep.add_argument("--out", type=Path)

    optimize_mlp_alphas = subparsers.add_parser(
        "optimize-fixed-mlp-alphas",
        help="select a baked W8A8 SmoothQuant alpha for each of 12 blocks",
    )
    optimize_mlp_alphas.add_argument("--package-dir", type=Path, required=True)
    optimize_mlp_alphas.add_argument(
        "--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    optimize_mlp_alphas.add_argument("--mac-lanes", type=int, default=1024)
    optimize_mlp_alphas.add_argument("--clock-mhz", type=float, default=250.0)
    optimize_mlp_alphas.add_argument("--out", type=Path)

    mdlm = subparsers.add_parser(
        "trace-mdlm", help="run the official MDLM-OWT checkpoint and write a trace"
    )
    mdlm.add_argument("--out", type=Path, required=True)
    mdlm.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    mdlm.add_argument("--revision", default=DEFAULT_REVISION)
    mdlm.add_argument("--device", default="auto")
    mdlm.add_argument("--canvas-tokens", type=int, default=64)
    mdlm.add_argument("--steps", type=int, default=64)
    mdlm.add_argument("--warmup", type=int, default=1)
    mdlm.add_argument("--seed", type=int, default=0)
    mdlm.add_argument(
        "--sampler",
        choices=[
            "ddpm",
            "ddpm-cache",
            "ddpm-candidate-cache",
            "confidence-smoke",
        ],
        default="ddpm-cache",
    )
    mdlm.add_argument("--local-files-only", action="store_true")

    apple_benchmark = subparsers.add_parser(
        "benchmark-mdlm-apple",
        help="benchmark the Apple Silicon event-driven MDLM sampler",
    )
    apple_benchmark.add_argument("--out", type=Path)
    apple_benchmark.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    apple_benchmark.add_argument("--revision", default=DEFAULT_REVISION)
    apple_benchmark.add_argument("--device", default="auto")
    apple_benchmark.add_argument("--canvas-tokens", type=int, default=64)
    apple_benchmark.add_argument("--steps", type=int, default=64)
    apple_benchmark.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    apple_benchmark.add_argument("--local-files-only", action="store_true")

    mlx_benchmark = subparsers.add_parser(
        "benchmark-mdlm-mlx",
        help="benchmark the MLX-native event-driven MDLM sampler",
    )
    mlx_benchmark.add_argument("--out", type=Path)
    mlx_benchmark.add_argument("--snapshot", type=Path)
    mlx_benchmark.add_argument("--golden-tensors", type=Path, required=True)
    mlx_benchmark.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    mlx_benchmark.add_argument("--canvas-tokens", type=int, default=64)
    mlx_benchmark.add_argument("--steps", type=int, default=64)
    mlx_benchmark.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    mlx_benchmark.add_argument(
        "--output-head-bits",
        type=int,
        choices=[4, 6, 8],
    )
    mlx_benchmark.add_argument("--fold-constants", action="store_true")
    mlx_benchmark.add_argument("--compile-sampler", action="store_true")
    mlx_benchmark.add_argument("--mps-validation-seeds", type=int, default=0)

    validate = subparsers.add_parser(
        "validate-mdlm-int8",
        help="compare FP32 and fake-quantized INT8 MDLM outputs",
    )
    validate.add_argument("--out", type=Path)
    validate.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    validate.add_argument("--revision", default=DEFAULT_REVISION)
    validate.add_argument("--device", default="auto")
    validate.add_argument("--canvas-tokens", type=int, default=64)
    validate.add_argument("--seed", type=int, default=0)
    validate.add_argument("--preserve-output-head", action="store_true")
    validate.add_argument("--only-output-head", action="store_true")
    validate.add_argument("--local-files-only", action="store_true")

    validate_generation = subparsers.add_parser(
        "validate-mdlm-output-head-int8-generation",
        help="compare full FP32 and output-head INT8 candidate-cached samples",
    )
    validate_generation.add_argument("--out", type=Path)
    validate_generation.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    validate_generation.add_argument("--revision", default=DEFAULT_REVISION)
    validate_generation.add_argument("--device", default="auto")
    validate_generation.add_argument("--canvas-tokens", type=int, default=64)
    validate_generation.add_argument("--steps", type=int, default=64)
    validate_generation.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    validate_generation.add_argument("--local-files-only", action="store_true")

    session = subparsers.add_parser(
        "analyze-session-cache",
        help="compare follow-up recomputation with retained session K/V",
    )
    session.add_argument(
        "--turn",
        action="append",
        type=_conversation_turn,
        required=True,
        help="completed turn as USER_TOKENS:ANSWER_TOKENS; repeat per turn",
    )
    session.add_argument("--layers", type=int, default=32)
    session.add_argument("--hidden-size", type=int, default=4096)
    session.add_argument("--attention-heads", type=int, default=32)
    session.add_argument("--kv-heads", type=int, default=32)
    session.add_argument("--kv-bits", type=int, default=16)
    session.add_argument("--parameter-count", type=int, default=8_000_000_000)
    session.add_argument("--weight-bits", type=int, default=4)
    session.add_argument("--ddr-gib", type=float, default=4.0)
    session.add_argument("--runtime-reserve-mib", type=float, default=512.0)
    session.add_argument(
        "--terminal-kv-available",
        action="store_true",
        help="last denoising evaluation already exposes exact final-token K/V",
    )
    session.add_argument("--out", type=Path)

    lifecycle = subparsers.add_parser(
        "analyze-session-lifecycle",
        help="replay a bounded TTL/LRU session K/V cache and emit a traffic trace",
    )
    lifecycle.add_argument(
        "--request",
        action="append",
        type=_session_request,
        required=True,
        help=(
            "SESSION:ARRIVAL_S:USER_TOKENS:ANSWER_TOKENS:EVALS; repeat in "
            "arrival order"
        ),
    )
    lifecycle.add_argument("--layers", type=int, default=32)
    lifecycle.add_argument("--hidden-size", type=int, default=4096)
    lifecycle.add_argument("--attention-heads", type=int, default=32)
    lifecycle.add_argument("--kv-heads", type=int, default=32)
    lifecycle.add_argument("--kv-bits", type=int, default=16)
    lifecycle.add_argument("--parameter-count", type=int, default=8_000_000_000)
    lifecycle.add_argument("--weight-bits", type=int, default=4)
    lifecycle.add_argument("--ddr-gib", type=float, default=4.0)
    lifecycle.add_argument("--runtime-reserve-mib", type=float, default=512.0)
    lifecycle.add_argument("--capacity-mib", type=float)
    lifecycle.add_argument("--ttl-s", type=float, default=300.0)
    lifecycle.add_argument("--terminal-kv-available", action="store_true")
    lifecycle.add_argument("--trace-out", type=Path)
    lifecycle.add_argument("--out", type=Path)

    candidate = subparsers.add_parser(
        "validate-candidate-cache",
        help="validate compact candidate caching against DDPM transition sampling",
    )
    candidate.add_argument("--positions", type=int, default=64)
    candidate.add_argument("--vocabulary-size", type=int, default=50_258)
    candidate.add_argument("--probability-bits", type=int, default=16)
    candidate.add_argument("--analytic-vocabulary-size", type=int, default=257)
    candidate.add_argument("--transitions", type=int, default=16)
    candidate.add_argument("--monte-carlo-trials", type=int, default=200_000)
    candidate.add_argument("--monte-carlo-vocabulary-size", type=int, default=16)
    candidate.add_argument("--seed", type=int, default=0)
    candidate.add_argument("--maximum-total-variation", type=float, default=0.01)
    candidate.add_argument("--out", type=Path)

    candidate_kernel = subparsers.add_parser(
        "analyze-candidate-kernel",
        help="estimate cycle and stream traffic for the candidate reveal RTL",
    )
    candidate_kernel.add_argument("--positions", type=int, default=64)
    candidate_kernel.add_argument("--vocabulary-size", type=int, default=50_258)
    candidate_kernel.add_argument("--cache-hit-transitions", type=int, default=23)
    candidate_kernel.add_argument("--clock-mhz", type=float, default=300.0)
    candidate_kernel.add_argument("--initiation-interval", type=int, default=1)
    candidate_kernel.add_argument("--measured-model-forward-ms", type=float, default=0.0)
    candidate_kernel.add_argument("--out", type=Path)

    candidate_rtl = subparsers.add_parser(
        "synthesize-candidate-rtl",
        help="technology-map candidate reveal RTL with optional Yosys",
    )
    candidate_rtl.add_argument(
        "--rtl",
        type=Path,
        default=Path("rtl/candidate_reveal/candidate_reveal_stream.sv"),
    )
    candidate_rtl.add_argument("--top", default="candidate_reveal_stream")
    candidate_rtl.add_argument("--family", choices=["xcup", "xcu", "xc7"], default="xcup")
    candidate_rtl.add_argument(
        "--device-reference",
        choices=["k26", "none"],
        default="k26",
        help="compare mapped primitives with a documented device capacity",
    )
    candidate_rtl.add_argument("--out", type=Path)

    generic_rtl = subparsers.add_parser(
        "synthesize-rtl",
        help="technology-map a standalone SystemVerilog top with Yosys",
    )
    generic_rtl.add_argument("--rtl", type=Path, nargs="+", required=True)
    generic_rtl.add_argument("--top", required=True)
    generic_rtl.add_argument(
        "--family", choices=["xcup", "xcu", "xc7"], default="xcup"
    )
    generic_rtl.add_argument(
        "--device-reference", choices=["k26", "none"], default="k26"
    )
    generic_rtl.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="NAME=INTEGER",
        help="override a top-level integer parameter; may be repeated",
    )
    generic_rtl.add_argument("--out", type=Path)

    candidate_producer = subparsers.add_parser(
        "validate-streaming-candidate-producer",
        help="validate fused streaming sampling and model its output-head cost",
    )
    candidate_producer.add_argument("--positions", type=int, default=64)
    candidate_producer.add_argument("--vocabulary-size", type=int, default=50_258)
    candidate_producer.add_argument("--hidden-size", type=int, default=768)
    candidate_producer.add_argument("--trials", type=int, default=8)
    candidate_producer.add_argument("--chunk-size", type=int, default=512)
    candidate_producer.add_argument("--seed", type=int, default=0)
    candidate_producer.add_argument("--weight-bits", type=int, default=16)
    candidate_producer.add_argument("--activation-bits", type=int, default=16)
    candidate_producer.add_argument("--score-bits", type=int, default=32)
    candidate_producer.add_argument("--mac-lanes", type=int, default=1024)
    candidate_producer.add_argument("--clock-mhz", type=float, default=300.0)
    candidate_producer.add_argument("--ddr-bandwidth-gbps", type=float, default=19.2)
    candidate_producer.add_argument("--ddr-efficiency", type=float, default=0.65)
    candidate_producer.add_argument("--out", type=Path)

    output_head_sweep = subparsers.add_parser(
        "analyze-output-head-sweep",
        help="sweep output-head MAC lanes over a measured mask schedule",
    )
    output_head_sweep.add_argument("--trace", type=Path, required=True)
    output_head_sweep.add_argument("--weight-bits", nargs="+", type=int, default=[16, 8])
    output_head_sweep.add_argument(
        "--mac-lanes", nargs="+", type=int, default=[256, 512, 768, 1024, 1248]
    )
    output_head_sweep.add_argument("--clock-mhz", type=float, default=300.0)
    output_head_sweep.add_argument("--ddr-bandwidth-gbps", type=float, default=19.2)
    output_head_sweep.add_argument("--ddr-efficiency", type=float, default=0.65)
    output_head_sweep.add_argument("--out", type=Path)

    tiled_output_head = subparsers.add_parser(
        "validate-tiled-output-head",
        help="validate tiled projection and fused noisy candidate reduction",
    )
    tiled_output_head.add_argument("--positions", type=int, default=7)
    tiled_output_head.add_argument("--hidden-size", type=int, default=19)
    tiled_output_head.add_argument("--vocabulary-size", type=int, default=37)
    tiled_output_head.add_argument("--trials", type=int, default=32)
    tiled_output_head.add_argument("--position-tile", type=int, default=4)
    tiled_output_head.add_argument("--vocabulary-tile", type=int, default=8)
    tiled_output_head.add_argument("--seed", type=int, default=0)
    tiled_output_head.add_argument("--hardware-vocabulary-tile", type=int, default=16)
    tiled_output_head.add_argument("--out", type=Path)

    rng_gumbel = subparsers.add_parser(
        "validate-rng-gumbel",
        help="validate Philox vectors and fixed-point Gumbel LUT precision",
    )
    rng_gumbel.add_argument("--distribution-trials", type=int, default=200_000)
    rng_gumbel.add_argument("--distribution-vocabulary-size", type=int, default=16)
    rng_gumbel.add_argument("--full-vocabulary-size", type=int, default=50_258)
    rng_gumbel.add_argument("--full-vocabulary-batches", type=int, default=16)
    rng_gumbel.add_argument("--full-vocabulary-batch-size", type=int, default=64)
    rng_gumbel.add_argument("--mantissa-bits", nargs="+", type=int, default=[4, 6, 8])
    rng_gumbel.add_argument("--fraction-bits", type=int, default=10)
    rng_gumbel.add_argument("--seed", type=int, default=0)
    rng_gumbel.add_argument("--maximum-distribution-tv", type=float, default=0.01)
    rng_gumbel.add_argument(
        "--minimum-full-vocabulary-agreement", type=float, default=0.995
    )
    rng_gumbel.add_argument("--out", type=Path)

    rng_hardware = subparsers.add_parser(
        "analyze-rng-hardware",
        help="compare unrolled and iterative Philox hardware throughput",
    )
    rng_hardware.add_argument("--positions", type=int, default=64)
    rng_hardware.add_argument("--vocabulary-size", type=int, default=50_258)
    rng_hardware.add_argument("--hidden-size", type=int, default=768)
    rng_hardware.add_argument("--output-head-mac-lanes", type=int, default=1_024)
    rng_hardware.add_argument("--clock-mhz", type=float, default=300.0)
    rng_hardware.add_argument("--gumbel-lanes", type=int, default=2)
    rng_hardware.add_argument("--maximum-iterative-cores", type=int, default=8)
    rng_hardware.add_argument(
        "--unrolled-synthesis",
        type=Path,
        default=Path("data/results/philox-gumbel-yosys-xcup.json"),
    )
    rng_hardware.add_argument(
        "--iterative-synthesis",
        type=Path,
        default=Path("data/results/philox-iterative-yosys-xcup.json"),
    )
    rng_hardware.add_argument(
        "--farm-synthesis",
        type=Path,
        default=Path("data/results/philox-farm-yosys-xcup.json"),
    )
    rng_hardware.add_argument(
        "--gumbel-synthesis",
        type=Path,
        default=Path("data/results/gumbel-dual-yosys-xcup.json"),
    )
    rng_hardware.add_argument(
        "--integrated-synthesis",
        type=Path,
        default=Path(
            "data/results/philox-gumbel-farm-stream-yosys-xcup.json"
        ),
    )
    rng_hardware.add_argument("--out", type=Path)

    drift = subparsers.add_parser(
        "analyze-mdlm-prefix-drift",
        help="measure real per-layer prefix K/V drift after a suffix edit",
    )
    drift.add_argument("--out", type=Path)
    drift.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    drift.add_argument("--revision", default=DEFAULT_REVISION)
    drift.add_argument("--device", default="auto")
    drift.add_argument("--prefix-text", default=DEFAULT_PREFIX_TEXT)
    drift.add_argument("--suffix-tokens", type=int, default=32)
    drift.add_argument("--changed-suffix-tokens", type=int, default=1)
    drift.add_argument("--warmup", type=int, default=1)
    drift.add_argument("--timing-repeats", type=int, default=5)
    drift.add_argument("--seed", type=int, default=0)
    drift.add_argument("--local-files-only", action="store_true")

    quality = subparsers.add_parser(
        "evaluate-mdlm-prefix-isolation",
        help="run the held-out reconstruction quality gate",
    )
    quality.add_argument("--out", type=Path)
    quality.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    quality.add_argument("--revision", default=DEFAULT_REVISION)
    quality.add_argument("--device", default="auto")
    quality.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    quality.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    quality.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    quality.add_argument("--split", default="test")
    quality.add_argument(
        "--prefix-lengths", nargs="+", type=int, default=[16, 64, 128]
    )
    quality.add_argument(
        "--suffix-lengths", nargs="+", type=int, default=[16, 32, 64]
    )
    quality.add_argument("--samples", type=int, default=8)
    quality.add_argument("--seed", type=int, default=0)
    quality.add_argument(
        "--maximum-accuracy-drop-fraction", type=float, default=0.05
    )
    quality.add_argument(
        "--maximum-nll-increase-fraction", type=float, default=0.05
    )
    quality.add_argument("--maximum-logit-nrmse", type=float, default=1e-5)
    quality.add_argument("--local-files-only", action="store_true")

    conditioned = subparsers.add_parser(
        "generate-mdlm-conditioned",
        help="compare conditioned DDPM generation with retained prefix K/V",
    )
    conditioned.add_argument("--out", type=Path)
    conditioned.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    conditioned.add_argument("--revision", default=DEFAULT_REVISION)
    conditioned.add_argument("--device", default="auto")
    conditioned.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    conditioned.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    conditioned.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    conditioned.add_argument("--split", default="test")
    conditioned.add_argument("--prefix-tokens", type=int, default=64)
    conditioned.add_argument("--suffix-tokens", type=int, default=32)
    conditioned.add_argument("--samples", type=int, default=3)
    conditioned.add_argument("--steps", type=int, default=64)
    conditioned.add_argument("--seed", type=int, default=0)
    conditioned.add_argument("--local-files-only", action="store_true")

    multiturn = subparsers.add_parser(
        "run-mdlm-two-turn",
        help="run an exact block-causal two-turn retained-K/V session",
    )
    multiturn.add_argument("--out", type=Path)
    multiturn.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    multiturn.add_argument("--revision", default=DEFAULT_REVISION)
    multiturn.add_argument("--device", default="auto")
    multiturn.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    multiturn.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    multiturn.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    multiturn.add_argument("--split", default="test")
    multiturn.add_argument("--prefix-tokens", type=int, default=64)
    multiturn.add_argument("--first-answer-tokens", type=int, default=16)
    multiturn.add_argument("--followup-tokens", type=int, default=16)
    multiturn.add_argument("--second-answer-tokens", type=int, default=16)
    multiturn.add_argument("--samples", type=int, default=3)
    multiturn.add_argument("--steps", type=int, default=64)
    multiturn.add_argument("--seed", type=int, default=0)
    multiturn.add_argument("--local-files-only", action="store_true")

    optimize = subparsers.add_parser(
        "optimize", help="apply an architectural what-if transform to a trace"
    )
    optimize.add_argument("--trace", type=Path, required=True)
    optimize.add_argument("--out", type=Path, required=True)
    optimize.add_argument(
        "--optimization",
        choices=[
            "masked-output-head",
            "fused-streaming-candidate-head",
            "weight-int8",
            "weight-int8-mixed-head",
            "weight-int4",
        ],
        required=True,
    )

    run = subparsers.add_parser("simulate", help="simulate a JSONL trace")
    run.add_argument("--trace", type=Path, required=True)
    run.add_argument("--hardware", type=Path, required=True)
    run.add_argument(
        "--policy", choices=["all-hbm", "canvas-sram"], required=True
    )
    run.add_argument("--out", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "inspect-model-spec":
        manifest = load_model_spec(args.model).hardware_manifest()
        rendered = json.dumps(manifest, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "analyze-realtime-target":
        report = analyze_realtime_target(args.target, args.project_root)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "export-mdlm-hardware-package":
        manifest = export_mdlm_hardware_package(
            args.out_dir,
            model_id=args.model_id,
            revision=args.revision,
            device=args.device,
            canvas_tokens=64,
            local_files_only=args.local_files_only,
        )
        print(
            json.dumps(
                {
                    "package_dir": str(args.out_dir),
                    "constant_fold_validation": manifest[
                        "constant_fold_validation"
                    ],
                    "golden_tensor_count": manifest["goldens"]["tensor_count"],
                    "weight_tensor_count": manifest["weights"]["tensor_count"],
                    "files": manifest["files"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "validate-mdlm-hardware-package":
        report = validate_mdlm_hardware_package(args.package_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "validate-fixed-mlp":
        report = validate_fixed_mlp(
            args.package_dir,
            block=args.block,
            bits=args.activation_bits,
            weight_bits=args.weight_bits,
            activation_granularity=args.activation_granularity,
            smoothquant_alpha=args.smoothquant_alpha,
            mac_lanes=args.mac_lanes,
            clock_mhz=args.clock_mhz,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "validate-hardware-fixed-mlp":
        report = validate_hardware_fixed_mlp(
            args.package_dir,
            block=args.block,
            activation_granularity=args.activation_granularity,
            down_activation_granularity=args.down_activation_granularity,
            smoothquant_alpha=args.smoothquant_alpha,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "sweep-hardware-fixed-mlp":
        report = sweep_hardware_fixed_mlp(
            args.package_dir,
            activation_granularity=args.activation_granularity,
            down_activation_granularity=args.down_activation_granularity,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "export-mlp-interstage":
        report = export_mlp_interstage_package(args.package_dir, args.out)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "export-block-execution-image":
        report = export_block_execution_image(
            args.package_dir,
            args.out,
            block=args.block,
            manifest_path=args.manifest,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "validate-block-execution-image":
        report = validate_block_execution_image(args.manifest)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(1)
        return

    if args.command == "sweep-fixed-layernorm":
        report = sweep_fixed_layer_norm(args.package_dir)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "sweep-fixed-rotary":
        report = sweep_fixed_rotary(args.package_dir)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "sweep-fixed-attention":
        report = sweep_fixed_attention(args.package_dir)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "sweep-qkv-weight-precision":
        report = sweep_qkv_weight_precision(
            args.package_dir,
            weight_bits=tuple(args.weight_bits),
            smoothquant_alphas=(
                tuple(args.smoothquant_alphas)
                if args.smoothquant_alphas
                else (None,)
            ),
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "screen-qkv-int8-logits":
        report = screen_qkv_int8_logits(
            args.package_dir,
            seeds=args.seeds,
            device=args.device,
            smoothquant_alpha=args.smoothquant_alpha,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "screen-packed-int8-attention":
        report = screen_packed_int8_attention(args.package_dir)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "screen-packed-int8-attention-heldout":
        report = screen_packed_int8_attention_heldout(
            args.package_dir,
            seeds=args.seeds,
            device=args.device,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "screen-packed-int8-attention-logits":
        report = screen_packed_int8_attention_logits(
            args.package_dir,
            seeds=args.seeds,
            device=args.device,
            quantize_qk=not args.preserve_qk18,
            quantize_pv=not args.preserve_pv18,
            qk_bits=args.qk_bits,
            qk_scale_mode=args.qk_scale_mode,
            qk_multiplier_fraction_bits=args.qk_multiplier_fraction_bits,
            pv_probability_levels=args.pv_probability_levels,
            pv_probability_sum_correction=args.pv_probability_sum_correction,
            pv_value_bits=args.pv_value_bits,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "analyze-mlp-pingpong":
        report = analyze_mlp_pingpong(
            clock_mhz=args.clock_mhz,
            effective_ddr_gbps=args.effective_ddr_gbps,
            evaluations=args.evaluations,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "sweep-fixed-mlp":
        report = sweep_fixed_mlp(
            args.package_dir,
            block=args.block,
            mac_lanes=args.mac_lanes,
            clock_mhz=args.clock_mhz,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "optimize-fixed-mlp-alphas":
        report = optimize_fixed_mlp_alphas(
            args.package_dir,
            alphas=args.alphas,
            mac_lanes=args.mac_lanes,
            clock_mhz=args.clock_mhz,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "generate-synthetic":
        trace = synthetic_trace(
            steps=args.steps,
            layers=args.layers,
            canvas_tokens=args.canvas_tokens,
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            model_weight_bytes=args.model_weight_bytes,
        )
        write_trace(trace, args.out)
        print(json.dumps({"trace": str(args.out), "steps": len(trace.steps)}, indent=2))
        return

    if args.command == "benchmark-mdlm-apple":
        result = benchmark_mdlm_apple(
            model_id=args.model_id,
            revision=args.revision,
            device=args.device,
            canvas_tokens=args.canvas_tokens,
            steps=args.steps,
            seeds=args.seeds,
            local_files_only=args.local_files_only,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "benchmark-mdlm-mlx":
        from .mlx_mdlm import DEFAULT_SNAPSHOT, benchmark_mlx_mdlm

        result = benchmark_mlx_mdlm(
            snapshot=args.snapshot or DEFAULT_SNAPSHOT,
            golden_tensors=args.golden_tensors,
            dtype=args.dtype,
            canvas_tokens=args.canvas_tokens,
            steps=args.steps,
            seeds=args.seeds,
            output_head_bits=args.output_head_bits,
            fold_constants=args.fold_constants,
            compile_sampler=args.compile_sampler,
            mps_validation_seeds=args.mps_validation_seeds,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "trace-mdlm":
        trace = run_mdlm_trace(
            model_id=args.model_id,
            revision=args.revision,
            device=args.device,
            canvas_tokens=args.canvas_tokens,
            steps=args.steps,
            warmup=args.warmup,
            seed=args.seed,
            sampler=args.sampler,
            local_files_only=args.local_files_only,
        )
        write_trace(trace, args.out)
        print(
            json.dumps(
                {
                    "trace": str(args.out),
                    "steps": len(trace.steps),
                    "sampling_transitions": trace.metadata.get(
                        "sampling_transitions", len(trace.steps)
                    ),
                    "model_evaluations": trace.metadata.get(
                        "model_evaluations", len(trace.steps)
                    ),
                    "probability_cache_hits": trace.metadata.get(
                        "probability_cache_hits", 0
                    ),
                    "device": trace.metadata["device"],
                    "parameter_count": trace.metadata["parameter_count"],
                    "all_tokens_committed": trace.metadata["all_tokens_committed"],
                    "generated_text": trace.metadata.get("generated_text"),
                    "measured_latency_ms": sum(
                        step.metadata["measured_step_latency_ms"]
                        for step in trace.steps
                    ),
                },
                indent=2,
            )
        )
        return

    if args.command == "validate-mdlm-int8":
        result = validate_mdlm_int8(
            model_id=args.model_id,
            revision=args.revision,
            device=args.device,
            canvas_tokens=args.canvas_tokens,
            seed=args.seed,
            preserve_output_head=args.preserve_output_head,
            only_output_head=args.only_output_head,
            local_files_only=args.local_files_only,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "validate-mdlm-output-head-int8-generation":
        result = validate_mdlm_output_head_int8_generation(
            model_id=args.model_id,
            revision=args.revision,
            device=args.device,
            canvas_tokens=args.canvas_tokens,
            steps=args.steps,
            seeds=args.seeds,
            local_files_only=args.local_files_only,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "analyze-session-cache":
        result = analyze_session_cache(
            args.turn,
            SessionCacheConfig(
                layers=args.layers,
                hidden_size=args.hidden_size,
                attention_heads=args.attention_heads,
                kv_heads=args.kv_heads,
                kv_bits=args.kv_bits,
                parameter_count=args.parameter_count,
                weight_bits=args.weight_bits,
                ddr_bytes=int(args.ddr_gib * 1024**3),
                runtime_reserve_bytes=int(args.runtime_reserve_mib * 1024**2),
            ),
            terminal_kv_available=args.terminal_kv_available,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "analyze-session-lifecycle":
        config = SessionCacheConfig(
            layers=args.layers,
            hidden_size=args.hidden_size,
            attention_heads=args.attention_heads,
            kv_heads=args.kv_heads,
            kv_bits=args.kv_bits,
            parameter_count=args.parameter_count,
            weight_bits=args.weight_bits,
            ddr_bytes=int(args.ddr_gib * 1024**3),
            runtime_reserve_bytes=int(args.runtime_reserve_mib * 1024**2),
        )
        capacity_bytes = (
            None
            if args.capacity_mib is None
            else int(args.capacity_mib * 1024**2)
        )
        result, trace = analyze_and_trace_session_lifecycle(
            args.request,
            config,
            ttl_s=args.ttl_s,
            capacity_bytes=capacity_bytes,
            terminal_kv_available=args.terminal_kv_available,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        if args.trace_out:
            write_trace(trace, args.trace_out)
        print(rendered)
        return

    if args.command == "validate-candidate-cache":
        result = validate_candidate_cache_equivalence(
            positions=args.positions,
            vocabulary_size=args.vocabulary_size,
            probability_bits=args.probability_bits,
            analytic_vocabulary_size=args.analytic_vocabulary_size,
            transitions=args.transitions,
            monte_carlo_trials=args.monte_carlo_trials,
            monte_carlo_vocabulary_size=args.monte_carlo_vocabulary_size,
            seed=args.seed,
            maximum_total_variation=args.maximum_total_variation,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "analyze-candidate-kernel":
        result = analyze_candidate_reveal_kernel(
            positions=args.positions,
            vocabulary_size=args.vocabulary_size,
            cache_hit_transitions=args.cache_hit_transitions,
            clock_mhz=args.clock_mhz,
            initiation_interval=args.initiation_interval,
            measured_model_forward_ms=args.measured_model_forward_ms,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "synthesize-candidate-rtl":
        result = synthesize_candidate_reveal(
            args.rtl,
            top=args.top,
            family=args.family,
            device_reference=(
                None if args.device_reference == "none" else args.device_reference
            ),
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "synthesize-rtl":
        parameters = {}
        for assignment in args.parameter:
            if "=" not in assignment:
                raise SystemExit("--parameter must use NAME=INTEGER")
            name, raw_value = assignment.split("=", 1)
            try:
                parameters[name] = int(raw_value, 0)
            except ValueError as error:
                raise SystemExit(
                    "--parameter values must be integers"
                ) from error
        result = synthesize_rtl(
            args.rtl,
            top=args.top,
            family=args.family,
            device_reference=(
                None if args.device_reference == "none" else args.device_reference
            ),
            parameters=parameters,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "validate-streaming-candidate-producer":
        result = validate_streaming_candidate_producer(
            positions=args.positions,
            vocabulary_size=args.vocabulary_size,
            hidden_size=args.hidden_size,
            trials=args.trials,
            chunk_size=args.chunk_size,
            seed=args.seed,
            weight_bits=args.weight_bits,
            activation_bits=args.activation_bits,
            score_bits=args.score_bits,
            mac_lanes=args.mac_lanes,
            clock_mhz=args.clock_mhz,
            ddr_bandwidth_gbps=args.ddr_bandwidth_gbps,
            ddr_efficiency=args.ddr_efficiency,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "analyze-output-head-sweep":
        source = read_trace(args.trace)
        result = analyze_output_head_lane_sweep(
            [step.active_tokens for step in source.steps],
            vocabulary_size=int(source.metadata["vocab_size"]),
            hidden_size=int(source.metadata["hidden_size"]),
            weight_bits=args.weight_bits,
            mac_lanes=args.mac_lanes,
            clock_mhz=args.clock_mhz,
            ddr_bandwidth_gbps=args.ddr_bandwidth_gbps,
            ddr_efficiency=args.ddr_efficiency,
        )
        result["trace"] = str(args.trace)
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "validate-tiled-output-head":
        result = validate_tiled_output_head(
            positions=args.positions,
            hidden_size=args.hidden_size,
            vocabulary_size=args.vocabulary_size,
            trials=args.trials,
            position_tile=args.position_tile,
            vocabulary_tile=args.vocabulary_tile,
            seed=args.seed,
            hardware_vocabulary_tile=args.hardware_vocabulary_tile,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "validate-rng-gumbel":
        result = validate_rng_and_gumbel(
            distribution_trials=args.distribution_trials,
            distribution_vocabulary_size=args.distribution_vocabulary_size,
            full_vocabulary_size=args.full_vocabulary_size,
            full_vocabulary_batches=args.full_vocabulary_batches,
            full_vocabulary_batch_size=args.full_vocabulary_batch_size,
            mantissa_bits=args.mantissa_bits,
            fraction_bits=args.fraction_bits,
            seed=args.seed,
            maximum_distribution_tv=args.maximum_distribution_tv,
            minimum_full_vocabulary_agreement=(
                args.minimum_full_vocabulary_agreement
            ),
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "analyze-rng-hardware":
        unrolled_result = json.loads(
            args.unrolled_synthesis.read_text(encoding="utf-8")
        )
        iterative_result = json.loads(
            args.iterative_synthesis.read_text(encoding="utf-8")
        )
        farm_result = json.loads(args.farm_synthesis.read_text(encoding="utf-8"))
        gumbel_result = json.loads(
            args.gumbel_synthesis.read_text(encoding="utf-8")
        )
        integrated_result = json.loads(
            args.integrated_synthesis.read_text(encoding="utf-8")
        )
        result = analyze_rng_hardware(
            positions=args.positions,
            vocabulary_size=args.vocabulary_size,
            hidden_size=args.hidden_size,
            output_head_mac_lanes=args.output_head_mac_lanes,
            clock_mhz=args.clock_mhz,
            gumbel_lanes=args.gumbel_lanes,
            maximum_iterative_cores=args.maximum_iterative_cores,
            unrolled_usage=unrolled_result["device_capacity_comparison"][
                "mapped_usage"
            ],
            iterative_core_usage=iterative_result[
                "device_capacity_comparison"
            ]["mapped_usage"],
            four_core_farm_usage=farm_result["device_capacity_comparison"][
                "mapped_usage"
            ],
            dual_gumbel_usage=gumbel_result["device_capacity_comparison"][
                "mapped_usage"
            ],
            integrated_stream_usage=integrated_result[
                "device_capacity_comparison"
            ]["mapped_usage"],
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "analyze-mdlm-prefix-drift":
        result = analyze_mdlm_prefix_drift(
            model_id=args.model_id,
            revision=args.revision,
            device=args.device,
            prefix_text=args.prefix_text,
            suffix_tokens=args.suffix_tokens,
            changed_suffix_tokens=args.changed_suffix_tokens,
            warmup=args.warmup,
            timing_repeats=args.timing_repeats,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "evaluate-mdlm-prefix-isolation":
        result = evaluate_mdlm_prefix_isolation_quality(
            model_id=args.model_id,
            revision=args.revision,
            device=args.device,
            dataset_id=args.dataset_id,
            dataset_config=args.dataset_config,
            dataset_revision=args.dataset_revision,
            split=args.split,
            prefix_lengths=args.prefix_lengths,
            suffix_lengths=args.suffix_lengths,
            samples=args.samples,
            seed=args.seed,
            maximum_accuracy_drop_fraction=args.maximum_accuracy_drop_fraction,
            maximum_nll_increase_fraction=args.maximum_nll_increase_fraction,
            maximum_logit_nrmse=args.maximum_logit_nrmse,
            local_files_only=args.local_files_only,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "generate-mdlm-conditioned":
        result = evaluate_conditioned_mdlm_ddpm(
            model_id=args.model_id,
            revision=args.revision,
            device=args.device,
            dataset_id=args.dataset_id,
            dataset_config=args.dataset_config,
            dataset_revision=args.dataset_revision,
            split=args.split,
            prefix_tokens=args.prefix_tokens,
            suffix_tokens=args.suffix_tokens,
            samples=args.samples,
            steps=args.steps,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "run-mdlm-two-turn":
        result = evaluate_two_turn_mdlm_session(
            model_id=args.model_id,
            revision=args.revision,
            device=args.device,
            dataset_id=args.dataset_id,
            dataset_config=args.dataset_config,
            dataset_revision=args.dataset_revision,
            split=args.split,
            prefix_tokens=args.prefix_tokens,
            first_answer_tokens=args.first_answer_tokens,
            followup_tokens=args.followup_tokens,
            second_answer_tokens=args.second_answer_tokens,
            samples=args.samples,
            steps=args.steps,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

    if args.command == "optimize":
        source = read_trace(args.trace)
        if args.optimization == "masked-output-head":
            optimized = masked_output_head(source)
        elif args.optimization == "fused-streaming-candidate-head":
            optimized = fused_streaming_candidate_head(source)
        elif args.optimization == "weight-int8":
            optimized = quantize_weights(source, target_bits=8)
        elif args.optimization == "weight-int8-mixed-head":
            optimized = quantize_weights(
                source,
                target_bits=8,
                preserve_output_head=True,
            )
        elif args.optimization == "weight-int4":
            optimized = quantize_weights(source, target_bits=4)
        else:  # argparse owns the public validation; keep this branch explicit.
            raise ValueError("unknown optimization %s" % args.optimization)
        write_trace(optimized, args.out)
        print(
            json.dumps(
                {
                    "source": str(args.trace),
                    "trace": str(args.out),
                    "optimization": args.optimization,
                    "steps": len(optimized.steps),
                },
                indent=2,
            )
        )
        return

    trace = read_trace(args.trace)
    result = simulate(trace, load_hardware(args.hardware), _policy(args.policy))
    payload = result.to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
