#!/usr/bin/env python3
"""Compare prompt-conditioned MDLM and autoregressive generation on one Mac."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import time

import mlx.core as mx

from diffusion_accel.mlx_mdlm import (
    CiderQuantizationPlan,
    MLXQuantizationPlan,
    load_mlx_mdlm,
    run_mlx_event_sampler,
)
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SMOLLM_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--SmolLM-135M-4bit"
    / "snapshots/f56bc6adfb74c794203dc8ca94e0bccfe2bcd6cc"
)


def _repeated_ngram_fraction(texts: list[str], order: int) -> float:
    repeated = 0
    total = 0
    for text in texts:
        words = text.split()
        ngrams = [tuple(words[index : index + order]) for index in range(len(words) - order + 1)]
        repeated += len(ngrams) - len(set(ngrams))
        total += len(ngrams)
    return repeated / total if total else 0.0


def _load_windows(samples: int) -> tuple[object, list[list[int]]]:
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tokenizer.model_max_length = 1_000_000
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        revision=DATASET_REVISION,
        download_config=DownloadConfig(local_files_only=True),
    )
    text = "\n\n".join(value for value in dataset["text"] if value.strip())
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    starts = random.Random(0).sample(range(len(token_ids) - 127), samples)
    windows = [token_ids[start : start + 128] for start in starts]
    return tokenizer, windows


def _run_mdlm(
    *,
    windows: list[list[int]],
    quantization_plan: MLXQuantizationPlan,
    cider_plan: CiderQuantizationPlan | None,
) -> tuple[list[list[int]], list[float]]:
    model = load_mlx_mdlm(
        dtype="float32",
        sequence_length=128,
        quantization_plan=quantization_plan,
        cider_quantization_plan=cider_plan,
    )
    logits = mx.compile(model.selected_logits)
    run_mlx_event_sampler(
        model,
        canvas_tokens=64,
        prefix_token_ids=windows[0][:64],
        steps=64,
        seed=10_000,
        logits_function=logits,
    )
    outputs = []
    throughputs = []
    for index, window in enumerate(windows):
        output, metrics = run_mlx_event_sampler(
            model,
            canvas_tokens=64,
            prefix_token_ids=window[:64],
            steps=64,
            seed=index,
            logits_function=logits,
        )
        outputs.append(output[0, 64:].tolist())
        throughputs.append(float(metrics["output_tokens_per_second"]))
    del logits, model
    mx.clear_cache()
    gc.collect()
    return outputs, throughputs


def _run_smollm(
    prompts: list[str],
    *,
    temperature: float,
    top_p: float = 0.0,
) -> tuple[list[str], list[float]]:
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(str(SMOLLM_SNAPSHOT))
    sampler = make_sampler(temp=temperature, top_p=top_p)

    def generate(prompt: str, seed: int) -> tuple[str, float]:
        prompt_ids = mx.array(tokenizer.encode(prompt))
        mx.random.seed(seed)
        mx.synchronize()
        started = time.perf_counter()
        output_ids = []
        for token, _ in generate_step(
            prompt_ids,
            model,
            max_tokens=64,
            sampler=sampler,
        ):
            output_ids.append(int(token))
        mx.synchronize()
        elapsed = time.perf_counter() - started
        return tokenizer.decode(output_ids), len(output_ids) / elapsed

    generate(prompts[0], 10_000)
    outputs = []
    throughputs = []
    for index, prompt in enumerate(prompts):
        output, throughput = generate(prompt, index)
        outputs.append(output)
        throughputs.append(throughput)
    mx.clear_cache()
    gc.collect()
    return outputs, throughputs


def _score_completions(
    prompts: list[str],
    configurations: dict[str, list[str]],
) -> dict[str, dict[str, float | int]]:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    scorer_name = "openai-community/gpt2-large"
    tokenizer = AutoTokenizer.from_pretrained(scorer_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        scorer_name,
        dtype=torch.float16,
        local_files_only=True,
    ).to("mps")
    model.eval()
    results = {}
    with torch.inference_mode():
        for name, completions in configurations.items():
            negative_log_likelihood = 0.0
            scored_tokens = 0
            for prompt, completion in zip(prompts, completions):
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                completion_ids = tokenizer.encode(completion, add_special_tokens=False)
                token_ids = torch.tensor(
                    [prompt_ids + completion_ids],
                    dtype=torch.long,
                    device="mps",
                )
                logits = model(token_ids).logits[:, :-1].float()
                targets = token_ids[:, 1:]
                start = max(0, len(prompt_ids) - 1)
                loss = functional.cross_entropy(
                    logits[:, start:].reshape(-1, logits.shape[-1]),
                    targets[:, start:].reshape(-1),
                    reduction="sum",
                )
                negative_log_likelihood += float(loss.item())
                scored_tokens += len(completion_ids)
            mean_nll = negative_log_likelihood / scored_tokens
            results[name] = {
                "gpt2_large_conditional_nll": mean_nll,
                "gpt2_large_conditional_perplexity": math.exp(mean_nll),
                "scored_gpt2_tokens": scored_tokens,
                "repeated_bigram_fraction": _repeated_ngram_fraction(completions, 2),
                "repeated_trigram_fraction": _repeated_ngram_fraction(completions, 3),
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer, windows = _load_windows(args.samples)
    prompts = [tokenizer.decode(window[:64]) for window in windows]
    reference = [tokenizer.decode(window[64:]) for window in windows]
    all_blocks = tuple(range(12))
    configurations = {
        "mdlm_q8_head": (MLXQuantizationPlan(output_head_bits=8), None),
        "mdlm_core_w8a8": (
            MLXQuantizationPlan(output_head_bits=8),
            CiderQuantizationPlan(),
        ),
        "mdlm_maximum_w8a8": (
            MLXQuantizationPlan(output_head_bits=8),
            CiderQuantizationPlan(mlp_down_layers=all_blocks),
        ),
        "mdlm_maximum_q4_head": (
            MLXQuantizationPlan(group_size=32, output_head_bits=4),
            CiderQuantizationPlan(mlp_down_layers=all_blocks),
        ),
    }
    generated_text = {"wikitext_reference": reference}
    speed = {}
    for name, (quantization_plan, cider_plan) in configurations.items():
        output_ids, throughputs = _run_mdlm(
            windows=windows,
            quantization_plan=quantization_plan,
            cider_plan=cider_plan,
        )
        generated_text[name] = [tokenizer.decode(output) for output in output_ids]
        speed[name] = {
            "median_output_tokens_per_second": statistics.median(throughputs),
            "mean_output_tokens_per_second": statistics.mean(throughputs),
        }
        print(json.dumps({"generated": name, **speed[name]}), flush=True)

    for name, temperature, top_p in (
        ("smollm_135m_4bit_sampled", 1.0, 0.0),
        ("smollm_135m_4bit_typical", 0.7, 0.9),
        ("smollm_135m_4bit_greedy", 0.0, 0.0),
    ):
        smollm_outputs, smollm_throughputs = _run_smollm(
            prompts,
            temperature=temperature,
            top_p=top_p,
        )
        generated_text[name] = smollm_outputs
        speed[name] = {
            "median_output_tokens_per_second": statistics.median(
                smollm_throughputs
            ),
            "mean_output_tokens_per_second": statistics.mean(smollm_throughputs),
        }
        print(json.dumps({"generated": name, **speed[name]}), flush=True)

    quality = _score_completions(prompts, generated_text)
    results = {
        "status": "experimental-common-scorer-screen",
        "machine": "Apple M5 Pro, 16-core GPU, 24 GiB unified memory",
        "software": {
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "transformers": importlib.metadata.version("transformers"),
        },
        "models": {
            "mdlm": "kuleshov-group/mdlm-owt@d0958fa851335ece6c15260ce0025f030673c0fb",
            "smollm": "mlx-community/SmolLM-135M-4bit@f56bc6adfb74c794203dc8ca94e0bccfe2bcd6cc",
            "scorer": "openai-community/gpt2-large@32b71b12589c2f8d625668d2335a01cac3249519",
        },
        "protocol": {
            "dataset": "Salesforce/wikitext:wikitext-2-raw-v1@test",
            "dataset_revision": DATASET_REVISION,
            "samples": args.samples,
            "prompt_tokens": 64,
            "generated_model_tokens": 64,
            "document_boundary": "plain newlines, with no tokenizer-specific EOS token",
            "sampling": (
                "MDLM and sampled SmolLM use the full categorical distribution "
                "at temperature 1.0; typical SmolLM uses temperature 0.7 and "
                "top-p 0.9; greedy SmolLM uses argmax"
            ),
            "external_scorer": "openai-community/gpt2-large float16 on MPS",
        },
        "configurations": {
            name: {**speed.get(name, {}), **metrics}
            for name, metrics in quality.items()
        },
        "examples": [
            {
                "prompt": prompts[index],
                **{name: texts[index] for name, texts in generated_text.items()},
            }
            for index in range(min(3, args.samples))
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["configurations"], indent=2))


if __name__ == "__main__":
    main()
