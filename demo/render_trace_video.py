#!/usr/bin/env python3
"""Render the measured MDLM cache trace as a short MP4 demo."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1280
HEIGHT = 720
FPS = 30
DURATION_SECONDS = 12

BG = "#07111f"
PANEL = "#0d1b2d"
PANEL_EDGE = "#223551"
TEXT = "#f5f7fb"
MUTED = "#91a4bf"
PURPLE = "#9b7cff"
CYAN = "#39d6db"
GREEN = "#5de2a5"
ORANGE = "#ffb454"


def font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont:
    name = "SFNSMono.ttf" if mono else "SFNS.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/{name}", size)


def load_trace(path: Path) -> tuple[dict, list[dict]]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return records[0]["metadata"], records[1:]


def ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, face, fill=TEXT) -> None:
    box = draw.textbbox((0, 0), value, font=face)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), value, font=face, fill=fill)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=22, fill=PANEL, outline=PANEL_EDGE, width=2)


def draw_transition_row(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    completed: int,
    changed: list[int] | None,
) -> None:
    gap = 3
    cell_width = (width - 63 * gap) / 64
    for index in range(64):
        left = int(x + index * (cell_width + gap))
        right = max(left + 2, int(left + cell_width))
        if index >= completed:
            fill = "#1b2a40"
        elif changed is not None and index > 0 and changed[index] == 0:
            fill = GREEN
        else:
            fill = PURPLE
        draw.rounded_rectangle((left, y, right, y + 13), radius=2, fill=fill)


def draw_canvas(draw: ImageDraw.ImageDraw, active: int, progress: float) -> None:
    start_x, start_y = 233, 511
    size, gap = 17, 8
    revealed = 64 - active
    for index in range(64):
        row, col = divmod(index, 16)
        x = start_x + col * (size + gap)
        y = start_y + row * (size + gap)
        is_revealed = index < revealed
        fill = CYAN if is_revealed else "#273653"
        outline = "#64eef0" if is_revealed else "#344966"
        draw.rounded_rectangle((x, y, x + size, y + size), radius=4, fill=fill, outline=outline)
    draw.text((233, 617), f"{revealed:02d}/64 tokens revealed together", font=font(19, mono=True), fill=MUTED)


def render_frame(
    seconds: float,
    *,
    uncached_ms: float,
    cached_ms: float,
    changed: list[int],
    active: list[int],
    generated_text: str,
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((64, 42), "DiffusionHard", font=font(42), fill=TEXT)
    draw.text((64, 91), "Exact reuse in masked diffusion inference", font=font(24), fill=MUTED)
    draw.rounded_rectangle((1067, 48, 1216, 88), radius=20, fill="#14263d")
    text_center(draw, (1141, 68), "APPLE MPS", font(17, mono=True), CYAN)

    if seconds < 1.35:
        alpha = ease(seconds / 0.6)
        color = tuple(int(int(TEXT[i : i + 2], 16) * alpha) for i in (1, 3, 5))
        text_center(draw, (WIDTH // 2, 320), "64 denoising transitions", font(53), color)
        text_center(draw, (WIDTH // 2, 385), "Same output. Fewer model evaluations.", font(29), MUTED)
        draw.text((64, 670), "Measured trace • MDLM-OWT • 169.6M parameters", font=font(17, mono=True), fill=MUTED)
        return image

    progress = ease((seconds - 1.35) / 6.65)
    completed = min(64, max(1, int(progress * 64) + 1))
    step = completed - 1
    cache_evals = 1 + sum(value > 0 for value in changed[1:completed])
    cache_hits = completed - cache_evals

    panel(draw, (64, 151, 610, 439))
    panel(draw, (638, 151, 1216, 439))

    draw.text((96, 181), "BASELINE", font=font(18, mono=True), fill=MUTED)
    draw.text((96, 215), "Always evaluate", font=font(30), fill=TEXT)
    draw.text((96, 273), f"{completed:02d}", font=font(72, mono=True), fill=PURPLE)
    draw.text((192, 319), "model evaluations", font=font(23), fill=MUTED)
    draw_transition_row(draw, x=96, y=384, width=482, completed=completed, changed=None)

    draw.text((670, 181), "EXACT CACHE", font=font(18, mono=True), fill=GREEN)
    draw.text((670, 215), "Skip unchanged state", font=font(30), fill=TEXT)
    draw.text((670, 273), f"{cache_evals:02d}", font=font(72, mono=True), fill=GREEN)
    draw.text((766, 299), "evaluations", font=font(23), fill=MUTED)
    draw.text((766, 329), f"{cache_hits:02d} exact cache hits", font=font(20, mono=True), fill=GREEN)
    draw_transition_row(draw, x=670, y=384, width=514, completed=completed, changed=changed)

    draw_canvas(draw, active[step], progress)
    draw.text((677, 504), "ONE GENERATED SAMPLE", font=font(17, mono=True), fill=MUTED)
    snippet = generated_text.replace("<|endoftext|>", "").strip()
    words = snippet.split()
    lines = [" ".join(words[:12]), " ".join(words[12:24]), " ".join(words[24:36]) + "…"]
    for index, line in enumerate(lines):
        draw.text((677, 544 + index * 31), line, font=font(20), fill=TEXT)

    draw.ellipse((671, 655, 683, 667), fill=PURPLE)
    draw.text((691, 651), "model eval", font=font(16), fill=MUTED)
    draw.ellipse((808, 655, 820, 667), fill=GREEN)
    draw.text((828, 651), "cache hit", font=font(16), fill=MUTED)

    if seconds >= 8.0:
        overlay_strength = ease((seconds - 8.0) / 0.5)
        overlay = Image.new("RGB", (WIDTH, HEIGHT), BG)
        odraw = ImageDraw.Draw(overlay)
        odraw.text((64, 42), "DiffusionHard", font=font(42), fill=TEXT)
        odraw.text((64, 91), "Measured cache result", font=font(24), fill=MUTED)
        odraw.rounded_rectangle((1067, 48, 1216, 88), radius=20, fill="#14263d")
        text_center(odraw, (1141, 68), "APPLE MPS", font(17, mono=True), CYAN)

        text_center(odraw, (WIDTH // 2, 194), "64 → 38", font(92, mono=True), GREEN)
        text_center(odraw, (WIDTH // 2, 276), "model evaluations", font(27), MUTED)

        uncached_tps = 64_000 / uncached_ms
        cached_tps = 64_000 / cached_ms
        reduction = 100 * (1 - cached_ms / uncached_ms)
        odraw.text((132, 355), f"{uncached_tps:0.1f}", font=font(55, mono=True), fill=PURPLE)
        odraw.text((132, 417), "output tok/s", font=font(20, mono=True), fill=MUTED)
        odraw.text((498, 355), f"{cached_tps:0.1f}", font=font(55, mono=True), fill=GREEN)
        odraw.text((498, 417), "output tok/s", font=font(20, mono=True), fill=MUTED)
        odraw.text((866, 355), f"{reduction:0.1f}%", font=font(55, mono=True), fill=ORANGE)
        odraw.text((866, 417), "less forward time", font=font(20, mono=True), fill=MUTED)
        odraw.line((321, 392, 449, 392), fill=PANEL_EDGE, width=3)
        odraw.polygon(((449, 392), (433, 382), (433, 402)), fill=PANEL_EDGE)

        text_center(odraw, (WIDTH // 2, 520), "Forward-only throughput from a measured 64-token trace", font(25), TEXT)
        text_center(odraw, (WIDTH // 2, 565), "Diffusion generates a block, so this is not autoregressive decode TPS.", font(20), MUTED)
        text_center(odraw, (WIDTH // 2, 630), "No FPGA timing claim • Cache-hit sampling overhead not timed", font(17, mono=True), ORANGE)
        image = Image.blend(image, overlay, overlay_strength)

    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uncached", type=Path, default=Path("data/traces/mdlm-owt-ddpm-seed0.jsonl"))
    parser.add_argument("--cached", type=Path, default=Path("data/traces/mdlm-owt-ddpm-cache-seed0.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("demo/diffusionhard-cache-demo.mp4"))
    parser.add_argument("--poster", type=Path, default=Path("demo/diffusionhard-cache-demo-poster.png"))
    args = parser.parse_args()

    uncached_meta, uncached_steps = load_trace(args.uncached)
    cached_meta, cached_steps = load_trace(args.cached)
    uncached_ms = sum(step["metadata"]["measured_step_latency_ms"] for step in uncached_steps)
    cached_ms = sum(step["metadata"]["measured_step_latency_ms"] for step in cached_steps)
    changed = cached_meta["transition_changed_tokens"]
    active = cached_meta["transition_active_tokens"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    final_frame = None
    for frame_index in range(FPS * DURATION_SECONDS):
        final_frame = render_frame(
            frame_index / FPS,
            uncached_ms=uncached_ms,
            cached_ms=cached_ms,
            changed=changed,
            active=active,
            generated_text=uncached_meta["generated_text"],
        )
        process.stdin.write(final_frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed")
    assert final_frame is not None
    final_frame.save(args.poster, optimize=True)
    print(f"wrote {args.output}")
    print(f"wrote {args.poster}")


if __name__ == "__main__":
    main()
