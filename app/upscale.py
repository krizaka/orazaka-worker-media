"""Optional video super-resolution after a clip is made: SeedVR2 3B (Apache-2.0) takes 720p to 1080p or 4K.

The Studio asks with `upscale: "1080p" | "2160p"` (absent or "none": the clip is left as the model made it). The
upscaler is a pinned checkout of the SeedVR2 video upscaler (`SEEDVR2_HOME`) run by its own interpreter
(`SEEDVR2_PYTHON`) so its dependencies never touch the generation venv; its weights live under `<models>/seedvr2`.
Measured on an H100 80 GB (docs/_generated/MEASURED_RUNS.md): ~1 min for a 5 s clip to 1080p, ~6 min to 4K.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time

from app import video_models

MODEL = "seedvr2-3b"
DIT = "seedvr2_ema_3b_fp16.safetensors"

# The short side SeedVR2 is asked for, and the exact frame the clip leaves in (16:9, the aspect is kept by cutting the sides).
TARGETS = {"1080p": (1920, 1080), "2160p": (3840, 2160), "4k": (3840, 2160)}


class InvalidUpscale(video_models.InvalidVideoRequest):
    """An `upscale` value this worker cannot make: the caller's to fix."""


def requested(payload: dict) -> str | None:
    """The target the caller asked for ("1080p" or "2160p"), None for no upscale; anything else is refused by name."""
    value = payload.get("upscale")
    if value in (None, "", "none", False):
        return None
    key = str(value).strip().lower()
    if key not in TARGETS:
        raise InvalidUpscale(f"upscale '{value}' is not one of none, 1080p, 2160p")
    return "2160p" if key == "4k" else key


def _config(env) -> tuple[str, str, str]:
    home, python = env.get("SEEDVR2_HOME"), env.get("SEEDVR2_PYTHON")
    if not home or not python:
        raise InvalidUpscale("this worker has no video upscaler installed (SEEDVR2_HOME / SEEDVR2_PYTHON)")
    return home, python, os.path.join(env.get("MEDIA_MODELS_DIR", "/models"), "seedvr2")


def _size(path: str, run) -> tuple[int, int]:
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
    width, height = out.strip().split(",")[:2]
    return int(width), int(height)


def upscale(source: str, target_key: str, run=subprocess.run, env=os.environ, clock=time.monotonic) -> dict:
    """Replaces `source` (an mp4) in place by its upscaled version; returns the facts the run reports."""
    home, python, models = _config(env)
    width, height = TARGETS[target_key]
    before = _size(source, run)
    started = clock()
    with tempfile.TemporaryDirectory(prefix="orazaka-upscale-") as scratch:
        raw = os.path.join(scratch, "seedvr2.mp4")
        run(
            [python, os.path.join(home, "inference_cli.py"), source, "--output", raw, "--resolution", str(height), "--dit_model", DIT,
             "--model_dir", models, "--batch_size", "5", "--attention_mode", "sdpa", "--color_correction", "lab", "--output_format", "mp4",
             *(["--vae_encode_tiled", "--vae_decode_tiled"] if height > 1080 else [])],
            check=True, cwd=home,
        )
        final = os.path.join(scratch, "final.mp4")
        # SeedVR2 keeps the source's aspect (1280x704 is a little wider than 16:9): scale to the height, cut the sides to the exact frame.
        run(
            ["ffmpeg", "-v", "error", "-y", "-i", raw, "-vf", f"scale=-2:{height}:flags=lanczos,crop={width}:{height}", "-c:v", "libx264", "-crf", "16",
             "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", final],
            check=True,
        )
        os.replace(final, source)
    return {
        "upscaleModel": MODEL,
        "upscaleFrom": f"{before[0]}x{before[1]}",
        "upscaleTo": f"{width}x{height}",
        "upscaleSeconds": round(clock() - started, 1),
    }
