"""Wan 2.x text-to-video on CUDA (diffusers) — the 9:16 clips of the GPU droplet (ADR-077).

One pipeline is resident at a time (the worker serialises media work); asking for the other Wan
model evicts the first. Weights come from the models volume (``HF_HOME``), never from the network at
run time: ``app.fetch_models`` put them there when the host started, and a missing file fails the job
loudly rather than downloading 34 GB in the middle of it.

§12 Zero-Fallback: a failure raises. Nothing here ever returns a placeholder frame.
"""
from __future__ import annotations

import gc
import os
import time
from typing import Callable, Optional

from app.video_models import VideoModel, find, load_manifest

#: The negative prompt of Wan's own examples, in English: what a short must not look like.
DEFAULT_NEGATIVE_PROMPT = (
    "vivid tones, overexposed, static, blurred details, subtitles, text, watermark, logo, style, "
    "artwork, painting, still picture, overall gray, worst quality, low quality, JPEG artifacts, "
    "ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, disfigured, malformed "
    "limbs, fused fingers, motionless frame, cluttered background, three legs, many people in the "
    "background, walking backwards"
)

_pipe = None
_pipe_name: str | None = None


def _torch():
    import torch  # noqa: PLC0415 — imported on first use: the worker starts without a GPU stack

    return torch


def _device() -> str:
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("Wan needs a CUDA (or Metal) device; this host has neither")


def should_offload(model: VideoModel, total_vram_gb: float, env: dict | None = None) -> bool:
    """Whether to let diffusers move each component to the GPU only while it runs.

    ``MEDIA_VIDEO_OFFLOAD`` decides when set (``true``/``false``); ``auto`` (the default) offloads
    when the card cannot hold the model's working set beside the image engine (``vramGb`` + 12 GB).
    Offloading costs seconds per clip; not offloading on a card too small costs the clip.
    """
    source = os.environ if env is None else env
    choice = (source.get("MEDIA_VIDEO_OFFLOAD") or "auto").strip().lower()
    if choice in ("true", "1", "yes"):
        return True
    if choice in ("false", "0", "no"):
        return False
    return total_vram_gb < model.vram_gb + 12


def _load(model: VideoModel):
    """Loads one Wan pipeline from the local cache, evicting whichever was resident."""
    global _pipe, _pipe_name  # noqa: PLW0603 — one resident pipeline per process, by design
    torch = _torch()
    from diffusers import AutoencoderKLWan, WanPipeline  # noqa: PLC0415

    if _pipe is not None:
        print(f"[wan] evicting {_pipe_name}", flush=True)
        _pipe = None
        _pipe_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    device = _device()
    started = time.time()
    common = {"revision": model.revision, "local_files_only": True}
    vae = AutoencoderKLWan.from_pretrained(
        model.repo, subfolder="vae", torch_dtype=torch.float32, **common
    )
    extra = {}
    if model.text_encoder_from:
        # Same umt5-xxl as the donor, kept in bf16 there: loaded from it rather than downloading a
        # float32 copy twice the size (models.yaml, textEncoderFrom).
        from transformers import UMT5EncoderModel  # noqa: PLC0415

        donor = find(model.text_encoder_from, load_manifest())
        extra["text_encoder"] = UMT5EncoderModel.from_pretrained(
            donor.repo,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            revision=donor.revision,
            local_files_only=True,
        )
    pipe = WanPipeline.from_pretrained(
        model.repo, vae=vae, torch_dtype=torch.bfloat16, **common, **extra
    )
    total_gb = 0.0
    if device == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if device == "cuda" and should_offload(model, total_gb):
        pipe.enable_model_cpu_offload()
        placement = f"model offload ({total_gb:.0f} GB card)"
    else:
        pipe.to(device)
        placement = f"resident on {device}"
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    print(f"[wan] {model.name} loaded in {time.time() - started:.0f}s, {placement}", flush=True)
    _pipe = pipe
    _pipe_name = model.name
    return pipe


def generate_wan_frames(
    model_name: str,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    seed: int,
    negative_prompt: str | None = None,
    progress_callback: Optional[Callable] = None,
) -> list:
    """Runs one Wan text-to-video generation and returns its frames as PIL images.

    Raises:
        RuntimeError: on any failure — the job reports it; no fallback frames exist.
    """
    model = find(model_name)
    if model.pipeline != "wan":
        raise RuntimeError(f"model '{model_name}' is not a Wan model")
    torch = _torch()
    pipe = _pipe if _pipe_name == model.name else _load(model)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    started = time.time()
    try:
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=model.guidance,
            generator=generator,
            output_type="pil",
            callback_on_step_end=progress_callback,
        )
    except Exception as failure:  # §12: report it, never hide it
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(f"Wan inference failed ({model_name}): {failure}") from failure
    frames = output.frames[0]
    if not frames:
        raise RuntimeError(f"Wan returned no frames ({model_name})")
    print(
        f"[wan] {model_name} {width}x{height} {len(frames)} frames, {steps} steps, seed {seed}: "
        f"{time.time() - started:.0f}s",
        flush=True,
    )
    return list(frames)
