"""Wan 2.x text- and image-to-video on CUDA (diffusers) — the 9:16 clips of the GPU droplet (ADR-077).

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
from pathlib import Path
from typing import Callable, Optional

from app import activity, job_control, model_cache
from app.video_models import (
    InvalidVideoRequest,
    VideoModel,
    crop_to_aspect,
    find,
    hub_cache,
    load_manifest,
)

#: The negative prompt of Wan's own examples, in English: what a short must not look like.
DEFAULT_NEGATIVE_PROMPT = (
    "vivid tones, overexposed, static, blurred details, subtitles, text, watermark, logo, style, "
    "artwork, painting, still picture, overall gray, worst quality, low quality, JPEG artifacts, "
    "ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, disfigured, malformed "
    "limbs, fused fingers, motionless frame, cluttered background, three legs, many people in the "
    "background, walking backwards, "
    # What image-to-video adds: the artefacts the Krizaka cinematic kit refuses by default.
    "morphing, warping, flickering, jittery, unstable geometry, melting, distorted face, extra limbs, "
    "duplicated objects, watermark, text"
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


def _resident_name(model: VideoModel, from_image: bool) -> str:
    """What `_pipe_name` holds: a model's text pipeline and its image pipeline are two residents."""
    if model.pipeline == "ltx":
        return model.name
    return f"{model.name}:image" if from_image else model.name


def is_resident(model_name: str, from_image: bool = False) -> bool:
    """Whether that model's pipeline is the one already loaded in this process (its memory is already taken)."""
    try:
        return _pipe is not None and _pipe_name == _resident_name(find(model_name), from_image)
    except InvalidVideoRequest:
        return False


def _staged_source(model: VideoModel, progress=None):
    """The model's snapshot on local NVMe (``MEDIA_MODELS_STAGE_DIR``) or warmed by parallel readers
    (``MEDIA_MODELS_PREFETCH=1``); None keeps the plain load from the volume (the model-cache contract, #63)."""
    stage_dir = os.environ.get("MEDIA_MODELS_STAGE_DIR")
    prefetch = os.environ.get("MEDIA_MODELS_PREFETCH") == "1"
    if not stage_dir and not prefetch:
        return None
    try:
        snapshot = model_cache.local_snapshot(model.repo, model.revision, hub_cache(model))
        if stage_dir:
            return str(model_cache.resolve(snapshot, stage_dir, progress=progress))
        model_cache.prefetch(snapshot, progress=progress)
        return None
    except Exception as unusable:  # noqa: BLE001 — staging is an optimisation: the plain load always remains
        print(f"[wan] staging {model.name} unavailable ({unusable}); loading from the volume", flush=True)
        return None


# Reading the weights leaves the GPU idle: the heartbeat says the pod is busy for the whole load (Wan and LTX).
@activity.beats
def _load(model: VideoModel, from_image: bool = False, weights_progress=None):
    """Loads one Wan pipeline from the local cache, evicting whichever was resident.

    ``from_image`` loads the image-to-video class. For Wan 2.2 TI2V 5B the same weights serve both: the
    checkpoint's config says ``expand_timesteps`` and ``WanImageToVideoPipeline`` then conditions the
    first latent frame on the photo (no CLIP image encoder, no second transformer).
    """
    global _pipe, _pipe_name  # noqa: PLW0603 — one resident pipeline per process, by design
    if model.pipeline == "ltx":
        return _load_ltx(model)
    torch = _torch()
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline  # noqa: PLC0415

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
    if hub_cache(model):
        common["cache_dir"] = hub_cache(model)
    staged = _staged_source(model, weights_progress)
    repo = staged or model.repo
    if staged:  # a local directory: no revision, no hub cache
        common = {"local_files_only": True}
    vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32, **common)
    extra = {}
    if model.text_encoder_from:
        # Same umt5-xxl as the donor, kept in bf16 there: loaded from it rather than downloading a
        # float32 copy twice the size (models.yaml, textEncoderFrom).
        from transformers import UMT5EncoderModel  # noqa: PLC0415

        donor = find(model.text_encoder_from, load_manifest())
        donor_staged = _staged_source(donor, weights_progress)
        extra["text_encoder"] = UMT5EncoderModel.from_pretrained(
            donor_staged or donor.repo,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            **({} if donor_staged else {"revision": donor.revision}),
            local_files_only=True,
        )
    if from_image:
        pipe = WanImageToVideoPipeline.from_pretrained(
            repo,
            vae=vae,
            torch_dtype=torch.bfloat16,
            # The 5B's own config; the A14B I2V conditions through its 36-channel input instead.
            **({"expand_timesteps": True} if model.expand_timesteps else {}),
            **common,
            **extra,
        )
    else:
        pipe = WanPipeline.from_pretrained(repo, vae=vae, torch_dtype=torch.bfloat16, **common, **extra)
    if model.lora:
        _load_lora(pipe, model)
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
    loaded = time.time() - started
    stage_note = (
        " (staged on NVMe)" if staged else (" (prefetched)" if os.environ.get("MEDIA_MODELS_PREFETCH") == "1" else "")
    )
    print(f"[wan] {model.name} loaded in {loaded:.1f}s{stage_note}, {placement}", flush=True)
    _pipe = pipe
    _pipe_name = _resident_name(model, from_image)
    return pipe


def lora_plan(model: VideoModel) -> list[tuple[str, str, bool]]:
    """The LoRA files a model's pipeline takes: ``(file, adapter name, into the second expert)``, high-noise first."""
    if not model.lora:
        return []
    plan = [(model.lora["high"], "distill", False)]
    if model.lora.get("low"):
        plan.append((model.lora["low"], "distill_2", True))
    return plan


def _load_lora(pipe, model: VideoModel) -> None:
    """Puts the distillation LoRAs on the two experts (high noise → ``transformer``, low → ``transformer_2``).

    The files were fetched with the weights (``app.fetch_models``) and are read from the cache only. A LoRA that
    does not load fails the model's load: a 4-step schedule on an undistilled expert makes noise, not a clip (§12).
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    spec = model.lora
    for filename, adapter, second in lora_plan(model):
        path = hf_hub_download(
            spec["repo"], filename, revision=spec.get("revision"), local_files_only=True, cache_dir=hub_cache(model)
        )
        pipe.load_lora_weights(path, adapter_name=adapter, **({"load_into_transformer_2": True} if second else {}))
        print(f"[wan] LoRA {filename} on {'transformer_2' if second else 'transformer'}", flush=True)


def _load_ltx(model: VideoModel):
    """Loads LTX-Video (diffusers' LTXConditionPipeline) from the local cache, evicting whichever pipeline was resident.

    One pipeline serves text and image: a still is a condition on the first frame. It stays on the card when the card holds it.
    """
    global _pipe, _pipe_name  # noqa: PLW0603
    torch = _torch()
    from diffusers import LTXConditionPipeline  # noqa: PLC0415

    if _pipe is not None:
        print(f"[ltx] evicting {_pipe_name}", flush=True)
        _pipe = None
        _pipe_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    device = _device()
    started = time.time()
    cache = {"cache_dir": hub_cache(model)} if hub_cache(model) else {}  # `cache: scratch`: the host's own disk
    pipe = LTXConditionPipeline.from_pretrained(
        model.repo, revision=model.revision, torch_dtype=torch.bfloat16, local_files_only=True, **cache
    )
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if device == "cuda" else 0.0
    if device == "cuda" and should_offload(model, total_gb):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    print(f"[ltx] {model.name} loaded in {time.time() - started:.0f}s", flush=True)
    _pipe = pipe
    _pipe_name = model.name
    return pipe


def _run_ltx(
    pipe,
    model: VideoModel,
    prompt,
    negative,
    width,
    height,
    num_frames,
    steps,
    generator,
    image,
    focus,
    progress_callback,
):
    """One LTX generation: the still (cut to the frame's aspect) conditions frame 0, the distilled settings of the model do the rest."""
    import tempfile  # noqa: PLC0415

    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition  # noqa: PLC0415
    from diffusers.utils import export_to_video, load_video  # noqa: PLC0415

    conditions = None
    with tempfile.TemporaryDirectory(prefix="orazaka-ltx-") as scratch:
        if image is not None:
            start = os.path.join(scratch, "start.mp4")
            export_to_video([crop_to_aspect(image, width, height, focus)], start, fps=model.fps)
            conditions = [LTXVideoCondition(video=load_video(start), frame_index=0)]
        return pipe(
            conditions=conditions,
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=model.guidance,
            decode_timestep=0.05,
            decode_noise_scale=0.025,
            generator=generator,
            output_type="pil",
            callback_on_step_end=progress_callback,
        ).frames[0]


def _is_out_of_memory(failure: BaseException) -> bool:
    """Whether a CUDA run died for want of memory — the one failure a smaller size can answer."""
    torch = _torch()
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    return bool((oom and isinstance(failure, oom)) or "out of memory" in str(failure).lower())


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
    image=None,
    focus: tuple[float, float] = (0.5, 0.5),
    fallback_sizes: tuple = (),
    weights_progress: Optional[Callable] = None,
) -> list:
    """Runs one Wan generation and returns its frames as PIL images.

    With ``image`` (a PIL photo) the clip starts from it: the photo is cut to each size's aspect
    around ``focus`` (never letterboxed). When the card runs out of memory at ``width`` x ``height``,
    each of ``fallback_sizes`` is tried in turn; the size actually made is the frames' own
    (``frames[0].size``), and the log says which.

    Raises:
        RuntimeError: on any failure — the job reports it; no fallback frames exist.
        InvalidVideoRequest: a photo for a model that cannot start from one.
    """
    model = find(model_name)
    if model.pipeline not in ("wan", "ltx"):
        raise RuntimeError(f"model '{model_name}' is not a diffusers video model")
    from_image = image is not None
    if model.requires_image and not from_image:
        raise InvalidVideoRequest(f"model '{model_name}' starts from a photo: give it a starting image (a keyframe)")
    if from_image and not model.image_to_video:
        raise InvalidVideoRequest(
            f"model '{model_name}' makes video from text only; leave the starting image empty "
            "or choose an image-to-video model (wan2.2-ti2v-5b)"
        )
    torch = _torch()
    wanted = _resident_name(model, from_image)
    pipe = _pipe if _pipe_name == wanted else _load(model, from_image, weights_progress)
    ladder = [(int(width), int(height)), *[(int(w), int(h)) for w, h in fallback_sizes]]
    last_failure: Exception | None = None
    for w, h in ladder:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        started = time.time()
        extra = {"image": crop_to_aspect(image, w, h, focus)} if from_image else {}
        try:
            if model.pipeline == "ltx":
                frames = _run_ltx(
                    pipe,
                    model,
                    prompt,
                    negative_prompt or DEFAULT_NEGATIVE_PROMPT,
                    w,
                    h,
                    num_frames,
                    steps,
                    generator,
                    image,
                    focus,
                    progress_callback,
                )
                print(
                    f"[ltx] {model_name}{' (from a photo)' if from_image else ''} {w}x{h} {len(frames)} frames, {steps} steps, seed {seed}: {time.time() - started:.0f}s",
                    flush=True,
                )
                return list(frames)
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
                width=w,
                height=h,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=model.guidance,
                **({"guidance_scale_2": model.guidance_two} if model.guidance_two is not None else {}),
                generator=generator,
                output_type="pil",
                callback_on_step_end=progress_callback,
                **extra,
            )
        except job_control.JobCancelled:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise  # the owner stopped it: not a failure, and not a smaller size to try
        except Exception as failure:  # §12: report it, never hide it
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            last_failure = failure
            if _is_out_of_memory(failure) and (w, h) != ladder[-1]:
                print(f"[wan] {model_name} {w}x{h}: out of memory, trying the next size", flush=True)
                continue
            raise RuntimeError(f"Wan inference failed ({model_name}): {failure}") from failure
        frames = output.frames[0]
        if not frames:
            raise RuntimeError(f"Wan returned no frames ({model_name})")
        print(
            f"[wan] {model_name}{' (from a photo)' if from_image else ''} {w}x{h} {len(frames)} "
            f"frames, {steps} steps, seed {seed}: {time.time() - started:.0f}s",
            flush=True,
        )
        return list(frames)
    raise RuntimeError(f"Wan inference failed ({model_name}): {last_failure}")
