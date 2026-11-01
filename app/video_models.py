"""The video models this worker runs, read from ``models.yaml`` — and the arithmetic of a clip's shape.

A Studio asks for an **aspect** ("9:16", the vertical short of TikTok, Reels and Shorts) and a
duration; a model knows which sizes it was trained at, how many frames per second it makes and how
many frames it can hold. Turning the first into the second is a pure function of the two, so it lives
here, apart from any pipeline, and is tested without a GPU.

The catalogue (``orazaka_models``) chooses the model; this module only knows how each one is shaped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MODELS_FILE = Path(__file__).resolve().parent.parent / "models.yaml"

#: The aspect a Studio gets when it names none: the vertical short.
DEFAULT_ASPECT = "9:16"

#: What a person may type for an aspect, and what it means. The ratio is the canonical form.
_ASPECT_ALIASES = {
    "9:16": "9:16",
    "9x16": "9:16",
    "vertical": "9:16",
    "portrait": "9:16",
    "16:9": "16:9",
    "16x9": "16:9",
    "horizontal": "16:9",
    "landscape": "16:9",
    "1:1": "1:1",
    "1x1": "1:1",
    "square": "1:1",
}

#: The model a host serves when a job names none and the deployment declared none either: the Mac's.
_FALLBACK_DEFAULT = "animatediff-lightning-mps"


class InvalidVideoRequest(ValueError):
    """A request this worker positively checked and cannot honour — the caller's to fix, not ours."""


class UnsupportedAspect(InvalidVideoRequest):
    """The aspect is not one this platform knows, or not one the chosen model can make."""


class UnknownVideoModel(InvalidVideoRequest):
    """The model name is not in ``models.yaml``: refused by name, never swapped for another."""


@dataclass(frozen=True)
class VideoModel:
    """One video model, as ``models.yaml`` describes it."""

    name: str
    label: str
    pipeline: str
    fps: int
    max_frames: int
    steps: int
    multiple: int
    sizes: dict = field(default_factory=dict)
    guidance: float = 5.0
    #: The low-noise expert's guidance (Wan 2.2 A14B has two experts); None for the single-transformer models.
    guidance_two: float | None = None
    repo: str | None = None
    revision: str | None = None
    text_encoder_from: str | None = None
    vram_gb: int = 0
    ram_gb: int = 0
    hardware: tuple = ()
    #: A model served by a provider rather than run here (``pipeline: remote``): its id there, the provider's
    #: name in the catalogue, and what a clip costs (USD, from the provider's pricing page).
    remote_model: str | None = None
    provider: str | None = None
    cost_usd: float | None = None
    #: Whether the model can start from a photo (``imageToVideo``): Wan 2.2 TI2V 5B only.
    image_to_video: bool = False
    #: Smaller sizes of the same aspect, tried in order when the card runs out of memory at the one
    #: before (``fallbackSizes``): 1280x704 on a 48 GB card, then what a 20 GB card can hold.
    fallback_sizes: dict = field(default_factory=dict)
    #: Wan 2.2 TI2V 5B conditions its first latent frame on the photo (``expandTimesteps``); the A14B I2V does not.
    expand_timesteps: bool = False
    #: A model that only starts from a photo (``requiresImage``: the I2V A14B): a text-only job is refused by name.
    requires_image: bool = False
    #: Where its weights live: ``volume`` (the shared models volume, default) or ``scratch`` (the host's own disk,
    #: fetched at its start: weights too big to keep on a volume whose size is a monthly cost).
    cache: str = "volume"
    #: A distillation LoRA pair (``lora``: repo, revision, ``high`` and ``low`` file for the two experts), or None.
    lora: dict | None = None

    def size_ladder(self, aspect: str | None, width=None, height=None) -> list[tuple[int, int]]:
        """The sizes to try for a request, the one asked first. An explicit size has no ladder."""
        first = self.size_for(aspect, width, height)
        if width and height:
            return [first]
        ratio = normalise_aspect(aspect)
        return [first, *[tuple(size) for size in self.fallback_sizes.get(ratio, ())]]

    def size_for(self, aspect: str | None, width=None, height=None) -> tuple[int, int]:
        """The frame size for an aspect, or for an explicit width and height.

        An explicit size wins (the HTTP API has always accepted one) and is rounded down to the
        model's multiple; otherwise the aspect picks one of the sizes the model was trained at.

        Raises:
            UnsupportedAspect: the aspect is unknown, or the model has no size for it.
        """
        if width and height:
            w = max(self.multiple, (int(width) // self.multiple) * self.multiple)
            h = max(self.multiple, (int(height) // self.multiple) * self.multiple)
            return w, h
        ratio = normalise_aspect(aspect)
        if ratio not in self.sizes:
            raise UnsupportedAspect(f"model '{self.name}' makes {', '.join(sorted(self.sizes))}, not {ratio}")
        w, h = self.sizes[ratio]
        return int(w), int(h)

    def frames_for(self, duration_seconds) -> int:
        """How many frames a clip of this duration needs, within what the model can hold.

        Wan counts frames as ``4k + 1``; other pipelines take any count. Always at least one second.
        The duration actually produced is ``frames / fps`` — that, not the request, is what is billed.
        """
        try:
            seconds = float(duration_seconds) if duration_seconds else 5.0
        except (TypeError, ValueError):
            seconds = 5.0
        seconds = max(1.0, seconds)
        frames = min(self.max_frames, int(round(seconds * self.fps)) + 1)
        if self.pipeline == "wan":
            frames = ((frames - 1) // 4) * 4 + 1
            frames = max(5, min(frames, self.max_frames))
        elif self.pipeline == "ltx":  # LTX counts frames 8k + 1
            frames = ((frames - 1) // 8) * 8 + 1
            frames = max(9, min(frames, self.max_frames))
        return frames


def normalise_aspect(aspect: str | None) -> str:
    """The canonical ratio for what a person typed, ``9:16`` when they typed nothing.

    Raises:
        UnsupportedAspect: for anything that is not a known aspect.
    """
    if aspect is None or str(aspect).strip() == "":
        return DEFAULT_ASPECT
    key = str(aspect).strip().lower().replace(" ", "")
    if key not in _ASPECT_ALIASES:
        known = sorted({value for value in _ASPECT_ALIASES.values()})
        raise UnsupportedAspect(f"unknown aspect '{aspect}' (one of {', '.join(known)})")
    return _ASPECT_ALIASES[key]


def load_manifest(path: Path | None = None) -> dict:
    """The parsed ``models.yaml``."""
    with open(path or MODELS_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def video_models(manifest: dict | None = None) -> dict[str, VideoModel]:
    """Every video model of the manifest, by catalogue name."""
    manifest = manifest or load_manifest()
    models = {}
    for name, spec in (manifest.get("video") or {}).items():
        models[name] = VideoModel(
            name=name,
            label=spec.get("label", name),
            pipeline=spec["pipeline"],
            fps=int(spec["fps"]),
            max_frames=int(spec["maxFrames"]),
            steps=int(spec["steps"]),
            multiple=int(spec["multiple"]),
            sizes={ratio: tuple(size) for ratio, size in (spec.get("sizes") or {}).items()},
            guidance=float(spec.get("guidance", 5.0)),
            guidance_two=float(spec["guidanceTwo"]) if spec.get("guidanceTwo") is not None else None,
            repo=spec.get("repo"),
            revision=spec.get("revision"),
            text_encoder_from=spec.get("textEncoderFrom"),
            vram_gb=int(spec.get("vramGb", 0)),
            ram_gb=int(spec.get("ramGb", 0)),
            hardware=tuple(spec.get("hardware") or ()),
            remote_model=spec.get("remoteModel"),
            provider=spec.get("provider"),
            cost_usd=float(spec["costUsd"]) if spec.get("costUsd") is not None else None,
            image_to_video=bool(spec.get("imageToVideo", False)),
            fallback_sizes={
                ratio: tuple(tuple(size) for size in ladder)
                for ratio, ladder in (spec.get("fallbackSizes") or {}).items()
            },
            expand_timesteps=bool(spec.get("expandTimesteps", False)),
            requires_image=bool(spec.get("requiresImage", False)),
            cache=str(spec.get("cache") or "volume"),
            lora=dict(spec["lora"]) if spec.get("lora") else None,
        )
    return models


def hub_cache(model: VideoModel, env: dict | None = None) -> str | None:
    """The Hugging Face cache a model's weights are read from: ``None`` for the shared volume (``HF_HUB_CACHE``),
    else ``$MEDIA_SCRATCH_DIR/hf/hub`` for a model kept on the host's own disk (``cache: scratch``)."""
    if model.cache != "scratch":
        return None
    source = os.environ if env is None else env
    root = (source.get("MEDIA_SCRATCH_DIR") or "/var/lib/orazaka-scratch").rstrip("/")
    return f"{root}/hf/hub"


def find(name: str, manifest: dict | None = None) -> VideoModel:
    """The model of that name.

    Raises:
        UnknownVideoModel: when ``models.yaml`` does not describe it.
    """
    models = video_models(manifest)
    if name not in models:
        raise UnknownVideoModel(f"model '{name}' is not one this worker runs ({', '.join(sorted(models))})")
    return models[name]


def default_model_name(env: dict | None = None, manifest: dict | None = None) -> str:
    """The model a job gets when neither it nor the catalogue named one.

    Declared by the deployment, in this order: ``MEDIA_VIDEO_DEFAULT_MODEL``, then the default of the
    GPU profile the host was started with (``MEDIA_GPU_PROFILE``, ``big`` or ``small``), then the Mac's.
    """
    source = os.environ if env is None else env
    explicit = (source.get("MEDIA_VIDEO_DEFAULT_MODEL") or "").strip()
    if explicit:
        return explicit
    profile = (source.get("MEDIA_GPU_PROFILE") or "").strip()
    if profile:
        manifest = manifest or load_manifest()
        declared = (manifest.get("profiles") or {}).get(profile)
        if declared and declared.get("defaultVideo"):
            return declared["defaultVideo"]
    return _FALLBACK_DEFAULT


def kept_fraction(image, width: int, height: int) -> float:
    """The share of the photo's pixels that stay in the frame once it is cut to ``width`` x ``height``'s aspect (1.0: none lost).

    A portrait still cut to 16:9 keeps about a third of itself, and image-to-video then amplifies what is left: the Krizaka
    cinematic kit makes the still at the video's own aspect (``image-generation`` 1360x768 or 768x1360) so this stays near 1.
    """
    source_w, source_h = image.size
    target = width / height
    if source_w / source_h > target:
        return round((source_h * target) / source_w, 4)
    return round((source_w / target) / source_h, 4)


def crop_to_aspect(image, width: int, height: int, focus: tuple[float, float] = (0.5, 0.5)):
    """The photo cut to the frame's aspect around its subject, then scaled to ``width`` x ``height``.

    Never a letterbox: what does not fit the frame is cut away, and ``focus`` (x, y in 0..1, the
    centre by default) says where the subject is, so the cut keeps it. Pure: a PIL image in, one out.
    """
    from PIL import Image  # noqa: PLC0415

    source_w, source_h = image.size
    target = width / height
    if source_w / source_h > target:  # too wide: cut the sides
        crop_w, crop_h = round(source_h * target), source_h
    else:  # too tall: cut top and bottom
        crop_w, crop_h = source_w, round(source_w / target)
    fx = min(1.0, max(0.0, float(focus[0])))
    fy = min(1.0, max(0.0, float(focus[1])))
    left = min(source_w - crop_w, max(0, round(fx * source_w - crop_w / 2)))
    top = min(source_h - crop_h, max(0, round(fy * source_h - crop_h / 2)))
    cut = image.convert("RGB").crop((left, top, left + crop_w, top + crop_h))
    return cut.resize((int(width), int(height)), Image.LANCZOS)
