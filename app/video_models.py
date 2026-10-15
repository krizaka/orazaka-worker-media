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
            raise UnsupportedAspect(
                f"model '{self.name}' makes {', '.join(sorted(self.sizes))}, not {ratio}"
            )
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
            repo=spec.get("repo"),
            revision=spec.get("revision"),
            text_encoder_from=spec.get("textEncoderFrom"),
            vram_gb=int(spec.get("vramGb", 0)),
            ram_gb=int(spec.get("ramGb", 0)),
            hardware=tuple(spec.get("hardware") or ()),
            remote_model=spec.get("remoteModel"),
            provider=spec.get("provider"),
            cost_usd=float(spec["costUsd"]) if spec.get("costUsd") is not None else None,
        )
    return models


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
