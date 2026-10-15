"""The models that make a clip's sound: a music bed from a prompt, foley from the picture.

Kept apart from the mix (``sound.py``) because they are the slow, heavy and optional half: a 15 GB
checkpoint, a GPU or the Mac's Metal, minutes. Each reads ``models.yaml`` (``sound:``) for what it is,
where its weights come from (pinned) and whether the licence of those weights allows commercial use;
a model the file does not name is refused by name, never replaced by another one.

A kind has one default, and it is always commercial. A non-commercial model (MMAudio's weights are
CC-BY-NC) runs only when a job names it.
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from app.video_models import load_manifest

RUNNER = Path(__file__).resolve().parent / "mmaudio_runner.py"


class UnknownSoundModel(ValueError):
    """The model is not one ``models.yaml`` describes for that kind."""


@dataclass(frozen=True)
class SoundModel:
    name: str
    kind: str
    label: str
    licence: str
    commercial: bool
    default: bool
    spec: dict


def sound_models(manifest: dict | None = None) -> dict[str, SoundModel]:
    manifest = manifest or load_manifest()
    found = {}
    for name, spec in (manifest.get("sound") or {}).items():
        found[name] = SoundModel(
            name=name, kind=spec["kind"], label=spec.get("label", name), licence=spec["licence"],
            commercial=bool(spec.get("commercial", False)), default=bool(spec.get("default", False)),
            spec=spec)
    return found


def resolve(kind: str, requested: str | None, manifest: dict | None = None) -> SoundModel | None:
    """The model of ``kind`` a job gets: the one it named, else the kind's commercial default, else none.

    Raises:
        UnknownSoundModel: the name is not a model of that kind in ``models.yaml``.
    """
    models = sound_models(manifest)
    if requested and requested.strip() not in ("", "default"):
        chosen = models.get(requested.strip())
        if chosen is None or chosen.kind != kind:
            names = ", ".join(sorted(n for n, m in models.items() if m.kind == kind))
            raise UnknownSoundModel(f"'{requested}' is not a {kind} model this worker runs ({names})")
        return chosen
    return next((m for m in models.values() if m.kind == kind and m.default), None)


def _device() -> tuple[str, "object"]:
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def generate_music(model: SoundModel, prompt: str, seconds: float, out_wav: str, seed: int | None = None) -> dict:
    """A music bed of about ``seconds`` from ``prompt``, written as a stereo 48 kHz wav.

    The model's shortest piece is ``minSeconds``: a shorter clip still gets that much, and the mix cuts it.
    """
    import soundfile
    import torch
    from diffusers import AceStepPipeline

    spec = model.spec
    seconds = max(float(spec.get("minSeconds", 10)), min(float(seconds), float(spec.get("maxSeconds", 120))))
    seed = int(seed) if seed is not None else int(time.time()) % 2**31
    device, dtype = _device()
    started = time.time()
    pipe = AceStepPipeline.from_pretrained(spec["repo"], revision=spec.get("revision"), torch_dtype=dtype)
    pipe = pipe.to(device)
    loaded = time.time()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    audio = pipe(
        prompt=prompt, lyrics="[Instrumental]", audio_duration=seconds,
        num_inference_steps=int(spec.get("steps", 8)), guidance_scale=float(spec.get("guidance", 7.0)),
        generator=generator,
    ).audios
    wave = audio[0].float().cpu().numpy().T   # (samples, channels)
    soundfile.write(out_wav, wave, int(spec.get("sampleRate", 48000)))
    timings = {"loadSeconds": round(loaded - started, 1), "generateSeconds": round(time.time() - loaded, 1)}
    del pipe, audio
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return {"model": model.name, "seed": seed, "device": device, "audioSeconds": seconds, **timings}


def generate_foley(model: SoundModel, video: str, prompt: str, seconds: float, out_wav: str,
                   seed: int | None = None) -> dict:
    """Foley synchronised to ``video`` (MMAudio), made by its own runner in its own interpreter.

    MMAudio pins a different torch stack than this worker's; ``SOUND_PYTHON`` names the interpreter that
    has it (default ``~/.orazaka/sound/venv/bin/python``), so the two never share an environment.
    """
    python = os.environ.get("SOUND_PYTHON") or os.path.expanduser("~/.orazaka/sound/venv/bin/python")
    if not os.path.isfile(python):
        raise FileNotFoundError(f"the foley interpreter {python} is not installed (SOUND_PYTHON)")
    seed = int(seed) if seed is not None else int(time.time()) % 2**31
    seconds = min(float(seconds), float(model.spec.get("maxSeconds", 8)))
    started = time.time()
    command = [python, str(RUNNER), "--video", video, "--prompt", prompt, "--duration", f"{seconds:.3f}",
               "--seed", str(seed), "--variant", model.spec["variant"], "--out", out_wav]
    # MMAudio downloads its weights (~6 GB) into ./weights and ./ext_weights of its working directory:
    # that is the sound home, never the checkout.
    home = os.environ.get("SOUND_HOME") or os.path.expanduser("~/.orazaka/sound")
    done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
                          cwd=home)
    if done.returncode != 0:
        sys.stderr.write(done.stderr[-2000:])
        raise RuntimeError(f"the foley model failed: {done.stderr.strip().splitlines()[-1] if done.stderr.strip() else done.returncode}")
    return {"model": model.name, "seed": seed, "audioSeconds": seconds,
            "generateSeconds": round(time.time() - started, 1)}
