"""Measure a model's load and a clip's render, the numbers every claim of #63 is made of.

``python -m app.measure load <model> --mode volume|stage|warm-stage|prefetch`` evicts the model's files from the page cache
(and, for ``stage``, removes the NVMe copy; ``warm-stage`` keeps it: a pod that has served this model before), loads the pipeline the way the worker does and prints one JSON line.
``python -m app.measure render <model> --steps N --seed S --out DIR`` makes one clip (fixed prompt and seed, so before
and after compare) and prints the render seconds. Nothing here changes the worker's behaviour.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ACCELERATIONS = ("fp8", "firstblock")  # applied in this order, whatever order they are asked in

PROMPT = "A red kite flying over a calm lake at sunrise, gentle camera pan, cinematic, soft light"


def evict(root: Path) -> int:
    """Drops a directory's files from the page cache (posix_fadvise DONTNEED); returns the bytes asked to leave."""
    total = 0
    advise = getattr(os, "posix_fadvise", None)  # Linux only: the pod; elsewhere nothing to drop
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        total += path.stat().st_size
        if advise:
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    advise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                pass
    return total


def parse_accel(value: str) -> tuple[str, ...]:
    """``none`` | ``fp8`` | ``firstblock`` | ``fp8+firstblock``: which accelerations a render measures.

    Refused by name when unknown: a measurement of something else than asked is worse than none.
    """
    names = [n for n in (value or "none").lower().split("+") if n and n != "none"]
    for name in names:
        if name not in ACCELERATIONS:
            raise SystemExit(f"unknown acceleration '{name}' (known: none, {', '.join(ACCELERATIONS)})")
    return tuple(a for a in ACCELERATIONS if a in names)


def apply_acceleration(pipe, accel: tuple[str, ...], threshold: float, *, fp8_dtype=None, cache_config=None, compute_dtype=None) -> None:
    """Applies diffusers' own accelerations to the resident Wan transformer (nothing here changes the worker).

    ``fp8``: layerwise casting, the weights kept in float8 (E4M3) and upcast to bf16 per layer, which halves the
    transformer's memory and its read from the volume. ``firstblock``: FirstBlockCache, which skips a step's remaining
    blocks when the first block's output barely moved (``threshold``); it needs no calibration, unlike MagCache.
    """
    if "fp8" in accel:
        pipe.transformer.enable_layerwise_casting(storage_dtype=fp8_dtype, compute_dtype=compute_dtype)
    if "firstblock" in accel:
        pipe.transformer.enable_cache(cache_config(threshold=threshold))


def result_line(mode: str, model: str, seconds: float, size_bytes: int, extra: dict | None = None) -> str:
    gbps = round(size_bytes / 1e9 / seconds, 2) if seconds > 0 else 0.0
    return json.dumps(
        {"mode": mode, "model": model, "seconds": round(seconds, 2), "gb": round(size_bytes / 1e9, 2), "gbPerSecond": gbps, **(extra or {})},
        sort_keys=True,
    )


def _snapshot(model):
    from app import model_cache  # noqa: PLC0415
    from app.video_models import hub_cache  # noqa: PLC0415

    return model_cache.local_snapshot(model.repo, model.revision, hub_cache(model))


def _resolve(module: str, names: tuple[str, ...]) -> None:
    """Resolves the concrete classes of a lazy module (diffusers, transformers import their implementation on first access)."""
    import importlib  # noqa: PLC0415

    loaded = importlib.import_module(module)
    for name in names:
        getattr(loaded, name)


def warm_imports(model=None) -> None:
    """Imports the heavy libraries, and the exact classes the model's loader resolves, before any clock starts: on a pod the venv
    lives on the network volume, and the first import of torch, diffusers and transformers (then of the pipeline classes behind
    their lazy imports) took ~30 s of the first measurement, which is not the weights' load. No weights are loaded here."""
    for module in ("torch", "diffusers", "transformers"):
        _resolve(module, ())
    if model is None:
        return
    if getattr(model, "pipeline", None) == "ltx":
        _resolve("diffusers", ("LTXConditionPipeline",))
        return
    _resolve("diffusers", ("AutoencoderKLWan", "WanImageToVideoPipeline", "WanPipeline"))
    if getattr(model, "text_encoder_from", None):
        _resolve("transformers", ("UMT5EncoderModel",))


def _snapshots_to_stage(model) -> list:
    """The model's snapshot, and its text-encoder donor's when the loader stages that too."""
    from app.video_models import find  # noqa: PLC0415

    models = [model]
    if getattr(model, "text_encoder_from", None):
        models.append(find(model.text_encoder_from))
    return [_snapshot(m) for m in models]


def prepare_stage(model, stage_dir: str) -> None:
    """``warm-stage``: the NVMe copy is made complete BEFORE the clock, so the timed load reads an existing copy.

    A failure stops the measurement (``model_cache.stage`` raises; the loader's own ``resolve`` would fall back silently)."""
    from app import model_cache  # noqa: PLC0415

    for snapshot in _snapshots_to_stage(model):
        model_cache.stage(snapshot, Path(stage_dir) / snapshot.name)


def measure_load(model_name: str, mode: str) -> str:
    from app import wan_generator  # noqa: PLC0415
    from app.video_models import find  # noqa: PLC0415

    model = find(model_name)
    warm_imports(model)

    snapshot = _snapshot(model)
    size = sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())
    stage_dir = os.environ.get("MEDIA_MODELS_STAGE_DIR")
    os.environ.pop("MEDIA_MODELS_PREFETCH", None)
    if mode in ("stage", "warm-stage"):
        if not stage_dir:
            raise SystemExit(f"--mode {mode} needs MEDIA_MODELS_STAGE_DIR")
        if mode == "stage":  # a fresh pod: the copy from the volume is part of the load
            shutil.rmtree(Path(stage_dir) / snapshot.name, ignore_errors=True)
        else:  # a pod that has served this model before: the copy is there, complete, before the timer
            prepare_stage(model, stage_dir)
    elif mode == "prefetch":
        os.environ["MEDIA_MODELS_PREFETCH"] = "1"
        os.environ.pop("MEDIA_MODELS_STAGE_DIR", None)
    else:
        os.environ.pop("MEDIA_MODELS_STAGE_DIR", None)
        stage_dir = None
    evict(snapshot)
    if stage_dir:
        evict(Path(stage_dir))
    started = time.time()
    wan_generator._load(model, from_image=False)
    seconds = time.time() - started
    return result_line(mode, model_name, seconds, size)


def measure_render(model_name: str, steps: int, seed: int, out: Path, accel: tuple[str, ...] = (), threshold: float = 0.2) -> str:
    from app import wan_generator  # noqa: PLC0415
    from app.video_models import find  # noqa: PLC0415

    model = find(model_name)
    width, height = model.sizes["9:16"] if hasattr(model, "sizes") else (480, 832)
    pipe = wan_generator._load(model, from_image=False)
    peak = None
    if accel:
        import torch  # noqa: PLC0415
        from diffusers import FirstBlockCacheConfig  # noqa: PLC0415

        apply_acceleration(pipe, accel, threshold, fp8_dtype=torch.float8_e4m3fn, cache_config=FirstBlockCacheConfig, compute_dtype=torch.bfloat16)
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            peak = True
    except ImportError:
        pass
    started = time.time()
    frames = wan_generator.generate_wan_frames(
        model_name, PROMPT, width, height, model.max_frames, steps, seed
    )
    seconds = time.time() - started
    out.mkdir(parents=True, exist_ok=True)
    label = "+".join(accel) or "none"
    for index in (0, len(frames) // 2, len(frames) - 1):
        frames[index].save(out / f"{model_name}-s{seed}-{label}-f{index}.png")
    extra = {"steps": steps, "frames": len(frames), "seed": seed, "accel": label}
    if accel:
        extra["threshold"] = threshold
    if peak:
        import torch  # noqa: PLC0415

        extra["peakGb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    return result_line("render", model_name, seconds, 0, extra)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.measure")
    sub = parser.add_subparsers(dest="what", required=True)
    load = sub.add_parser("load")
    load.add_argument("model")
    load.add_argument("--mode", choices=["volume", "stage", "warm-stage", "prefetch"], default="volume")
    render = sub.add_parser("render")
    render.add_argument("model")
    render.add_argument("--steps", type=int, default=20)
    render.add_argument("--seed", type=int, default=7)
    render.add_argument("--out", type=Path, default=None, help="output directory (default: a fresh private temp directory)")
    render.add_argument("--accel", default="none", help="none | fp8 | firstblock | fp8+firstblock")
    render.add_argument("--threshold", type=float, default=0.2, help="FirstBlockCache threshold")
    args = parser.parse_args(argv)
    if args.what == "load":
        line = measure_load(args.model, args.mode)
    else:
        out = args.out or Path(tempfile.mkdtemp(prefix="measure-"))
        line = measure_render(args.model, args.steps, args.seed, out, parse_accel(args.accel), args.threshold)
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
