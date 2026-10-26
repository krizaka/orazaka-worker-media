"""Measure a model's load and a clip's render, the numbers every claim of #63 is made of.

``python -m app.measure load <model> --mode volume|stage|prefetch`` evicts the model's files from the page cache
(and, for ``stage``, removes the NVMe copy), loads the pipeline the way the worker does and prints one JSON line.
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


def measure_load(model_name: str, mode: str) -> str:
    from app import wan_generator  # noqa: PLC0415
    from app.video_models import find  # noqa: PLC0415

    model = find(model_name)
    snapshot = _snapshot(model)
    size = sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())
    stage_dir = os.environ.get("MEDIA_MODELS_STAGE_DIR")
    os.environ.pop("MEDIA_MODELS_PREFETCH", None)
    if mode == "stage":
        if not stage_dir:
            raise SystemExit("--mode stage needs MEDIA_MODELS_STAGE_DIR")
        shutil.rmtree(Path(stage_dir) / snapshot.name, ignore_errors=True)
    elif mode == "prefetch":
        os.environ["MEDIA_MODELS_PREFETCH"] = "1"
        os.environ.pop("MEDIA_MODELS_STAGE_DIR", None)
    else:
        os.environ.pop("MEDIA_MODELS_STAGE_DIR", None)
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
    load.add_argument("--mode", choices=["volume", "stage", "prefetch"], default="volume")
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
