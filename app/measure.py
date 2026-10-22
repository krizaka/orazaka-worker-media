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
import time
from pathlib import Path

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


def result_line(mode: str, model: str, seconds: float, size_bytes: int, extra: dict | None = None) -> str:
    gbps = round(size_bytes / 1e9 / seconds, 2) if seconds > 0 else 0.0
    return json.dumps(
        {"mode": mode, "model": model, "seconds": round(seconds, 2), "gb": round(size_bytes / 1e9, 2), "gbPerSecond": gbps, **(extra or {})},
        sort_keys=True,
    )


def _snapshot(model):
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    from app.wan_generator import hub_cache  # noqa: PLC0415

    return Path(snapshot_download(model.repo, revision=model.revision, local_files_only=True, cache_dir=hub_cache(model) or None))


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


def measure_render(model_name: str, steps: int, seed: int, out: Path) -> str:
    from app import wan_generator  # noqa: PLC0415
    from app.video_models import find  # noqa: PLC0415

    model = find(model_name)
    width, height = model.sizes["9:16"] if hasattr(model, "sizes") else (480, 832)
    wan_generator._load(model, from_image=False)
    started = time.time()
    frames = wan_generator.generate_wan_frames(
        model_name, PROMPT, width, height, model.max_frames, steps, seed
    )
    seconds = time.time() - started
    out.mkdir(parents=True, exist_ok=True)
    for index in (0, len(frames) // 2, len(frames) - 1):
        frames[index].save(out / f"{model_name}-s{seed}-f{index}.png")
    return result_line("render", model_name, seconds, 0, {"steps": steps, "frames": len(frames), "seed": seed})


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
    render.add_argument("--out", type=Path, default=Path("/tmp/measure"))  # noqa: S108
    args = parser.parse_args(argv)
    line = measure_load(args.model, args.mode) if args.what == "load" else measure_render(args.model, args.steps, args.seed, args.out)
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
