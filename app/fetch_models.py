"""Puts a GPU profile's weights on the models volume, once (``python -m app.fetch_models big``).

A GPU droplet is created and destroyed by the hour; its weights are not. They live on a volume that
outlives every droplet, so this runs at each boot and finds, almost always, that there is nothing to
do. What it downloads is pinned: the Wan repositories at a commit (``revision``), the sd-server files
by size and SHA-256 — the same recipe as the Mac's CLI (``models.yaml``). A file that does not match
is never used: it is written beside its name and renamed only once its checksum is right.

Layout under ``MEDIA_MODELS_DIR`` (``/models``): ``hf/`` is the Hugging Face cache (``HF_HOME``)
the Wan pipelines load from with ``local_files_only``; ``sd/`` holds the sd-server files at the
paths of the recipe.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

from app import model_cache
from app.video_models import load_manifest

#: What a Wan pipeline reads from its repository. The text encoder is skipped when the model borrows
#: its donor's (``textEncoderFrom``), which is the whole point of declaring it.
_WAN_PARTS = ["model_index.json", "scheduler/*", "tokenizer/*", "transformer/*", "transformer_2/*", "vae/*"]


def plan(profile: str, manifest: dict | None = None) -> dict:
    """What a profile needs: ``{"hf": [(repo, revision, patterns, cache)], "files": [file specs]}``.

    ``cache`` is ``volume`` (the shared models volume) or ``scratch`` (the host's own disk, ``cache: scratch``).
    A model's distillation LoRAs (``lora``) are fetched beside it, into the same cache.

    Raises:
        KeyError: for a profile ``models.yaml`` does not declare.
    """
    manifest = manifest or load_manifest()
    declared = manifest["profiles"][profile]
    snapshots: dict[tuple[str, str, str], set[str]] = {}
    for name in declared["video"]:
        spec = manifest["video"][name]
        cache = spec.get("cache") or "volume"
        parts = set(_WAN_PARTS)
        if spec.get("pipeline") == "ltx":  # LTX-Video: one transformer, its own T5 text encoder, no donor
            snapshots.setdefault((spec["repo"], spec["revision"], cache), set()).update(
                {"model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*"}
            )
            continue
        if spec.get("textEncoderFrom"):
            donor = manifest["video"][spec["textEncoderFrom"]]
            snapshots.setdefault((donor["repo"], donor["revision"], donor.get("cache") or "volume"), set()).update(
                {"text_encoder/*", "tokenizer/*"}
            )
        else:
            parts.add("text_encoder/*")
        snapshots.setdefault((spec["repo"], spec["revision"], cache), set()).update(parts)
        lora = spec.get("lora")
        if lora:
            files = {lora["high"], *([lora["low"]] if lora.get("low") else [])}
            snapshots.setdefault((lora["repo"], lora["revision"], cache), set()).update(files)
    # A profile without an image engine (`hero`: an 80 GB card running video only) fetches no image files.
    files = list(manifest["image"][declared["image"]]["files"]) if declared.get("image") else []
    return {
        "hf": [(repo, revision, sorted(patterns), cache) for (repo, revision, cache), patterns in snapshots.items()],
        "files": files,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file(spec: dict, root: Path) -> Path:
    """Downloads one pinned file unless it is already there, verified.

    A file is trusted once its checksum matched (a ``.sha256`` marker beside it records that), so a
    restart re-hashes nothing. Anything else is downloaded to ``.part`` and renamed only when right.

    Raises:
        RuntimeError: when the downloaded bytes do not match the recipe.
    """
    target = root / spec["path"]
    marker = target.with_name(target.name + ".sha256")
    if target.exists() and target.stat().st_size == spec["bytes"] and marker.exists():
        if marker.read_text().strip() == spec["sha256"]:
            return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    started = time.time()
    print(f"[fetch] {spec['path']} ({spec['bytes'] / 1e9:.1f} GB)", flush=True)
    urllib.request.urlretrieve(spec["url"], partial)  # noqa: S310 — a pinned https URL from models.yaml
    actual = _sha256(partial)
    if actual != spec["sha256"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{spec['path']}: checksum {actual} is not the recipe's {spec['sha256']}")
    partial.replace(target)
    marker.write_text(spec["sha256"])
    print(f"[fetch] {spec['path']} verified in {time.time() - started:.0f}s", flush=True)
    return target


def fetch(profile: str, models_dir: Path) -> None:
    """Everything a profile needs, on the volume."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415 — only the GPU image has it

    snapshot_dir = model_cache.local_snapshot  # by layout: the files fetched are only the model's, which the hub calls incomplete

    work = plan(profile)
    hf_home = models_dir / "hf"
    scratch = Path(os.environ.get("MEDIA_SCRATCH_DIR", "/var/lib/orazaka-scratch")) / "hf"
    for repo, revision, patterns, cache in work["hf"]:
        started = time.time()
        print(f"[fetch] {repo}@{revision[:8]} {patterns} → {cache}", flush=True)
        snapshot_download(
            repo_id=repo,
            revision=revision,
            allow_patterns=patterns,
            cache_dir=str((scratch if cache == "scratch" else hf_home) / "hub"),
            max_workers=16,
        )
        print(f"[fetch] {repo} ready in {time.time() - started:.0f}s", flush=True)
        # The model-cache contract: size + SHA-256 per file beside the snapshot (app/model_cache.py).
        snapshot = Path(snapshot_dir(repo, revision, cache_dir=str((scratch if cache == "scratch" else hf_home) / "hub")))
        if model_cache.verify(snapshot):
            model_cache.write_manifest(snapshot)
    for spec in work["files"]:
        fetch_file(spec, models_dir / "sd")


def verify_profile(profile: str, models_dir: Path, deep: bool = False) -> list[str]:
    """``--verify``: every snapshot of the profile against its manifest (sizes, or digests with ``--deep``)."""
    problems = []
    for repo, revision, _patterns, cache in plan(profile)["hf"]:
        hub = str(((Path(os.environ.get("MEDIA_SCRATCH_DIR", "/var/lib/orazaka-scratch")) / "hf") if cache == "scratch" else models_dir / "hf") / "hub")
        try:
            snapshot = model_cache.local_snapshot(repo, revision, hub)
        except FileNotFoundError as absent:
            problems.append(f"{repo}: not on the volume ({absent})")
            continue
        problems += [f"{repo}: {p}" for p in model_cache.verify(Path(snapshot), deep=deep)]
    return problems


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--verify":
        models_dir = Path(os.environ.get("MEDIA_MODELS_DIR", "/models"))
        problems = verify_profile(argv[2], models_dir, deep="--deep" in argv[3:]) if len(argv) >= 3 else ["usage: --verify <profile> [--deep]"]
        print("\n".join(problems) or f"[fetch] profile {argv[2]} verified", flush=True)
        return 1 if problems else 0
    if len(argv) != 2:
        print("usage: python -m app.fetch_models big|small | --verify <profile> [--deep]", file=sys.stderr)
        return 2
    models_dir = Path(os.environ.get("MEDIA_MODELS_DIR", "/models"))
    fetch(argv[1], models_dir)
    print(f"[fetch] profile {argv[1]} complete under {models_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
