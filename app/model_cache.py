"""The model-cache contract (orazaka-worker-media#63).

A model lives in one directory (the Hugging Face snapshot ``$MEDIA_MODELS_DIR/hf/hub/models--<org>--<name>/snapshots/<revision>``
or an sd-server folder). Beside its files, ``.manifest.json`` records the size and SHA-256 of each one:

* ``verify`` checks the sizes (fast, every boot) or, ``deep``, the digests (after a download, or on demand);
* ``stage`` copies a verified snapshot to local NVMe (``MEDIA_MODELS_STAGE_DIR``) with parallel readers, reporting the
  bytes read (the ``STREAMING_WEIGHTS_NVME`` percentage of ``app.stages``); a staged copy that verifies is never copied
  again, a partial one is repaired; the network volume stays the source of truth;
* ``resolve`` is what a loader calls: the staged copy when a stage dir is set and writable, else the source, never an error;
* ``prefetch`` reads the files with parallel readers into the page cache, so ``from_pretrained``'s mmap then runs at RAM speed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MANIFEST = ".manifest.json"
CHUNK = 64 * 1024 * 1024
WORKERS = 8


def local_snapshot(repo: str, revision: str, cache_dir: str | None = None) -> Path:
    """The directory of a snapshot already on disk, found by the hub's layout (``models--<org>--<name>/snapshots/<revision>``).

    Not ``snapshot_download(local_files_only=True)``: the worker fetches only the files a model needs, which that call reports
    as an incomplete snapshot.

    Raises:
        FileNotFoundError: the snapshot is not under the cache.
    """
    root = Path(
        cache_dir
        or os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    )
    snapshot = root / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"{repo}@{revision} is not under {root}")
    return snapshot


def _files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != MANIFEST and not path.name.endswith(".part"):
            yield path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path, digests: bool = True) -> dict:
    """Records size and SHA-256 per file; ``digests=False`` records sizes only (no hashing of 30 GB on a billed GPU)."""
    root = Path(root)
    manifest = {
        "files": {
            str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": _sha256(p) if digests else ""}
            for p in _files(root)
        }
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def _read_manifest(root: Path):
    try:
        return json.loads((Path(root) / MANIFEST).read_text())
    except (OSError, ValueError):
        return None


def verify(root: Path, deep: bool = False) -> list[str]:
    """The problems found (empty when the snapshot is exactly what its manifest says)."""
    root = Path(root)
    manifest = _read_manifest(root)
    if manifest is None:
        return ["no manifest"]
    problems = []
    for rel, want in sorted(manifest["files"].items()):
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: missing")
        elif path.stat().st_size != want["bytes"]:
            problems.append(f"{rel}: size {path.stat().st_size}, expected {want['bytes']}")
        elif deep and want["sha256"] and _sha256(path) != want["sha256"]:
            problems.append(f"{rel}: digest differs")
    return problems


class _Counter:
    def __init__(self, total, progress):
        self.total, self.read, self._progress, self._lock = total, 0, progress, threading.Lock()

    def add(self, n):
        with self._lock:
            self.read += n
            if self._progress:
                self._progress(self.read, self.total)  # under the lock: the reported bytes never go down


def _copy_one(src: Path, dst: Path, chunk: int, counter: _Counter) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    with open(src, "rb") as fin, open(part, "wb") as fout:
        while block := fin.read(chunk):
            fout.write(block)
            counter.add(len(block))
    part.replace(dst)


def stage(src: Path, dst: Path, workers: int = WORKERS, chunk: int = CHUNK, progress=None) -> Path:
    """Copies ``src`` (a verified snapshot) to ``dst`` with parallel readers; returns ``dst``."""
    src, dst = Path(src), Path(dst)
    manifest = _read_manifest(src) or write_manifest(src, digests=False)
    total = sum(f["bytes"] for f in manifest["files"].values())
    if _read_manifest(dst) == manifest and not verify(dst):
        if progress:
            progress(total, total)
        return dst
    counter = _Counter(total, progress)
    todo = []
    for rel, want in manifest["files"].items():
        target = dst / rel
        if target.is_file() and target.stat().st_size == want["bytes"]:
            counter.add(want["bytes"])  # already there from an earlier, interrupted staging
        else:
            todo.append(rel)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(_copy_one, src / rel, dst / rel, chunk, counter) for rel in todo]:
            future.result()
    shutil.copyfile(src / MANIFEST, dst / MANIFEST)
    return dst


def resolve(src: Path, stage_dir, progress=None) -> Path:
    """The directory a loader should read: the NVMe copy when ``stage_dir`` is set and usable, else ``src``."""
    if not stage_dir:
        return Path(src)
    try:
        return stage(Path(src), Path(stage_dir) / Path(src).name, progress=progress)
    except OSError as unusable:
        print(f"[model-cache] staging unavailable ({unusable}); reading {src}", flush=True)
        return Path(src)


def prefetch(root: Path, workers: int = WORKERS, chunk: int = CHUNK, progress=None) -> None:
    """Reads every file with parallel readers (page cache warm-up for a following mmap load)."""
    paths = list(_files(Path(root)))
    counter = _Counter(sum(p.stat().st_size for p in paths), progress)

    def read(path):
        with open(path, "rb") as handle:
            while block := handle.read(chunk):
                counter.add(len(block))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(read, p) for p in paths]:
            future.result()


def timed_load(name: str, size_bytes: float, load, clock=time.time, log=print):
    """Runs ``load()`` and logs how long it took and at what rate: the number every claim here is measured with."""
    started = clock()
    result = load()
    seconds = clock() - started
    rate = size_bytes / 1e9 / seconds if seconds > 0 else 0.0
    log(f"[model-cache] {name}: {size_bytes / 1e9:.1f} GB in {seconds:.1f}s ({rate:.1f} GB/s)")
    return result, seconds
