"""Writes a finished output straight to the pod's own storage zone (krizaka-internal#2149).

A rented GPU pod dies with its disk and its relay tunnel. The launcher gives it a write-only view of ONE zone
(``GPU_OUT_URL`` = ``<zone>/<day>/<run>``, ``GPU_OUT_HEADER``, ``GPU_OUT_KEY``: orazaka#569) and never a site credential.
When these are set, each clip is put under ``<GPU_OUT_URL>/<pod>/<jobId>-<file>`` before anything else, so the Mac
can promote it whatever happens to the tunnel or the pod afterwards.
"""
from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse
import urllib.request

#: An output file name: a word, never a path.
_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_POD = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class GpuOutError(RuntimeError):
    """The zone did not take the file. The job fails: an output nobody can fetch is not done."""


def configured(env: dict | None = None) -> bool:
    """True when the launcher gave this worker an output zone."""
    source = os.environ if env is None else env
    return bool((source.get("GPU_OUT_URL") or "").strip() and (source.get("GPU_OUT_KEY") or "").strip())


def put(path: str, job: dict, filename: str, content_type: str, env: dict | None = None) -> str | None:
    """Puts one output file into the zone; returns its key in the zone, or ``None`` when no zone is set."""
    source = os.environ if env is None else env
    if not configured(source):
        return None
    if not _FILENAME.match(filename):
        raise GpuOutError(f"refusing to write under the name {filename!r}")
    pod = source.get("RUNPOD_POD_ID") or "unknown"
    if not _POD.match(pod):
        raise GpuOutError("the pod id is not a safe path segment")
    base = source["GPU_OUT_URL"].strip().rstrip("/")
    name = f"{job['jobId']}-{filename}"
    if not _FILENAME.match(name):
        raise GpuOutError("the job id is not a safe path segment")
    with open(path, "rb") as handle:
        data = handle.read()
    request = urllib.request.Request(  # noqa: S310 — the launcher's own https zone URL, never caller-supplied
        f"{base}/{pod}/{name}",
        data=data,
        method="PUT",
        headers={source.get("GPU_OUT_HEADER") or "AccessKey": source["GPU_OUT_KEY"].strip(), "Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=300):  # noqa: S310 — see above
            pass
    except urllib.error.HTTPError as refused:
        raise GpuOutError(f"the output zone refused the write ({refused.code})") from refused
    except OSError as unreachable:
        raise ConnectionError(f"the output zone is unreachable: {unreachable}") from unreachable
    key = urllib.parse.urlparse(base).path.strip("/").split("/", 1)[-1] + f"/{pod}/{name}"
    print(f"[gpu-out] {len(data)} bytes written to {key}", flush=True)
    return key
