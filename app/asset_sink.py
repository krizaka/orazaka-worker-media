"""Hands a finished output to the platform's asset store when this worker does not share its disk.

On the Mac the worker writes into the same upload directory the services serve from, and seals the
file there (ADR-054). A GPU droplet has a disk of its own that is destroyed with it: a clip left
there would be gone within the hour, and the platform could never have served it anyway. So when
the deployment declares ``MEDIA_ASSET_SINK_URL`` (the job service, reached through the GPU relay),
the output is sent to ``PUT /internal/v1/jobs/{jobId}/output/{filename}`` instead — the job service
seals it with the platform's key, adopts the job's row when a Studio step left none (so the run's
asset is readable by its owner), and answers the URL the asset is served under.

The droplet therefore never holds the platform's master key: what crosses the tunnel is the
plaintext the worker just made, over TLS, authenticated by a short-lived ``SERVICE`` token.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from app.registration import _service_token

#: An output file name: a word, never a path. The job service checks it again.
_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class AssetSinkError(RuntimeError):
    """The platform did not take the file. The job fails: an output nobody can read is not done."""


def sink_url(env: dict | None = None) -> str | None:
    """The job service's base URL as the deployment declared it, or ``None`` (the Mac)."""
    source = os.environ if env is None else env
    value = (source.get("MEDIA_ASSET_SINK_URL") or "").strip().rstrip("/")
    return value or None


def upload(path: str, job: dict, filename: str, content_type: str, env: dict | None = None) -> str:
    """Sends one output file to the platform and returns the URL it is served under.

    Args:
        path: the plaintext file this worker produced.
        job: the JobCommand — its ``jobId``, ``userId`` (the run's actor, which the job service uses
            only when it has no row of its own for the job), ``featureKey`` and ``dataClass``.
        filename: the name the asset is served under (``video.mp4``).
        content_type: its media type.

    Raises:
        AssetSinkError: no sink is declared, no token can be minted, or the job service refused.
    """
    base = sink_url(env)
    if base is None:
        raise AssetSinkError("MEDIA_ASSET_SINK_URL is not set")
    if not _FILENAME.match(filename):
        raise AssetSinkError(f"refusing to upload under the name {filename!r}")
    token = _service_token()
    if not token:
        raise AssetSinkError("no SERVICE token (ORAZAKA_SERVICE_TOKEN) to authenticate the upload")
    with open(path, "rb") as handle:
        data = handle.read()
    request = urllib.request.Request(  # noqa: S310 — the deployment's own job service, never caller-supplied
        f"{base}/internal/v1/jobs/{job['jobId']}/output/{filename}",
        data=data,
        method="PUT",
        headers={
            "Content-Type": content_type,
            "Authorization": f"Bearer {token}",
            "X-Orazaka-User-Id": str(job.get("userId") or ""),
            "X-Orazaka-Feature-Key": str(job.get("featureKey") or ""),
            "X-Orazaka-Data-Class": str(job.get("dataClass") or ""),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 — see above
            body = json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as refused:
        raise AssetSinkError(f"the job service refused the upload ({refused.code})") from refused
    except OSError as unreachable:
        raise ConnectionError(f"the job service is unreachable for the upload: {unreachable}") from unreachable
    url = body.get("url")
    if not isinstance(url, str) or not url:
        raise AssetSinkError("the job service accepted the upload but returned no URL")
    print(f"[asset-sink] {job['jobId']}/{filename}: {len(data)} bytes stored as {url}", flush=True)
    return url


def fetch_image(job: dict, target: str, env: dict | None = None, asset: str | None = None) -> str:
    """Brings a job's starting image to this host, for a worker that does not share the platform's disk.

    The counterpart of :func:`upload`: ``GET /internal/v1/jobs/{jobId}/input/image`` on the job service
    answers the image the job's archived input names — or, for a Studio step, which reaches the worker
    without being archived, the uploaded ``asset`` it names — in the clear, to the actor the job carries.
    It is written to ``target`` (a scratch file the caller deletes).

    Raises:
        AssetSinkError: no sink or token, or the job service has no such image for that actor.
    """
    base = sink_url(env)
    if base is None:
        raise AssetSinkError("MEDIA_ASSET_SINK_URL is not set")
    token = _service_token()
    if not token:
        raise AssetSinkError("no SERVICE token (ORAZAKA_SERVICE_TOKEN) to fetch the starting image")
    request = urllib.request.Request(  # noqa: S310 — the deployment's own job service, never caller-supplied
        f"{base}/internal/v1/jobs/{job['jobId']}/input/image" + (f"?asset={asset}" if asset else ""),
        method="GET",
        headers={"Authorization": f"Bearer {token}", "X-Orazaka-User-Id": str(job.get("userId") or "")},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 — see above
            data = response.read()
    except urllib.error.HTTPError as refused:
        raise AssetSinkError(f"the job service has no starting image for this job ({refused.code})") from refused
    except OSError as unreachable:
        raise ConnectionError(f"the job service is unreachable for the starting image: {unreachable}") from unreachable
    if not data:
        raise AssetSinkError("the job service answered an empty starting image")
    with open(target, "wb") as handle:
        handle.write(data)
    print(f"[asset-sink] {job['jobId']}: starting image of {len(data)} bytes fetched", flush=True)
    return target


#: An asset URL as the platform serves it: ``/api/v1/assets/<jobId>/<file>``. Two words, never a path.
_ASSET_URL = re.compile(r"^/api/v1/assets/([A-Za-z0-9-]{1,64})/([A-Za-z0-9][A-Za-z0-9._-]{0,99})$")


def parse_asset_url(reference: str):
    """``(jobId, filename)`` of an asset URL, or ``None`` when the reference is anything else."""
    found = _ASSET_URL.match(reference or "")
    return (found.group(1), found.group(2)) if found else None


def fetch(job_id: str, filename: str, user_id: str, destination: str, env: dict | None = None) -> str | None:
    """Reads back one output of the acting user's from the job service into ``destination``.

    The job service answers only for the job's owner (``X-Orazaka-User-Id``); anyone else's is a 404, which
    is returned as ``None`` — *not found*, like a file that does not exist.

    Raises:
        AssetSinkError: no sink or token, or the job service failed.
    """
    base = sink_url(env)
    if base is None:
        raise AssetSinkError("MEDIA_ASSET_SINK_URL is not set")
    token = _service_token()
    if not token:
        raise AssetSinkError("no SERVICE token (ORAZAKA_SERVICE_TOKEN) to read the output")
    request = urllib.request.Request(  # noqa: S310 — the deployment's own job service
        f"{base}/internal/v1/jobs/{job_id}/output/{filename}",
        method="GET",
        headers={"Authorization": f"Bearer {token}", "X-Orazaka-User-Id": str(user_id or "")},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 — see above
            data = response.read()
    except urllib.error.HTTPError as refused:
        if refused.code == 404:
            return None
        raise AssetSinkError(f"the job service refused the read ({refused.code})") from refused
    except OSError as unreachable:
        raise ConnectionError(f"the job service is unreachable for the read: {unreachable}") from unreachable
    os.makedirs(destination, exist_ok=True)
    target = os.path.join(destination, f"{job_id}-{filename}")
    with open(target, "wb") as handle:
        handle.write(data)
    return target
