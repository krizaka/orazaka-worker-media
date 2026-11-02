"""Wan 2.2 A14B on DigitalOcean serverless inference — a video model this worker *routes to* rather than runs.

The clip is made by the provider; Orazaka's part is everything around it: the Studio run, the catalogue
row (``orazaka_models``, provider ``do-inference``), the aspect and the seed, the metering, the sealed
asset and the provenance kept with it. The worker submits the job, polls it, and stores the MP4 like any
other output — so a run of this model is a real Orazaka run, traced by its run id, and honest about where
the pixels came from (``metrics.provider``).

API (docs.digitalocean.com, "Generate video from text"): ``POST /v1/videos`` answers a job id,
``GET /v1/videos/{id}`` its status, ``GET /v1/videos/{id}/content`` the MP4.

Configuration, by bounded context (never a vendor name): ``AI_INFERENCE_URL`` (default
``https://inference.do-ai.run``) and ``AI_INFERENCE_KEY``, a model access key that must list the model.

§12 Zero-Fallback: a refusal, a failed job or a timeout raises. Nothing here returns a placeholder clip.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from app import spend
from app.video_models import VideoModel

DEFAULT_URL = "https://inference.do-ai.run"
#: How long a job may take before the clip is given up on (the provider says 30 s to several minutes).
DEFAULT_TIMEOUT_SECONDS = 900
POLL_SECONDS = 5


class RemoteVideoError(RuntimeError):
    """The provider refused, failed or never answered. The job fails with this reason."""


def base_url(env: dict | None = None) -> str:
    source = os.environ if env is None else env
    return (source.get("AI_INFERENCE_URL") or DEFAULT_URL).strip().rstrip("/")


def api_key(env: dict | None = None) -> str:
    source = os.environ if env is None else env
    key = (source.get("AI_INFERENCE_KEY") or "").strip()
    if not key:
        raise RemoteVideoError("AI_INFERENCE_KEY is not set: this model is served by a provider and needs its key")
    return key


def request_body(model: VideoModel, prompt: str, width: int, height: int, seed: int | None) -> dict:
    """What is sent to the provider: the model's catalogue ``remoteModel`` id, the size, the fps."""
    body = {
        "model": model.remote_model or model.name,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "fps": model.fps,
    }
    if seed is not None and seed >= 0:
        body["seed"] = int(seed)
    return body


def _call(method: str, url: str, key: str, body: dict | None = None, timeout: int = 60) -> tuple[int, bytes]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(  # noqa: S310 — the deployment's configured provider, https
        url,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, response.read()
    except urllib.error.HTTPError as refused:
        detail = refused.read().decode("utf-8", "replace")[:300]
        raise RemoteVideoError(f"the provider answered {refused.code}: {detail}") from refused
    except OSError as unreachable:
        raise RemoteVideoError(f"the provider is unreachable: {unreachable}") from unreachable


def generate(
    model: VideoModel,
    prompt: str,
    width: int,
    height: int,
    seed: int | None,
    progress=None,
    env: dict | None = None,
    sleep=time.sleep,
    clock=time.monotonic,
    call=_call,
) -> tuple[bytes, dict]:
    """Submits, waits and downloads one clip. Returns the MP4 bytes and the facts about the call."""
    key = api_key(env)
    base = base_url(env)
    try:
        spend.reserve(model.cost_usd if model.cost_usd is not None else 0.60, env)
    except spend.SpendRefused as refused:
        raise RemoteVideoError(str(refused)) from refused
    started = clock()
    _, raw = call("POST", f"{base}/v1/videos", key, request_body(model, prompt, width, height, seed))
    job = json.loads(raw.decode("utf-8") or "{}")
    job_id = job.get("id") or job.get("job_id")
    if not job_id:
        raise RemoteVideoError("the provider accepted the request but returned no job id")
    timeout = int((env or os.environ).get("MEDIA_REMOTE_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS)
    status, result = job.get("status"), job
    while status not in ("completed", "succeeded", "failed", "error", "cancelled"):
        if clock() - started > timeout:
            raise RemoteVideoError(f"the provider's job {job_id} did not finish within {timeout}s")
        sleep(POLL_SECONDS)
        _, raw = call("GET", f"{base}/v1/videos/{job_id}", key)
        result = json.loads(raw.decode("utf-8") or "{}")
        status = result.get("status")
        if progress:
            progress(min(95, int((clock() - started) / max(timeout, 1) * 100)))
    if status != "completed" and status != "succeeded":
        raise RemoteVideoError(f"the provider's job {job_id} ended as {status}: {result.get('error')}")
    content_id = result.get("id") or job_id
    output = result.get("output") or {}
    url = output.get("url") if isinstance(output, dict) else None
    if url:  # the documented presigned form; the gateway answers null and serves /content instead
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 — the provider's presigned https URL
            video = response.read()
    else:
        _, video = call("GET", f"{base}/v1/videos/{content_id}/content", key, timeout=300)
    if not video:
        raise RemoteVideoError(f"the provider's job {job_id} returned an empty file")
    facts = {
        "provider": model.provider or "remote",
        "providerJobId": str(content_id),
        "providerSeconds": round(clock() - started, 1),
    }
    if model.cost_usd is not None:
        facts["costUsd"] = model.cost_usd
    return video, facts


def probe_duration(path: str) -> float | None:
    """The length of the clip actually received (ffprobe), the duration that is metered — or ``None``."""
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(  # noqa: S603 — a fixed binary on a file this worker just wrote
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        return round(float(out), 2)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
