#!/usr/bin/env python3
"""AMQP entrypoint of the media worker (strangler-fig Phase 3b).

Consumes video generation commands from the platform topology instead of being
called over HTTP by the router: queue ``orazaka.jobs.video`` bound to
``job.video.*`` on the ``orazaka.jobs`` topic exchange (AGENTS.md §6), one
message at a time (heavy inference is the backpressure). Terminal outcomes are
reported as ``job.{jobId}.done|error`` events on ``orazaka.events`` — the
router's event listener applies them to the job row, so this worker needs no
database.

The generation itself still runs through the local HTTP pipeline (``app.main``
served in a background thread and POSTed on localhost): the AMQP boundary flips
now, the internal pipeline refactor can come later without touching the
contract. Synchronous callers (the core's video client) keep working against
the same HTTP port.

**This worker knows its bindings and nothing else** (ADR-037 §4.2, P4). It used
to compare the message's ``featureKey`` against a capability name it recognised,
which made a general-purpose media worker coupled to one pack: a fourth media
capability meant editing Python. What it does now is read the routing key the
broker delivered under — which the dispatcher took from the capability's row —
so a new capability bound to ``job.compose.*`` works here with no change, and a
capability this worker was never bound to never arrives.
"""

import functools
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request

import pika

from app import (
    activity,
    asset_sink,
    broker,
    envelope,
    gpu_out,
    job_control,
    main,
    sound,
    sound_generators,
    stages,
    video_models,
)
from app import cut as film
from app.composer import compose
from app.registration import load_declaration, start_background_registration
from app.telemetry import (
    EXECUTOR_FAULT,
    INPUT_INVALID,
    PLATFORM_UNAVAILABLE,
    extract_consumption,
    send_job_cancelled,
    send_job_done,
    send_job_error,
    send_progress_update,
)

JOBS_EXCHANGE = "orazaka.jobs"
DLX_EXCHANGE = "orazaka.dlx"


def hosted_only(env: dict | None = None) -> bool:
    """This process runs only the models a provider serves (``MEDIA_WORKER_HOSTED_ONLY=true``): a cheap CPU consumer.

    It calls the provider's API (app/remote_video.py) and needs no GPU, no torch and no weights, so dev keeps it
    always on beside the JVMs; the GPU worker (a pod, the Mac) stays off until a film needs it. It drains a queue of
    its own — hosted jobs are published on ``job.hostedvideo.*``, which the GPU worker's ``job.video.*`` never
    matches — so a GPU job is never taken by a worker that cannot run it, nor a hosted one left behind a GPU that is off.
    """
    return str((os.environ if env is None else env).get("MEDIA_WORKER_HOSTED_ONLY", "")).strip().lower() == "true"


HOSTED_BINDINGS = ["job.hostedvideo.*"]
HOSTED_WORKER_NAME = "orazaka-worker-media-hosted"


def video_queue(env: dict | None = None) -> str:
    return "orazaka.jobs.video.hosted" if hosted_only(env) else "orazaka.jobs.video"


def apply_hosted(declaration: dict, env: dict | None = None) -> dict:
    """The declaration this process registers: the worker.yaml's, or the hosted-only worker's own (name and bindings)."""
    if not hosted_only(env):
        return declaration
    return {**declaration, "name": HOSTED_WORKER_NAME, "family": "media-hosted", "bindings": list(HOSTED_BINDINGS)}


def hosted_refusal(model_name: str | None, env: dict | None = None, find=None) -> str | None:
    """Why this process will not run the model: the hosted-only worker has no GPU, so a model run here is refused by name."""
    if not hosted_only(env):
        return None
    if not model_name:
        return "this worker runs hosted models only and needs one named: leave « automatic » for the GPU worker"
    find = find or video_models.find
    try:
        remote = getattr(find(model_name), "remote_model", None)
    except Exception:  # an unknown model: the pipeline refuses it by name, as everywhere
        return None
    if remote:
        return None
    return f"model '{model_name}' runs on a GPU and this worker runs hosted models only: it needs the GPU worker"


VIDEO_QUEUE = video_queue()
VIDEO_DLQ = VIDEO_QUEUE + ".dlq"

# What this worker drains is declared in worker.yaml and read at boot (ADR-038, S3).
# It used to be VIDEO_BINDING / COMPOSE_BINDING literals right here, which meant adding a
# binding was a Python edit; now the declaration is one file, used both to bind the queues
# and to register with the platform.
DECLARATION = apply_hosted(load_declaration())

# Studio media composition (ADR-034 §7.2) shares this queue: it is ffmpeg work on this host's
# accelerator, so it shares the worker's serialised budget rather than competing with generation
# from a second consumer. Which binding means "compose" is still a property of the ROUTING KEY the
# broker delivered under — never of a capability name (P4).
COMPOSE_PREFIX = "job.compose."
# Sound for a video (orazaka-packs#25) is ffmpeg + a local model (MPS/CUDA weights of 15 GB the other hosts
# of this worker do not have), so it has a queue of its own, consumed only where MEDIA_SOUND_ENABLED=true: a
# GPU droplet that drains `orazaka.jobs.video` never takes a sound job it cannot run. Told apart by the key.
SOUND_QUEUE = "orazaka.jobs.sound"
SOUND_PREFIX_BINDING = "job.sound."
_SERIAL = threading.Lock()  # one job at a time on this host, whichever queue it came from


def sound_enabled(env: dict | None = None) -> bool:
    return str((os.environ if env is None else env).get("MEDIA_SOUND_ENABLED", "")).strip().lower() == "true"


def split_bindings(declared: list, enabled: bool) -> tuple:
    """The declaration's bindings, split by queue: (the shared queue's, the sound queue's).

    The sound bindings are DECLARED (the platform must know someone drains them) but bound, consumed and
    registered only where this host enabled sound: elsewhere they are left out of both.
    """
    sound = [b for b in declared if str(b).startswith(SOUND_PREFIX_BINDING)]
    shared = [b for b in declared if b not in sound]
    return shared, (sound if enabled else [])


def sound_only(env: dict | None = None) -> bool:
    """This process drains the sound queue and nothing else (``MEDIA_WORKER_ONLY_SOUND=true``).

    The Mac joins the shared dev queue to add sound; without this it would also take the video jobs of
    whoever else is using the platform and run them on its own model.
    """
    return str((os.environ if env is None else env).get("MEDIA_WORKER_ONLY_SOUND", "")).strip().lower() == "true"


BINDINGS, SOUND_BINDINGS = split_bindings(DECLARATION["bindings"], sound_enabled() and not hosted_only())
DECLARATION = {**DECLARATION, "bindings": BINDINGS + SOUND_BINDINGS}

SOUND_PREFIX = "job.sound."


def _http_port() -> int:
    try:
        return int(os.environ.get("MEDIA_WORKER_PORT", os.environ.get("VIDEO_WORKER_PORT", "8188")))
    except ValueError:
        return 8188


def _upload_root() -> str:
    """Mirror of app.main._upload_roots: first candidate is the canonical root."""
    return main._upload_roots()[0]


_KEYRING = None


def _keyring():
    """The master keyring, loaded once. The same file, and the same format, as the Java side."""
    global _KEYRING  # noqa: PLW0603 — one process, one keyring, loaded lazily so a worker that
    # never writes an asset does not require one to start.
    if _KEYRING is None:
        path = os.environ.get(
            "ORAZAKA_ASSETS_MASTER_KEY_FILE",
            os.path.join(os.path.expanduser("~"), ".orazaka", "master.key"),
        )
        _KEYRING = envelope.MasterKeyring(path)
    return _KEYRING


def _materialise(path: str, scratch: str) -> str:
    """A readable copy of an asset for a tool that can only open a file.

    ffmpeg and Pillow take paths, not streams, so an encrypted input has to exist in the clear
    somewhere for exactly as long as they read it. That somewhere is a scratch directory this
    worker owns and deletes, never the store.
    """
    if not envelope.is_encrypted(path):
        return path
    plain = envelope.decrypt_bytes(path, _keyring())
    target = os.path.join(scratch, os.path.basename(path))
    with open(target, "wb") as handle:
        handle.write(plain)
    return target


def _seal(path: str) -> None:
    """Encrypts a file this worker just produced, unless encryption is off for a migration."""
    if os.environ.get("ORAZAKA_ASSETS_ENCRYPTION_ENABLED", "true").lower() != "true":
        print(f"[consumer] encryption disabled: {path} stays in the clear", flush=True)
        return
    envelope.encrypt_in_place(path, _keyring())


def _store(path: str, job: dict, filename: str, content_type: str, local_url: str) -> str:
    """Keeps an output where its owner can read it, and returns the URL it is served under.

    Where the worker shares the platform's upload directory (the Mac), the file is sealed in place
    (ADR-054). Where it does not (a GPU droplet, ``MEDIA_ASSET_SINK_URL``), the file is handed to the
    job service, which seals it with the platform's key, and the local copy is deleted: the droplet
    keeps nothing once the job is done.
    """
    if asset_sink.sink_url() is None:
        _seal(path)
        return local_url
    try:
        # The pod's own zone first: it outlives the tunnel and the pod (krizaka-internal#2149).
        gpu_out.put(path, job, filename, content_type)
        return asset_sink.upload(path, job, filename, content_type)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _output_paths(user_id: str, job_id: str) -> tuple:
    """Absolute output file path + the public URL the router serves it under."""
    out_dir = os.path.join(_upload_root(), user_id, job_id, "output")
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, "video.mp4")
    public_url = f"/uploads/{user_id}/{job_id}/output/video.mp4"
    return file_path, public_url


# The model a Studio step leaves unresolved: its payload carries "" (``{{inputs.model:}}``) and the
# job "default" (JobCommand's sentinel). Neither names a pipeline: sent on, the HTTP pipeline answered
# 400 "model is not registered" and every video-generation run failed.
_UNRESOLVED_MODELS = ("", "default")


def _requested_model(payload: dict, job: dict):
    """The model the caller asked for — the payload's, then the job's — or None to let the pipeline pick."""
    for candidate in (payload.get("model"), job.get("model")):
        if isinstance(candidate, str) and candidate.strip() not in _UNRESOLVED_MODELS:
            return candidate.strip()
    return None


# An uploaded asset's id: a UUID, and nothing else is mistaken for one.
_ASSET_ID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def _starting_image(job: dict, payload: dict, scratch: str):
    """A readable file of the photo a clip starts from, or ``None`` when the job names none.

    The job service resolved the Studio's asset id to a file of the run's actor (``imagePath``, and
    ``image`` rewritten to the same path). Where this worker shares that disk (the Mac) the file is
    opened — decrypted into ``scratch`` when it is sealed. Where it does not (a GPU droplet) it is
    fetched from the job service over the relay. A blank value is the Studio's empty input.
    """
    named = next(
        (str(payload[key]).strip() for key in ("imagePath", "image") if str(payload.get(key) or "").strip()),
        "",
    )
    if not named:
        return None
    # A Studio step is dispatched straight to this worker: it names the uploaded asset by id, which
    # nobody resolved. A job the platform's listener archived carries the resolved path instead.
    asset = named if _ASSET_ID.fullmatch(named) else None
    if asset_sink.sink_url() is not None:
        return asset_sink.fetch_image(job, os.path.join(scratch, "starting-image"), asset=asset)
    if asset:
        found = _resolve_owned_asset(_upload_root(), str(job.get("userId") or ""), asset)
        if found:
            return _materialise(found, scratch)
    elif os.path.isabs(named) and os.path.isfile(named):
        return _materialise(named, scratch)
    # Named but not there: a run that quietly ignored its photo is not the run that was asked for (§12).
    raise InvalidJobPayload("the starting image the job names is not readable on this host")


def _generate(job: dict) -> tuple:
    """Runs the generation through the local HTTP pipeline, with a scratch folder for its input photo.

    Returns the job result and, separately, the consumption measurements the
    billing service settles the hold against.
    """
    scratch = tempfile.mkdtemp(prefix="orazaka-input-")
    try:
        return _generate_with(job, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _generate_with(job: dict, scratch: str) -> tuple:
    job_id = job["jobId"]
    payload = job.get("payload") or {}
    prompt = payload.get("prompt") or payload.get("text")
    if not prompt:
        raise ValueError("Payload does not contain prompt or text field")
    try:
        # Checked before a GPU second is spent: an aspect nobody can make is the caller's to fix.
        video_models.normalise_aspect(payload.get("aspect"))
    except video_models.UnsupportedAspect as bad:
        raise InvalidJobPayload(str(bad)) from bad

    output_path, public_url = _output_paths(job["userId"], job_id)
    request_body = dict(payload)
    request_body["prompt"] = prompt
    request_body["job_id"] = job_id
    request_body["output_path"] = output_path
    photo = _starting_image(job, payload, scratch)
    for key in ("image", "imagePath", "image_path"):
        request_body.pop(key, None)  # the pipeline reads one key, and only a file this host can open
    if photo:
        request_body["image_path"] = photo
    model = _requested_model(payload, job)
    refusal = hosted_refusal(model)
    if refusal:
        raise InvalidJobPayload(refusal)
    if model:
        request_body["model"] = model
    else:
        request_body.pop("model", None)

    request = urllib.request.Request(
        f"http://127.0.0.1:{_http_port()}/v1/videos/generations",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:  # noqa: S310 — fixed loopback target (127.0.0.1 + configured port), never a caller-supplied URL
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as refused:
        if refused.code == main.CANCELLED_STATUS:
            raise job_control.JobCancelled(f"job {job_id} was cancelled by its owner") from refused
        if refused.code == 400:
            # The pipeline positively checked the request and refused it (unknown model, an aspect
            # the model cannot make, a starting image for a text-to-video model): the caller's.
            raise InvalidJobPayload(refused.read().decode("utf-8", "replace")[:300]) from refused
        raise

    # The generator wrote plaintext through ffmpeg; seal it before anything else can read it
    # (ADR-054). In place and atomically, so a kill here leaves the plaintext or the envelope and
    # never a half-sealed file — or, on a host that does not share the store, hand it over.
    send_progress_update(job_id, 100, phase="uploading")
    url = _store(output_path, job, "video.mp4", "video/mp4", public_url)
    result = {"url": url, "format": "mp4"}
    metrics = body.get("metrics")
    if isinstance(metrics, dict) and metrics:
        result["metrics"] = metrics
    return result, extract_consumption(metrics)


def _resolve_owned_asset(upload_root: str, user_id: str, candidate: str):
    """Resolves one asset id to a file **inside its owner's directory, and nowhere else**.

    The Python twin of the job service's ``AssetFileResolver``. It has to exist
    here because ``orazaka.studio.media.compose`` routes to ``job.compose.assemble``
    — this worker — so a job on that binding never passes through the Java
    listener that resolves assets for every other capability (ADR-042, ADR-046).

    Ownership is the path, deliberately: an asset id inside a blueprint is user
    input, and treating it as a filename to look up anywhere would let one actor's
    run read another's uploads. An id belonging to somebody else is not "denied",
    it is *not found* — the same answer as one that never existed, which leaks
    nothing about which it was.

    :param upload_root: the root upload directory
    :param user_id: the acting user — the run's actor, never a payload value
    :param candidate: an asset id — never a path
    :returns: the absolute path, or ``None`` when this user has no such asset
    """
    if not candidate or not user_id:
        return None
    # An absolute path is not an asset reference, and is refused rather than checked (ADR-065).
    # This used to return it untouched, "already resolved upstream" — but the upstream it trusted,
    # the job service's resolver, never sees a compose job, which routes straight here; a run input
    # naming another actor's upload by path was composed into the caller's reel. Checking the path
    # against the owner's root would have kept a way of asking for files by location that nothing
    # needs. Nothing legitimate sends one: every producer of this binding passes asset ids.
    if os.path.isabs(candidate):
        return None

    owner_root = os.path.realpath(os.path.join(upload_root, user_id))
    prefix = os.path.basename(candidate)
    # basename() first: a "../" in an id cannot survive it, so traversal is gone
    # before any lookup rather than caught after one.
    if not prefix or prefix != candidate:
        return None

    for directory in (os.path.join(owner_root, "temp"), owner_root):
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.startswith(prefix):
                continue
            found = os.path.realpath(os.path.join(directory, name))
            # A regular file, never a directory: output lives under
            # {root}/{user}/{jobId}/, so a job id would otherwise resolve to it.
            if os.path.isfile(found) and found.startswith(owner_root + os.sep):
                return found
    return None


def _as_list(value) -> list:
    """A list input as the interpreter renders it: a list, a JSON array, or a comma list."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def _cut(job: dict) -> tuple:
    """The cut of a cinematic short (ADR-080): the actor's own clips and an edit decision list → one film.

    ``clips`` are what earlier Studio steps returned (``/api/v1/assets/<job>/<file>``, or an asset id), resolved
    with the same ownership rule as a sound's video: another actor's clip is *not found*. ``edit`` is checked by
    ``app.cut.parse_edit`` before a frame is touched; an edit that cannot be made is the caller's to fix.
    """
    payload = job.get("payload") or {}
    references = _as_list(payload.get("clips"))
    if not references:
        raise InvalidJobPayload("the cut has no clips")
    try:
        edit = film.parse_edit(payload.get("edit") or "{}", len(references))
    except film.EditError as bad:
        raise InvalidJobPayload(str(bad)) from bad
    upload_root = _upload_root()
    user_id = job.get("userId")
    output_path, public_url = _output_paths(job["userId"], job["jobId"])
    with tempfile.TemporaryDirectory(prefix="orz-cut-in-") as scratch:
        readable = []
        for i, reference in enumerate(references):
            if _ASSET_ID.fullmatch(reference) and asset_sink.sink_url() is not None:
                # An uploaded clip (an asset id), on a host that does not share the platform's disk: the job service
                # hands back the actor's own upload, as it does a video's starting image.
                try:
                    found = asset_sink.fetch_image(job, os.path.join(scratch, f"clip-{i:02d}.mp4"), asset=reference)
                except asset_sink.AssetSinkError:
                    found = None
            else:
                found = _resolve_owned_output(upload_root, user_id, reference, scratch)
            if not found:
                raise InvalidJobPayload(f"clip {i} was not found among your files")
            readable.append(_materialise(found, scratch))
        send_progress_update(job["jobId"], 10, phase="cutting")
        metrics = film.cut(readable, edit, output_path, provenance={"clips": len(readable)})
    url = _store(output_path, job, "video.mp4", "video/mp4", public_url)
    result = {"url": url, "assetId": url, "format": "mp4", "metrics": metrics}
    return result, extract_consumption(metrics)


def _compose(job: dict) -> tuple:
    """Assembles a Studio run's stills (and optional voiceover) into a vertical MP4 — or, given ``clips``, cuts a film.

    The payload arrives with every template already rendered by the interpreter,
    so this reads plain values and never interprets a blueprint.
    """
    job_id = job["jobId"]
    payload = job.get("payload") or {}
    if payload.get("clips"):
        return _cut(job)

    photos = payload.get("photos")
    if isinstance(photos, str):
        photos = [item.strip() for item in photos.split(",") if item.strip()]
    if not isinstance(photos, list) or not photos:
        raise ValueError("compose payload does not contain a photos list")

    upload_root = _upload_root()
    user_id = job.get("userId")
    # Asset ids, not paths: the blueprint names what the actor uploaded, and only
    # that actor's own files may answer (ADR-046). An id that resolves to nothing
    # is dropped here rather than passed on as a path that cannot exist, so the
    # failure names the missing asset instead of the composer's empty input.
    resolved = [path for path in (_resolve_owned_asset(upload_root, user_id, str(item)) for item in photos) if path]

    audio = payload.get("audio") or None
    if audio:
        audio = _resolve_owned_asset(upload_root, user_id, str(audio))

    output_path, public_url = _output_paths(job["userId"], job_id)
    # Inputs come out of the store encrypted; ffmpeg and Pillow read paths, so they are
    # materialised into a scratch directory this worker owns and removes (ADR-054 §4).
    with tempfile.TemporaryDirectory(prefix="orz-plain-") as scratch:
        readable = [_materialise(path, scratch) for path in resolved]
        readable_audio = _materialise(audio, scratch) if audio else None
        metrics = compose(readable, output_path, audio_path=readable_audio)

    url = _store(output_path, job, "video.mp4", "video/mp4", public_url)
    result = {"url": url, "assetId": url, "format": "mp4", "metrics": metrics}
    return result, extract_consumption(metrics)


#: The assembly engine, as the pricebook knows it. A name for what this branch RUNS —
#: not a capability key and not a pack key, so [PACK-002] is untouched.
COMPOSE_ENGINE = "orazaka-compose"


def _is_compose(routing_key: str) -> bool:
    """Whether the broker delivered this message under one of the compose bindings.

    The routing key is the dispatcher's decision, read from the capability's
    ``routing_key`` column — not a name this worker recognises. That distinction
    is the whole of P4: the worker declares which keys it drains and stays
    ignorant of which pack asked.
    """
    return bool(routing_key) and routing_key.startswith(COMPOSE_PREFIX)


#: Hosts a clip may be read from by https URL: the platform's own media CDN, not an arbitrary address (no SSRF).
#: ``MEDIA_SOUND_INPUT_HOSTS`` (comma list) overrides.
DEFAULT_INPUT_HOSTS = ("krizaka-assets.b-cdn.net",)
MAX_INPUT_BYTES = 256 * 1024 * 1024


def _input_hosts(env: dict | None = None) -> tuple:
    raw = (os.environ if env is None else env).get("MEDIA_SOUND_INPUT_HOSTS", "")
    hosts = tuple(h.strip().lower() for h in str(raw).split(",") if h.strip())
    return hosts or DEFAULT_INPUT_HOSTS


def _fetch_published(reference: str, scratch: str, env: dict | None = None):
    """A published clip of the platform's media CDN (``https://<cdn host>/…``), saved into ``scratch``.

    Only https, only the allow-listed hosts, bounded in size; anything else is *not found*.
    """
    from urllib.parse import urlparse

    parts = urlparse(reference)
    if parts.scheme != "https" or (parts.hostname or "").lower() not in _input_hosts(env) or parts.username:
        return None
    name = os.path.basename(parts.path) or "clip.mp4"
    target = os.path.join(scratch, "published-" + "".join(c for c in name if c.isalnum() or c in "._-"))
    try:
        with urllib.request.urlopen(reference, timeout=120) as response:  # noqa: S310 — https + allow-listed host only
            data = response.read(MAX_INPUT_BYTES + 1)
    except OSError:
        return None
    if not data or len(data) > MAX_INPUT_BYTES:
        return None
    with open(target, "wb") as handle:
        handle.write(data)
    return target


def _resolve_owned_output(upload_root: str, user_id: str, reference: str, scratch: str | None = None):
    """A file this actor's earlier step produced, named by the URL that step returned.

    A Studio chains steps with ``{{steps.video.url}}``, which is ``/api/v1/assets/<job>/<file>`` (or, for a
    worker sharing the platform's disk, ``/uploads/<user>/<job>/output/<file>``) — not an asset id.
    Ownership is still the path or the job's row: the file must resolve, after symlinks, inside the acting
    user's own directory, or the job service must say the actor owns that job. Another actor's output is
    *not found*, the same answer as one that never existed. Anything else falls back to an asset id.
    """
    if not reference or not user_id:
        return None
    if reference.startswith("https://"):
        return _fetch_published(reference, scratch or tempfile.mkdtemp(prefix="orz-in-"))
    owner_root = os.path.realpath(os.path.join(upload_root, user_id))
    asset = asset_sink.parse_asset_url(reference)
    if asset:
        job_id, filename = asset
        if asset_sink.sink_url() is not None:
            # This host does not share the platform's disk: the job service hands back what the actor owns.
            return asset_sink.fetch(job_id, filename, user_id, scratch or tempfile.mkdtemp(prefix="orz-in-"))
        found = os.path.realpath(os.path.join(owner_root, job_id, "output", filename))
    elif reference.startswith("/uploads/"):
        found = os.path.realpath(os.path.join(upload_root, reference[len("/uploads/") :]))
    else:
        return _resolve_owned_asset(upload_root, user_id, reference)
    if found.startswith(owner_root + os.sep) and os.path.isfile(found):
        return found
    return None


def _seed(payload: dict):
    raw = payload.get("seed")
    try:
        return int(str(raw).strip()) if raw not in (None, "") else None
    except ValueError:
        raise InvalidJobPayload("seed must be a whole number") from None


def _sound(job: dict) -> tuple:
    """Gives a generated clip its sound: foley and/or a music bed and/or a voice-over, mixed and muxed.

    Reads plain values (the interpreter rendered the templates). The silent original is kept beside the
    clip with sound; both are sealed like every output. Foley asks for a model by name, because the only
    one this worker can run is non-commercial (``models.yaml``): without ``foleyModel`` a foley prompt is
    refused rather than quietly made with a licence the caller did not choose.
    """
    with tempfile.TemporaryDirectory(prefix="orz-sound-") as scratch:
        import time

        payload = job.get("payload") or {}
        upload_root = _upload_root()
        user_id = job.get("userId")

        reference = payload.get("video")
        video = _resolve_owned_output(upload_root, user_id, str(reference), scratch) if reference else None
        if not video:
            raise InvalidJobPayload("the video to give sound to was not found among your files")
        voice_ref = payload.get("voice") or None
        voice = _resolve_owned_output(upload_root, user_id, str(voice_ref), scratch) if voice_ref else None
        if voice_ref and not voice:
            raise InvalidJobPayload("the voice-over was not found among your files")

        foley_prompt = (payload.get("foley") or "").strip()
        music_prompt = (payload.get("music") or "").strip()
        if not (foley_prompt or music_prompt or voice):
            raise InvalidJobPayload("describe a foley, a music bed, or bring a voice-over")
        seed = _seed(payload)
        try:
            foley_model = sound_generators.resolve("foley", payload.get("foleyModel")) if foley_prompt else None
            music_model = sound_generators.resolve("music", payload.get("musicModel")) if music_prompt else None
        except sound_generators.UnknownSoundModel as unknown:
            raise InvalidJobPayload(str(unknown)) from unknown
        if foley_prompt and foley_model is None:
            raise InvalidJobPayload(
                "no commercially licensed foley model is installed: ask for one by name with foleyModel "
                "(a non-commercial model is refused unless MEDIA_ALLOW_NON_COMMERCIAL_WEIGHTS=true) or leave foley empty"
            )

        output_path, public_url = _output_paths(job["userId"], job["jobId"])
        stages = {}
        readable = _materialise(video, scratch)
        tracks = {}
        if foley_model:
            started = time.time()
            wav = os.path.join(scratch, "foley.wav")
            stages["foley"] = sound_generators.generate_foley(
                foley_model, readable, foley_prompt, sound.probe(readable)["durationSeconds"], wav, seed
            )
            tracks["foley"] = wav
            stages["foley"]["wallSeconds"] = round(time.time() - started, 1)
        if music_model:
            started = time.time()
            wav = os.path.join(scratch, "music.wav")
            stages["music"] = sound_generators.generate_music(
                music_model, music_prompt, sound.probe(readable)["durationSeconds"] + 1, wav, seed
            )
            tracks["music"] = wav
            stages["music"]["wallSeconds"] = round(time.time() - started, 1)
        if voice:
            tracks["voice"] = _materialise(voice, scratch)
        if not foley_model and sound.probe(readable)["audio"]:
            # The picture already carries its own sound (a cut's synthesized foley, ADR-080): it is the foley track
            # of the mix, under the music and the voice — never silently dropped.
            own = os.path.join(scratch, "own.wav")
            sound._run(
                ["ffmpeg", "-y", "-hide_banner", "-v", "error", "-i", readable, "-vn", "-ac", "2", "-ar", "48000", own]
            )
            tracks["foley"] = own
            stages["foley"] = {"model": "picture's own sound"}
        started = time.time()
        try:
            measured = sound.add_sound(readable, sound.Tracks(**tracks), output_path)
        except sound.SoundError as bad:
            raise InvalidJobPayload(str(bad)) from bad
        stages["mix"] = {"wallSeconds": round(time.time() - started, 1)}

        silent_path = measured.pop("silentPath")
        url = _store(output_path, job, "video.mp4", "video/mp4", public_url)
        silent_url = _store(
            silent_path, job, "video.silent.mp4", "video/mp4", public_url.replace("video.mp4", "video.silent.mp4")
        )
        metrics = {
            "durationSeconds": measured["durationSeconds"],
            "width": measured["width"],
            "height": measured["height"],
            "loudnessLufs": measured["loudnessLufs"],
            "tracks": measured["tracks"],
            "stages": stages,
            "models": {kind: stage["model"] for kind, stage in stages.items() if "model" in stage},
        }
        result = {"url": url, "assetId": url, "silentUrl": silent_url, "format": "mp4", "metrics": metrics}
        return result, extract_consumption(metrics)


def _is_sound(routing_key: str) -> bool:
    """Whether the broker delivered this message under the sound binding (read from the key, like compose)."""
    return bool(routing_key) and routing_key.startswith(SOUND_PREFIX)


class InvalidJobPayload(ValueError):
    """The payload was positively checked and found unusable (ADR-053 ``INPUT_INVALID``).

    A distinct type rather than a string test, because ``INPUT_INVALID`` is the only cause that
    BILLS the actor for the work a run already did. Raising it is a claim this worker is
    accountable for, and a bare ``except`` must never be able to make that claim by accident —
    ADR-046 §2's counter-example was our own defect wearing a user's fault.
    """


def _hold_id(job: dict):
    """The credit reservation the submitting service stamped into the payload."""
    payload = job.get("payload") or {}
    hold_id = payload.get("holdId")
    return hold_id if isinstance(hold_id, str) and hold_id.strip() else None


def _dispatch(channel, method, properties, body) -> None:
    """The broker's callback: runs the job on a thread of its own and returns at once.

    A Wan clip takes minutes. Run inline, it would stop the connection's I/O loop for all of them,
    heartbeats included, and the broker would close the connection mid-render and redeliver the job
    to start it again from zero. Prefetch is one, so there is never more than one such thread.
    """
    threading.Thread(
        target=_serialised,
        args=(channel, method, properties, body),
        daemon=True,
        name="media-job",
    ).start()


def _serialised(channel, method, properties, body) -> None:
    """One job at a time on this host: a video render and a sound mix never share the accelerator."""
    with _SERIAL:
        _on_message(channel, method, properties, body)


#: Jobs this process is running or finished within the hour, by id. A tunnel that drops mid-clip makes the
#: broker requeue the unacknowledged message, and the reconnected consumer is handed it again while the first
#: render is still going (or just done): that second delivery is acknowledged and skipped, never rendered twice.
_SEEN: dict[str, float] = {}
_SEEN_LOCK = threading.Lock()
_SEEN_TTL_SECONDS = 3600


def _first_delivery(job_id) -> bool:
    """Records the job as seen and says whether this is its first delivery to this process."""
    if not job_id:
        return True
    now = time.time()
    with _SEEN_LOCK:
        for old in [k for k, at in _SEEN.items() if now - at > _SEEN_TTL_SECONDS]:
            del _SEEN[old]
        if job_id in _SEEN:
            return False
        _SEEN[job_id] = now
        return True


def _on_message(channel, method, properties, body) -> None:
    job_id = None
    hold_id = None
    activity.touch()
    # The whole job, model load included, keeps the pod busy for the watchdog: a load after `ready` leaves the GPU idle.
    heartbeat = activity.Heartbeat().start()
    try:
        job = json.loads(body.decode("utf-8"))
        job_id = job.get("jobId")
        if not _first_delivery(job_id):
            print(f"[consumer] job {job_id} delivered again while it runs or just ran: skipped", flush=True)
            return  # acknowledged by the finally below, once
        hold_id = _hold_id(job)
        if not job_id or not job.get("userId"):
            raise InvalidJobPayload("JobCommand is missing jobId or userId")
        if job_control.is_cancelled(job_id):
            # Its owner stopped it while it waited in the queue: drop it before a GPU second is spent.
            print(f"[consumer] job {job_id} was cancelled while queued: dropped", flush=True)
            send_job_cancelled(job_id, hold_id=hold_id)
            return
        send_progress_update(job_id, 0, phase="loading")
        stage_reporter = stages.reporter_for(job_id, job.get("model"))
        stage_reporter.wake()
        if _is_sound(method.routing_key):
            print(f"[consumer] received sound job {job_id}", flush=True)
            result, consumption = _sound(job)
            model = COMPOSE_ENGINE
        elif _is_compose(method.routing_key):
            print(f"[consumer] received compose job {job_id}", flush=True)
            result, consumption = _compose(job)
            # The engine that ran, reported so an assembly is priced as an assembly. Both
            # branches are VIDEO work, and without this the pricebook can only reach the
            # capability default — the diffusion rate — for an ffmpeg concat. Derived from the
            # routing key like the branch above it, never from the feature key, so no pack
            # identity enters this worker [PACK-003].
            model = COMPOSE_ENGINE
        else:
            print(f"[consumer] received video job {job_id}", flush=True)
            result, consumption = _generate(job)
            model = job.get("model") or None
        stage_reporter.completed()
        send_job_done(job_id, result, hold_id=hold_id, consumption=consumption, model=model)
        print(f"[consumer] job {job_id} done: {result['url']}", flush=True)
    except job_control.JobCancelled:
        print(f"[consumer] job {job_id} cancelled by its owner", flush=True)
        if job_id:
            send_job_cancelled(job_id, hold_id=hold_id)
    except InvalidJobPayload as bad:
        print(f"[consumer] job {job_id} rejected: {bad}", flush=True)
        if job_id:
            send_job_error(job_id, str(bad), hold_id=hold_id, cause=INPUT_INVALID)
    except (ConnectionError, FileNotFoundError) as missing:
        # Something this worker depends on was not there — a socket, a binary. Not our logic being
        # wrong and not the user's payload: it releases, and the audit trail says which morning
        # this was rather than lumping it in with our defects.
        print(f"[consumer] job {job_id} could not run: {missing}", flush=True)
        if job_id:
            send_job_error(job_id, str(missing), hold_id=hold_id, cause=PLATFORM_UNAVAILABLE)
    except Exception as e:  # honest failure: report and ack — a bad job never succeeds on retry
        print(f"[consumer] job {job_id} failed: {e}", flush=True)
        if job_id:
            # EXECUTOR_FAULT, not a guess at the user's fault: anything reaching here is something
            # this worker did not anticipate, and not anticipating it is ours by definition. The
            # hold is carried so billing releases it.
            send_job_error(job_id, str(e), hold_id=hold_id, cause=EXECUTOR_FAULT)
    finally:
        heartbeat.stop()
        stages.release(job_id)
        activity.touch()
        # Acked from the connection's own thread (BlockingConnection is not thread-safe); a connection that is
        # gone cannot be acked — the broker requeued the message and its redelivery is skipped (_first_delivery).
        try:
            channel.connection.add_callback_threadsafe(
                functools.partial(channel.basic_ack, delivery_tag=method.delivery_tag)
            )
        except pika.exceptions.AMQPError as lost:
            print(f"[consumer] could not acknowledge job {job_id} (the connection is gone): {lost}", flush=True)


def _consume_once() -> None:
    """One connection's worth of consuming: returns on a clean stop, raises when the connection is lost."""
    params = broker.connection_parameters()
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    # Idempotent duplicate of the platform topology (§6) so the worker can start first.
    channel.exchange_declare(exchange=JOBS_EXCHANGE, exchange_type="topic", durable=True)
    channel.exchange_declare(exchange=DLX_EXCHANGE, exchange_type="direct", durable=True)
    try:
        max_length = int(os.environ.get("BROKER_QUEUE_MAX_LENGTH", "1000"))
    except ValueError:
        max_length = 1000
    # Argument-for-argument copy of the platform's workQueueArguments (JobsTopologyConfig):
    # RabbitMQ rejects a re-declare with inequivalent arguments (PRECONDITION_FAILED).
    channel.queue_declare(
        queue=VIDEO_QUEUE,
        durable=True,
        arguments={
            "x-max-length": max_length,
            "x-overflow": "reject-publish",
            "x-dead-letter-exchange": DLX_EXCHANGE,
            "x-dead-letter-routing-key": VIDEO_QUEUE,
        },
    )
    # Bind from the declaration, not from literals: worker.yaml is the single source for what
    # this process drains, and the same list is what it registers with the platform.
    for binding in BINDINGS:
        channel.queue_bind(queue=VIDEO_QUEUE, exchange=JOBS_EXCHANGE, routing_key=binding)
    channel.queue_declare(queue=VIDEO_DLQ, durable=True)
    channel.queue_bind(queue=VIDEO_DLQ, exchange=DLX_EXCHANGE, routing_key=VIDEO_QUEUE)

    channel.basic_qos(prefetch_count=1)
    if not sound_only():
        channel.basic_consume(queue=VIDEO_QUEUE, on_message_callback=_dispatch)
    if sound_enabled() and not hosted_only():
        channel.queue_declare(
            queue=SOUND_QUEUE,
            durable=True,
            arguments={
                "x-max-length": max_length,
                "x-overflow": "reject-publish",
                "x-dead-letter-exchange": DLX_EXCHANGE,
                "x-dead-letter-routing-key": SOUND_QUEUE,
            },
        )
        for binding in SOUND_BINDINGS:
            channel.queue_bind(queue=SOUND_QUEUE, exchange=JOBS_EXCHANGE, routing_key=binding)
        channel.queue_declare(queue=SOUND_QUEUE + ".dlq", durable=True)
        channel.queue_bind(queue=SOUND_QUEUE + ".dlq", exchange=DLX_EXCHANGE, routing_key=SOUND_QUEUE)
        channel.basic_consume(queue=SOUND_QUEUE, on_message_callback=_dispatch)
        print(f"[consumer] consuming {SOUND_QUEUE} ({' + '.join(SOUND_BINDINGS)})", flush=True)
    print(
        f"[consumer] consuming {VIDEO_QUEUE} ({' + '.join(BINDINGS)}) ({broker.describe(params)})",
        flush=True,
    )
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
        raise
    finally:
        try:
            connection.close()
        except pika.exceptions.AMQPError:
            pass  # already gone: the reconnect below is the answer


def run_consumer(retry_seconds: float = 5.0, consume=None, sleep=time.sleep) -> None:
    """Consumes for as long as the process lives, reconnecting when the connection is lost.

    The GPU droplet reaches the broker through a tunnel that App Platform restarts at every deploy of the
    app. A dropped connection used to end this process (and with it the clip being rendered, minutes of
    GPU); now the render thread keeps going, the consumer reconnects, and the broker's redelivery of the
    unacknowledged message is recognised and skipped (:func:`_first_delivery`).
    """
    consume = consume or _consume_once
    attempt = 0
    while True:
        try:
            consume()
            return
        except KeyboardInterrupt:
            return
        except (pika.exceptions.AMQPError, OSError) as lost:
            attempt += 1
            wait = min(retry_seconds * attempt, 30.0)
            print(f"[consumer] connection lost ({type(lost).__name__}); reconnecting in {wait:.0f}s", flush=True)
            sleep(wait)


if __name__ == "__main__":
    http_thread = threading.Thread(target=main.run, args=(_http_port(),), daemon=True, name="media-http")
    http_thread.start()
    # Registration runs on its own daemon thread and never gates consumption: a worker that
    # cannot reach the job service still drains its queues (ADR-038).
    start_background_registration(DECLARATION)
    job_control.start_cancel_listener()
    run_consumer()
