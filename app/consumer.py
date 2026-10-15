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
import shutil
import tempfile
import threading
import urllib.error
import urllib.request

import pika

from app import activity, asset_sink, broker, envelope, main, video_models
from app.composer import compose
from app.registration import load_declaration, start_background_registration
from app.telemetry import (
    EXECUTOR_FAULT,
    INPUT_INVALID,
    PLATFORM_UNAVAILABLE,
    extract_consumption,
    send_job_done,
    send_job_error,
)

JOBS_EXCHANGE = "orazaka.jobs"
DLX_EXCHANGE = "orazaka.dlx"
VIDEO_QUEUE = "orazaka.jobs.video"
VIDEO_DLQ = VIDEO_QUEUE + ".dlq"

# What this worker drains is declared in worker.yaml and read at boot (ADR-038, S3).
# It used to be VIDEO_BINDING / COMPOSE_BINDING literals right here, which meant adding a
# binding was a Python edit; now the declaration is one file, used both to bind the queues
# and to register with the platform.
DECLARATION = load_declaration()
BINDINGS = DECLARATION["bindings"]

# Studio media composition (ADR-034 §7.2) shares this queue: it is ffmpeg work on this host's
# accelerator, so it shares the worker's serialised budget rather than competing with generation
# from a second consumer. Which binding means "compose" is still a property of the ROUTING KEY the
# broker delivered under — never of a capability name (P4).
COMPOSE_PREFIX = "job.compose."


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
    if asset_sink.sink_url() is not None:
        return asset_sink.fetch_image(job, os.path.join(scratch, "starting-image"))
    if os.path.isabs(named) and os.path.isfile(named):
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
        if refused.code == 400:
            # The pipeline positively checked the request and refused it (unknown model, an aspect
            # the model cannot make, a starting image for a text-to-video model): the caller's.
            raise InvalidJobPayload(refused.read().decode("utf-8", "replace")[:300]) from refused
        raise

    # The generator wrote plaintext through ffmpeg; seal it before anything else can read it
    # (ADR-054). In place and atomically, so a kill here leaves the plaintext or the envelope and
    # never a half-sealed file — or, on a host that does not share the store, hand it over.
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


def _compose(job: dict) -> tuple:
    """Assembles a Studio run's stills (and optional voiceover) into a vertical MP4.

    The payload arrives with every template already rendered by the interpreter,
    so this reads plain values and never interprets a blueprint.
    """
    job_id = job["jobId"]
    payload = job.get("payload") or {}

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
    resolved = [
        path
        for path in (
            _resolve_owned_asset(upload_root, user_id, str(item)) for item in photos
        )
        if path
    ]

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
        target=_on_message,
        args=(channel, method, properties, body),
        daemon=True,
        name="media-job",
    ).start()


def _on_message(channel, method, properties, body) -> None:
    job_id = None
    hold_id = None
    activity.touch()
    try:
        job = json.loads(body.decode("utf-8"))
        job_id = job.get("jobId")
        hold_id = _hold_id(job)
        if not job_id or not job.get("userId"):
            raise InvalidJobPayload("JobCommand is missing jobId or userId")
        if _is_compose(method.routing_key):
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
        send_job_done(job_id, result, hold_id=hold_id, consumption=consumption, model=model)
        print(f"[consumer] job {job_id} done: {result['url']}", flush=True)
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
        activity.touch()
        # Acked from the connection's own thread: pika's BlockingConnection is not thread-safe.
        channel.connection.add_callback_threadsafe(
            functools.partial(channel.basic_ack, delivery_tag=method.delivery_tag)
        )


def run_consumer() -> None:
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
    channel.basic_consume(queue=VIDEO_QUEUE, on_message_callback=_dispatch)
    print(
        f"[consumer] consuming {VIDEO_QUEUE} ({' + '.join(BINDINGS)}) ({broker.describe(params)})",
        flush=True,
    )
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    finally:
        connection.close()


if __name__ == "__main__":
    http_thread = threading.Thread(
        target=main.run, args=(_http_port(),), daemon=True, name="media-http"
    )
    http_thread.start()
    # Registration runs on its own daemon thread and never gates consumption: a worker that
    # cannot reach the job service still drains its queues (ADR-038).
    start_background_registration(DECLARATION)
    run_consumer()
