"""Stopping a job that is running or waiting on this host (orazaka-web-client#85).

The platform asks a worker to stop by publishing ``job.cancel.<jobId>`` on the jobs exchange
(the Studio does it when its owner cancels a run). Every worker hears it on a queue of its own
(exclusive, auto-delete — it dies with the connection, so nothing piles up while a worker is down).

Stopping is **cooperative**: killing a half-rendered GPU job leaves the accelerator in a worse state
than finishing it. The generators call :func:`raise_if_cancelled` at each step boundary, so a render
ends within one denoising step, and a job that was still queued is dropped when its message arrives.
"""

from __future__ import annotations

import threading
import time
import uuid

try:  # pika is replaced by a mock in some test runs; the registry below does not need it
    import pika
except ImportError:  # pragma: no cover
    pika = None

from app import broker

JOBS_EXCHANGE = "orazaka.jobs"
CANCEL_PREFIX = "job.cancel."
CANCEL_BINDING = CANCEL_PREFIX + "*"

#: How long a cancellation is remembered: a cancel that reaches the worker before the job does must
#: still drop the job when the broker finally hands it over (the queue can be long).
_TTL_SECONDS = 6 * 3600

_CANCELLED: dict[str, float] = {}
_LOCK = threading.Lock()


class JobCancelled(Exception):  # noqa: N818 — a control-flow signal, not an error
    """Raised inside a running job when its owner cancelled it."""


def request_cancel(job_id: str, now: float | None = None) -> None:
    """Records that the job must stop (idempotent)."""
    if not job_id:
        return
    stamp = time.time() if now is None else now
    with _LOCK:
        for old in [k for k, at in _CANCELLED.items() if stamp - at > _TTL_SECONDS]:
            del _CANCELLED[old]
        _CANCELLED[job_id] = stamp


def is_cancelled(job_id) -> bool:
    """Whether a cancel was received for this job."""
    if not job_id:
        return False
    with _LOCK:
        return job_id in _CANCELLED


def raise_if_cancelled(job_id) -> None:
    """The step boundary of a generator: ends the job right here when it was cancelled."""
    if is_cancelled(job_id):
        raise JobCancelled(f"job {job_id} was cancelled by its owner")


def job_id_from_routing_key(routing_key: str) -> str | None:
    """``job.cancel.<id>`` -> ``<id>``; anything else -> ``None``."""
    if not routing_key or not routing_key.startswith(CANCEL_PREFIX):
        return None
    job_id = routing_key[len(CANCEL_PREFIX) :].strip()
    return job_id or None


def _on_cancel(channel, method, properties, body) -> None:
    job_id = job_id_from_routing_key(method.routing_key)
    if job_id:
        request_cancel(job_id)
        print(f"[cancel] job {job_id} will stop at its next step", flush=True)
    channel.basic_ack(delivery_tag=method.delivery_tag)


def _listen_once() -> None:
    connection = pika.BlockingConnection(broker.connection_parameters())
    try:
        channel = connection.channel()
        channel.exchange_declare(exchange=JOBS_EXCHANGE, exchange_type="topic", durable=True)
        queue = f"orazaka.cancel.{uuid.uuid4().hex}"
        channel.queue_declare(queue=queue, exclusive=True, auto_delete=True)
        channel.queue_bind(queue=queue, exchange=JOBS_EXCHANGE, routing_key=CANCEL_BINDING)
        channel.basic_consume(queue=queue, on_message_callback=_on_cancel)
        channel.start_consuming()
    finally:
        try:
            connection.close()
        except Exception:  # noqa: BLE001, S110 — already gone: the loop reconnects
            pass


def listen_for_cancels(retry_seconds: float = 5.0, listen=None, sleep=time.sleep) -> None:
    """Hears cancellations for as long as the process lives, reconnecting when the link drops."""
    listen = listen or _listen_once
    attempt = 0
    while True:
        try:
            listen()
            return
        except KeyboardInterrupt:
            return
        except Exception as lost:  # noqa: BLE001 — any broker failure: wait and dial again
            attempt += 1
            wait = min(retry_seconds * attempt, 30.0)
            print(f"[cancel] listener lost ({type(lost).__name__}); retrying in {wait:.0f}s", flush=True)
            sleep(wait)


def start_cancel_listener() -> threading.Thread:
    """Starts the listener on a daemon thread; it never gates consumption."""
    thread = threading.Thread(target=listen_for_cancels, daemon=True, name="media-cancel")
    thread.start()
    return thread
