import gc
import json
import os
import time
from contextlib import contextmanager

import pika

try:
    import torch
except ImportError:
    torch = None

# AGENTS.md §6 topology: job lifecycle events go to the orazaka.events topic
# exchange with the key job.{jobId}.progress|done|error. This worker duplicates
# the contract subset it publishes (no shared jar across the service boundary).
EVENTS_EXCHANGE = 'orazaka.events'


def _publish_job_event(routing_key: str, payload: dict) -> None:
    host = os.environ.get("SPRING_RABBITMQ_HOST", "localhost")
    try:
        port = int(os.environ.get("SPRING_RABBITMQ_PORT", "5672"))
    except ValueError:
        port = 5672

    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host, port=port)
        )
        channel = connection.channel()
        channel.exchange_declare(exchange=EVENTS_EXCHANGE, exchange_type='topic', durable=True)
        message = json.dumps(payload)
        channel.basic_publish(
            exchange=EVENTS_EXCHANGE,
            routing_key=routing_key,
            body=message,
            properties=pika.BasicProperties(
                content_type='application/json',
                delivery_mode=2,
                # spring-amqp 4.0.0 (Boot 4) SimpleAmqpHeaderMapper.toHeaders NPEs on a
                # null AMQP priority; set an explicit 0 so the router's event consumer
                # doesn't log a stack trace per event.
                priority=0
            )
        )
        connection.close()
        print(f"Sent AMQP job event {routing_key}: {message}", flush=True)
    except Exception as e:
        print(f"Failed to send AMQP job event {routing_key} to RabbitMQ: {e}", flush=True)


def send_progress_update(job_id: str, progress: int) -> None:
    _publish_job_event(f"job.{job_id}.progress", {"jobId": job_id, "progress": int(progress)})


def _with_reservation(payload: dict, hold_id) -> dict:
    """Attach the credit reservation when the job carried one (ADR-033 §6.3).

    The key is omitted rather than set to null when absent, so absence means
    "never authorised through billing" instead of "authorised, hold unknown".
    """
    if hold_id:
        payload["holdId"] = str(hold_id)
    return payload


def send_job_done(
    job_id: str,
    result: dict,
    hold_id=None,
    consumption: dict = None,
    model: str = None,
) -> None:
    """Terminal success event — the router applies COMPLETED with this result.

    ``consumption`` carries raw measurements only (frames, steps, pixels, GPU
    seconds). The billable unit is a property of the pricebook row the hold was
    pinned to and is resolved by the billing service; this worker reports what
    it measured and never names a unit.

    ``model`` names the engine that produced the result. It is the other half of
    the pricebook key ``(capability, model)``, and it is what lets an assembly be
    priced differently from a generation even though both are VIDEO work: an
    orchestrator settling a run prices each step against its own row, and without
    a model it can only reach the capability default. Reporting what ran is not
    naming a price — the rate for that engine still lives in the pricebook.
    """
    payload = {"jobId": job_id, "result": result}
    if consumption:
        payload["consumption"] = consumption
    if model:
        payload["model"] = model
    _publish_job_event(f"job.{job_id}.done", _with_reservation(payload, hold_id))


#: The closed vocabulary of ADR-053, mirrored from ``FailureCause`` on the Java side.
#: A worker MUST name one of these. The saga reads the category and never the message.
GUARD_REFUSAL = "GUARD_REFUSAL"
INPUT_INVALID = "INPUT_INVALID"
EXECUTOR_FAULT = "EXECUTOR_FAULT"
PLATFORM_UNAVAILABLE = "PLATFORM_UNAVAILABLE"
TIMEOUT = "TIMEOUT"

FAILURE_CAUSES = (
    GUARD_REFUSAL,
    INPUT_INVALID,
    EXECUTOR_FAULT,
    PLATFORM_UNAVAILABLE,
    TIMEOUT,
)


def send_job_error(job_id: str, error: str, hold_id=None, cause: str = EXECUTOR_FAULT) -> None:
    """Terminal failure event — the saga applies FAILED with this **category**.

    Carries the reservation so billing releases it: a failed generation is
    never billed, and a hold nobody closes freezes the user's balance until the
    sweeper expires it.

    ``cause`` is the typed reason, from :data:`FAILURE_CAUSES` (ADR-053). It is
    **declared**, not inferred: this worker is the only party that knows whether
    it rejected a payload, lost its model, or was stopped by a guard, and every
    consumer downstream was reading prose and guessing.

    ``INPUT_INVALID`` is the only cause that BILLS the actor for the work a run
    already did. Set it only where this worker positively validated the payload
    and found it unusable — never in a bare ``except``, where the exception is
    as likely to be our own defect. An unrecognised or absent cause degrades to
    ``EXECUTOR_FAULT``, which releases the hold and blames nobody.
    """
    declared = cause if cause in FAILURE_CAUSES else EXECUTOR_FAULT
    _publish_job_event(
        f"job.{job_id}.error",
        _with_reservation(
            {"jobId": job_id, "error": str(error), "cause": declared}, hold_id
        ),
    )


# Keys of the resource_guard metrics dict that are consumption measurements
# rather than host diagnostics — see ConsumptionReport on the billing side.
# `durationSeconds` is what OUTPUT_SECOND is priced on since ADR-066; `frames`/`fps` stay for
# executors that report a frame count and no duration, and are no longer why the composer forces
# a constant rate.
CONSUMPTION_KEYS = (
    "gpuSeconds", "durationSeconds", "frames", "fps", "images", "steps", "width", "height",
)


def extract_consumption(metrics: dict) -> dict:
    """Split the consumption measurements out of a metrics dict.

    ``resource_guard`` mixes host diagnostics (peak RSS, allocated VRAM) with
    the measurements billing settles against. Only the latter cross the wire as
    ``consumption``; the rest stay in ``result.metrics`` for the operator.
    """
    if not isinstance(metrics, dict):
        return {}
    return {k: metrics[k] for k in CONSUMPTION_KEYS if metrics.get(k) is not None}

@contextmanager
def resource_guard(process):
    """
    Context manager that tracks execution duration, peak RSS memory usage,
    and ensures garbage collection and GPU cache eviction are run.
    """
    start_time = time.time()
    metrics = {}
    try:
        yield metrics
    finally:
        gc.collect()
        if torch and hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:  # noqa: S110
                # Best-effort hint to the allocator. A failed cache flush must never turn a
                # successful generation into a failed telemetry write.
                pass
            
        end_time = time.time()
        end_mem = process.memory_info().rss if process else 0
        gpu_allocated_mb = 0.0
        if torch and torch.cuda.is_available():
            gpu_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
        metrics["inference_time_sec"] = round(end_time - start_time, 2)
        metrics["peak_memory_rss_mb"] = round(end_mem / (1024 * 1024), 2)
        metrics["gpu_allocated_mb"] = round(gpu_allocated_mb, 2)
        # On owned hardware the accelerator is occupied for the whole render, so wall
        # clock is the honest cost signal (design §7). This is the calibration basis
        # that turns guessed credits_per_unit into measured ones during phase 0.
        metrics["gpuSeconds"] = metrics["inference_time_sec"]
