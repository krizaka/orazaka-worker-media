"""Deterministic progress (orazaka-worker-media#64): the named stages of a job, in order, with real values.

``COLD_START_WAKEUP`` -> ``STREAMING_WEIGHTS_NVME`` (the real percentage of weight bytes read) ->
``INFERENCE_STEP`` (step n of N, with a sliding ETA) -> ``SANDBOX_VERIFICATION`` -> ``COMPLETED``.

They ride the existing ``job.<id>.progress`` event on the RabbitMQ events exchange, as optional keys
(ADR-048): a core that does not know them still reads ``jobId`` / ``progress`` / ``phase``. Nothing is
invented: an unknown total gives no percentage, no history gives no ETA.
"""

import json
import os
import statistics
import threading
import time
from collections import deque

STAGES = (
    "COLD_START_WAKEUP",
    "STREAMING_WEIGHTS_NVME",
    "INFERENCE_STEP",
    "SANDBOX_VERIFICATION",
    "COMPLETED",
)
_RANK = {name: rank for rank, name in enumerate(STAGES)}

#: The closed phase vocabulary the UI already words (telemetry.PHASES) each stage belongs to.
_PHASE = {
    "COLD_START_WAKEUP": "loading",
    "STREAMING_WEIGHTS_NVME": "loading",
    "INFERENCE_STEP": "generating",
    "SANDBOX_VERIFICATION": "encoding",
    "COMPLETED": "uploading",
}

#: The latest step durations the ETA slides over, and the past loads kept per (model, hardware).
STEP_WINDOW = 5
HISTORY_KEEP = 5


class EtaHistory:
    """The last durations per (model, hardware, kind), kept in memory and, when asked, in a small JSON file."""

    def __init__(self, keep: int = HISTORY_KEEP, path: str | None = None):
        self._keep = keep
        self._path = path
        self._lock = threading.Lock()
        self._runs: dict[str, list[float]] = {}
        if path:
            try:
                with open(path, encoding="utf-8") as handle:
                    self._runs = {k: [float(v) for v in vs][-keep:] for k, vs in json.load(handle).items()}
            except (OSError, ValueError, AttributeError):
                self._runs = {}

    @staticmethod
    def _key(model, hardware, kind) -> str:
        return f"{model}|{hardware}|{kind}"

    def record(self, model, hardware, kind, seconds: float) -> None:
        if seconds is None or seconds < 0:
            return
        with self._lock:
            runs = self._runs.setdefault(self._key(model, hardware, kind), [])
            runs.append(float(seconds))
            del runs[: -self._keep]
            if self._path:
                try:
                    with open(self._path, "w", encoding="utf-8") as handle:
                        json.dump(self._runs, handle)
                except OSError:
                    pass  # the ETA is a convenience: it never fails a job

    def estimate(self, model, hardware, kind):
        with self._lock:
            runs = self._runs.get(self._key(model, hardware, kind))
            return round(statistics.median(runs)) if runs else None


def history_from_env(env: dict | None = None) -> EtaHistory:
    return EtaHistory(path=(env if env is not None else os.environ).get("MEDIA_ETA_FILE") or None)


def _default_send(job_id):
    from app import telemetry

    def send(payload):
        telemetry._publish_job_event(f"job.{job_id}.progress", payload)

    return send


class StageReporter:
    """One job's stages: each one is said once (the weights and the steps as often as they move), never back."""

    def __init__(self, job_id, model, hardware, history: EtaHistory, send=None, clock=time.time):
        self.job_id, self.model, self.hardware = job_id, model, hardware
        self._history, self._clock = history, clock
        self._send = send or _default_send(job_id)
        self._rank = -1
        self._at = 0
        self._percent = -1
        self._woke = None
        self._load_recorded = False
        self._last_step_at = None
        self._steps = deque(maxlen=STEP_WINDOW)

    def _emit(self, stage, progress, **extra) -> None:
        from app.telemetry import progress_payload

        self._at = max(self._at, int(self._clock() * 1000))  # monotonic even if the clock steps back
        self._send(progress_payload(self.job_id, progress, phase=_PHASE[stage], stage=stage, at=self._at, **extra))

    def _advance(self, stage, repeatable=False) -> bool:
        rank = _RANK[stage]
        if rank < self._rank or (rank == self._rank and not repeatable):
            return False
        self._rank = rank
        return True

    def wake(self) -> None:
        if self._advance("COLD_START_WAKEUP"):
            self._woke = self._clock()
            self._emit("COLD_START_WAKEUP", 0, eta_seconds=self._history.estimate(self.model, self.hardware, "load"))

    def weights(self, read: int, total: int) -> None:
        if not total or total <= 0 or not self._advance("STREAMING_WEIGHTS_NVME", repeatable=True):
            return
        percent = max(self._percent, min(100, int(read * 100 / total)))
        self._percent = percent
        load = self._history.estimate(self.model, self.hardware, "load")
        eta = round(load * (100 - percent) / 100) if load is not None else None
        self._emit("STREAMING_WEIGHTS_NVME", percent, weights_percent=percent, eta_seconds=eta)

    def step(self, n: int, total: int) -> None:
        if not total or total <= 0 or not self._advance("INFERENCE_STEP", repeatable=True):
            return
        now = self._clock()
        if self._woke is not None and not self._load_recorded:
            self._load_recorded = True  # the load ended when generation began
            self._history.record(self.model, self.hardware, "load", now - self._woke)
        if self._last_step_at is not None:
            self._steps.append(now - self._last_step_at)
        self._last_step_at = now
        eta = round(statistics.fmean(self._steps) * max(total - n, 0)) if self._steps else None
        self._emit("INFERENCE_STEP", min(int(n * 100 / total), 100), step=n, steps=total, eta_seconds=eta)

    def verify(self) -> None:
        if self._advance("SANDBOX_VERIFICATION"):
            self._emit("SANDBOX_VERIFICATION", 100)

    def completed(self) -> None:
        if self._advance("COMPLETED"):
            self._emit("COMPLETED", 100)


_history = None
_reporters: dict = {}
_registry_lock = threading.Lock()


def hardware() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001, S110 — no torch (a slim image, a test): "cpu", never a failure
        pass
    return "cpu"


def reporter_for(job_id, model=None) -> StageReporter | None:
    """The job's reporter, shared by the consumer and the generator (same process); None without a job."""
    global _history
    if not job_id:
        return None
    with _registry_lock:
        if _history is None:
            _history = history_from_env()
        rep = _reporters.get(job_id)
        if rep is None:
            rep = _reporters[job_id] = StageReporter(job_id, model or "unknown", hardware(), _history)
        elif model and rep.model == "unknown":
            rep.model = model
        return rep


def release(job_id) -> None:
    with _registry_lock:
        _reporters.pop(job_id, None)
