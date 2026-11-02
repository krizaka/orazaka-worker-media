"""The worker's last sign of work, for whatever decides that an hourly GPU is idle.

A GPU droplet is billed every hour it exists, busy or not, so the host runs a watchdog that destroys
it after a stretch without jobs (krizaka/orazaka ``infra/gpu``). The watchdog cannot see inside this
process; this module is how the process tells it: the modification time of one file, touched when a
job starts and when it ends. A long generation is not idle — the start touch and the GPU's own
utilisation both say so.

A second file, ``MEDIA_HEARTBEAT_FILE``, says "busy right now": :func:`beating` touches it at once and then every
:data:`HEARTBEAT_EVERY_SECONDS` while a model loads or a job runs, so a load after ``ready`` with the GPU still idle is
never taken for an idle pod (the watchdog counts the pod busy while the file is younger than its horizon, 180 s;
krizaka-internal#2101).

Unset (``MEDIA_ACTIVITY_FILE`` / ``MEDIA_HEARTBEAT_FILE`` empty, the Mac), nothing is written.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
import functools
from typing import Callable, Iterator, TypeVar

_F = TypeVar("_F", bound=Callable)

#: Well inside the watchdog's 180 s horizon: two beats may be missed before the pod looks idle.
HEARTBEAT_EVERY_SECONDS = 30.0


def activity_file(env: dict | None = None) -> Path | None:
    """The file the deployment declared, or ``None`` when none was."""
    source = os.environ if env is None else env
    value = (source.get("MEDIA_ACTIVITY_FILE") or "").strip()
    return Path(value) if value else None


def heartbeat_file(env: dict | None = None) -> Path | None:
    """The "busy now" file the deployment declared, or ``None`` when none was."""
    source = os.environ if env is None else env
    value = (source.get("MEDIA_HEARTBEAT_FILE") or "").strip()
    return Path(value) if value else None


def _touch_file(target: Path) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    except OSError as failure:
        print(f"[activity] could not touch {target}: {failure}", flush=True)


def touch(env: dict | None = None) -> None:
    """Marks "work happened now". Never raises: a watchdog signal must not fail a job."""
    target = activity_file(env)
    if target is not None:
        _touch_file(target)


class Heartbeat:
    """Keeps the heartbeat file fresh from :meth:`start` to :meth:`stop` (a model load, a job).

    Touches at once, then every ``every`` seconds from a daemon thread. Never raises: a watchdog signal must not fail
    the work it reports on. Without a declared file, it does nothing.
    """

    def __init__(self, env: dict | None = None, every: float = HEARTBEAT_EVERY_SECONDS) -> None:
        self._target = heartbeat_file(env)
        self._every = every
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "Heartbeat":
        if self._target is None or self._thread is not None:
            return self
        _touch_file(self._target)
        self._thread = threading.Thread(target=self._beat, name="media-heartbeat", daemon=True)
        self._thread.start()
        return self

    def _beat(self) -> None:
        while not self._stop.wait(self._every):
            _touch_file(self._target)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


@contextmanager
def beating(env: dict | None = None, every: float = HEARTBEAT_EVERY_SECONDS) -> Iterator[None]:
    """:class:`Heartbeat` for the duration of a block; the block's own exception goes through unchanged."""
    heartbeat = Heartbeat(env, every).start()
    try:
        yield
    finally:
        heartbeat.stop()


def beats(function: _F) -> _F:
    """Decorator: the heartbeat runs for every call of ``function`` (a model load)."""

    @functools.wraps(function)
    def beating_call(*args, **kwargs):
        with beating():
            return function(*args, **kwargs)

    return beating_call  # type: ignore[return-value]
