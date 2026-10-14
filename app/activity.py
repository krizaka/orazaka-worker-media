"""The worker's last sign of work, for whatever decides that an hourly GPU is idle.

A GPU droplet is billed every hour it exists, busy or not, so the host runs a watchdog that destroys
it after a stretch without jobs (krizaka/orazaka ``infra/gpu``). The watchdog cannot see inside this
process; this module is how the process tells it: the modification time of one file, touched when a
job starts and when it ends. A long generation is not idle — the start touch and the GPU's own
utilisation both say so.

Unset (``MEDIA_ACTIVITY_FILE`` empty, the Mac), nothing is written.
"""
from __future__ import annotations

import os
from pathlib import Path


def activity_file(env: dict | None = None) -> Path | None:
    """The file the deployment declared, or ``None`` when none was."""
    source = os.environ if env is None else env
    value = (source.get("MEDIA_ACTIVITY_FILE") or "").strip()
    return Path(value) if value else None


def touch(env: dict | None = None) -> None:
    """Marks "work happened now". Never raises: a watchdog signal must not fail a job."""
    target = activity_file(env)
    if target is None:
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    except OSError as failure:
        print(f"[activity] could not touch {target}: {failure}", flush=True)
