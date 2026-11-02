"""Daily ceilings on what the paid video provider may cost: N clips and a budget in USD per UTC day.

Every remote clip (Wan on serverless inference, $0.60 each) first ``reserve``s here; over a ceiling the
job fails with a clear reason *before* the provider is called. Local models never pass through here.
Configuration, bounded context ``AI_``: ``AI_VIDEO_DAILY_RUNS`` (default 5, negative = off),
``AI_SPEND_DAILY_BUDGET_USD`` (default 5, negative = off), ``AI_SPEND_STATE_FILE`` (the day's counters,
kept on disk so a restart does not give the day back).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import tempfile
import threading

log = logging.getLogger(__name__)
_lock = threading.Lock()

DEFAULT_DAILY_RUNS = 5
DEFAULT_DAILY_BUDGET_USD = 5.0


class SpendRefused(RuntimeError):
    """The day's ceiling is reached: the clip is not requested."""


def _state_path(env: dict) -> str:
    return env.get("AI_SPEND_STATE_FILE") or os.path.join(tempfile.gettempdir(), "orazaka-worker-spend.json")


def _today(now: dt.datetime | None) -> str:
    return (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%d")


def _load(path: str, day: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("day") == day:
            return state
    except (OSError, ValueError):
        pass
    return {"day": day, "runs": 0, "usd": 0.0}


def reserve(cost_usd: float, env: dict | None = None, now: dt.datetime | None = None) -> None:
    """Counts one paid clip against today's ceilings, or raises :class:`SpendRefused` (and logs an alert)."""
    source = os.environ if env is None else env
    max_runs = int(source.get("AI_VIDEO_DAILY_RUNS") or DEFAULT_DAILY_RUNS)
    budget = float(source.get("AI_SPEND_DAILY_BUDGET_USD") or DEFAULT_DAILY_BUDGET_USD)
    path, day = _state_path(source), _today(now)
    with _lock:
        state = _load(path, day)
        if max_runs >= 0 and state["runs"] + 1 > max_runs:
            log.error(
                "SPEND ALERT: %d paid video clips today, the daily limit (%d) is reached", int(state["runs"]), max_runs
            )
            raise SpendRefused(f"the daily limit of {max_runs} video clips is reached; it resets at 00:00 UTC")
        if budget >= 0 and state["usd"] + cost_usd > budget:
            log.error(
                "SPEND ALERT: %.2f USD spent today, the daily budget (%.2f) is reached", float(state["usd"]), budget
            )
            raise SpendRefused(
                f"the daily budget of {budget:.2f} USD for paid providers is reached; it resets at 00:00 UTC"
            )
        state["runs"] += 1
        state["usd"] = round(state["usd"] + cost_usd, 4)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(state, handle)
        except OSError:
            log.warning("could not persist the spend counters at %s; counting in this run only", path)
