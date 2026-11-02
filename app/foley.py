"""Foley by synthesis (orazaka#289): the sound list of a cinematic short made without a model and without a licence.

ADR-080 refused the foley models a customer could not publish with (MMAudio and AudioLDM2 weights are non-commercial;
Stable Audio Open is gated behind an account decision; freesound's CC0 files need an account and per-file checks), so
the v1 cut laid one synthesized wind bed and whooshes on its transitions. orazaka#289 asked for more: **a sound for what
each shot shows** — wings, wind, rain, a lake, steps, paper, a coin, birds at dawn. This module makes each of them from
noise, resonators and envelopes, shaped in the frequency domain (numpy only: no scipy, no weights, nothing to license),
deterministic for a seed so a film is made again identically.

* **beds** span a shot (rain, storm, wind, lake, waves, a room, dawn birds, a far city…);
* **events** land on a second of it (a wing beat, a step, a coin, thunder, a splash…).

:func:`render` lays a timeline's cues (``orazaka cinema timeline``'s ``sound``) into one stereo stem, panned and
crossfaded, that the mix (``app/mixer.py``) puts under the voice and the music. :func:`coverage` says which shots have
a sound of their own, the gate's « foley coverage ».
"""

from __future__ import annotations

import numpy as np

RATE = 48000

BEDS = (
    "rain",
    "storm",
    "wind",
    "wind-soft",
    "lake",
    "waves",
    "drizzle",
    "room",
    "dawn",
    "city",
    "rain-window",
    "drips",
    "tunnel",
    "crowd",
    "neon",
)
EVENTS = (
    "thunder",
    "gust",
    "flap",
    "whoosh",
    "steps",
    "rustle",
    "shake",
    "paper",
    "block",
    "hum",
    "ignite",
    "chime",
    "call",
    "chirp",
    "splash",
    "splash-small",
    "coin",
    "parcel",
    "tap",
    "swell",
    "train",
    "skate",
    "sparks",
    "impact",
    "slide",
    "riser",
)
KINDS = BEDS + EVENTS

#: Beds fade in and out over this many seconds at their shot's edges (a cut is not a click).
BED_FADE = 0.25


class FoleyError(ValueError):
    """A cue the synthesizer cannot make (the caller's to fix)."""


def _noise(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def shaped(rng: np.random.Generator, n: int, low: float, high: float, tilt: float = 0.0) -> np.ndarray:
    """Noise kept between ``low`` and ``high`` Hz (soft edges), its spectrum tilted by ``tilt`` dB/octave (pure for a seed)."""
    spectrum = np.fft.rfft(_noise(rng, n))
    f = np.fft.rfftfreq(n, 1 / RATE)
    f[0] = 1.0
    gain = 1 / (1 + (low / f) ** 4) / (1 + (f / high) ** 4)
    gain *= (f / 1000.0) ** (tilt / 6.02)
    out = np.fft.irfft(spectrum * gain, n).astype(np.float32)
    return out / (np.max(np.abs(out)) + 1e-9)


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / RATE
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return (a * np.exp(-np.maximum(t - attack, 0) / max(decay, 1e-4))).astype(np.float32)


def _lfo(n: int, hz: float, depth: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / RATE
    phase = rng.uniform(0, 2 * np.pi)
    slow = np.sin(2 * np.pi * hz * t + phase) + 0.5 * np.sin(2 * np.pi * hz * 2.7 * t + phase * 1.3)
    return (1 - depth + depth * (0.5 + 0.5 * slow / 1.5)).astype(np.float32)


def _drops(rng: np.random.Generator, n: int, per_second: float, low: float, high: float, decay: float) -> np.ndarray:
    """Sparse resonant drops (rain on stone, a drip): each a short decaying sine at a random pitch."""
    out = np.zeros(n, np.float32)
    count = int(per_second * n / RATE)
    length = int(decay * 6 * RATE)
    t = np.arange(length) / RATE
    for _ in range(count):
        at = rng.integers(0, max(1, n - length))
        f = rng.uniform(low, high)
        drop = np.sin(2 * np.pi * f * t * (1 + 0.15 * np.exp(-t / decay))) * np.exp(-t / decay) * rng.uniform(0.2, 1.0)
        out[at : at + length] += drop.astype(np.float32)
    return out / (np.max(np.abs(out)) + 1e-9)


def _chirps(rng: np.random.Generator, n: int, count: int, base: float) -> np.ndarray:
    """Little bird calls: quick frequency sweeps with a warble, in short phrases."""
    out = np.zeros(n, np.float32)
    for _ in range(count):
        length = int(rng.uniform(0.05, 0.14) * RATE)
        if n <= length:
            break
        at = rng.integers(0, n - length)
        t = np.arange(length) / RATE
        f0 = base * rng.uniform(0.8, 1.4)
        sweep = f0 * (1 + rng.uniform(-0.35, 0.5) * t / t[-1]) + 120 * np.sin(2 * np.pi * rng.uniform(25, 45) * t)
        phase = 2 * np.pi * np.cumsum(sweep) / RATE
        out[at : at + length] += (np.sin(phase) * np.sin(np.pi * t / t[-1]) ** 2 * rng.uniform(0.3, 1)).astype(
            np.float32
        )
    return out / (np.max(np.abs(out)) + 1e-9)


def _flap(rng: np.random.Generator, size: float = 1.0) -> np.ndarray:
    """One wing beat: a soft low push of air and a feathered rustle on top, ~0.18 s."""
    n = int(0.22 * RATE)
    push = shaped(rng, n, 80, 600 / size) * _env(n, 0.02, 0.05)
    feathers = shaped(rng, n, 1500, 7000, -2) * _env(n, 0.008, 0.04) * 0.45
    return push + feathers


def bed(kind: str, seconds: float, seed: int) -> np.ndarray:
    """A bed of ``kind`` for ``seconds`` (mono, peak 1)."""
    rng = np.random.default_rng(seed)
    n = max(1, int(seconds * RATE))
    if kind in ("rain", "storm", "rain-window", "drizzle"):
        hiss = shaped(rng, n, 400 if kind != "rain-window" else 200, 9000 if kind != "rain-window" else 3500, -1.5)
        rate = {"rain": 220, "storm": 420, "rain-window": 120, "drizzle": 35}[kind]
        out = hiss * (0.35 if kind == "drizzle" else 0.8) + _drops(rng, n, rate, 1800, 5200, 0.004) * 0.35
        if kind == "storm":
            out = out + shaped(rng, n, 25, 160) * _lfo(n, 0.11, 0.6, rng) * 0.7
        return out
    if kind in ("wind", "wind-soft"):
        out = shaped(rng, n, 60, 1100 if kind == "wind" else 600, -3) * _lfo(n, 0.17, 0.7, rng)
        whistle = shaped(rng, n, 500, 1400) * _lfo(n, 0.23, 0.9, rng) * 0.15
        return out + (whistle if kind == "wind" else 0)
    if kind in ("lake", "waves"):
        laps = shaped(rng, n, 120, 1800 if kind == "lake" else 2600, -2) * _lfo(
            n, 0.45 if kind == "lake" else 0.22, 0.85, rng
        )
        return laps + _drops(rng, n, 6 if kind == "lake" else 12, 300, 900, 0.02) * 0.15
    if kind == "room":
        return shaped(rng, n, 40, 400, -3) * 0.4
    if kind == "dawn":
        air = shaped(rng, n, 200, 3000, -3) * 0.25
        return air + _chirps(rng, n, max(1, int(seconds * 3)), 3200) * 0.6
    if kind == "city":
        return shaped(rng, n, 30, 700, -3) * _lfo(n, 0.05, 0.3, rng) * 0.7
    if kind == "drips":
        return _drops(rng, n, 2.5, 700, 1600, 0.03)
    if kind == "tunnel":  # a metro tunnel: a deep air rumble, a far hum, water dripping in a big space
        rumble = shaped(rng, n, 20, 180, -3) * _lfo(n, 0.07, 0.5, rng)
        return rumble * 0.8 + _drops(rng, n, 1.5, 500, 1200, 0.06) * 0.4 + shaped(rng, n, 50, 70) * 0.15
    if kind == "crowd":  # a crowd heard through a wall: many murmurs, band-limited, breathing
        murmur = shaped(rng, n, 180, 1400, -2) * _lfo(n, 1.7, 0.35, rng) * _lfo(n, 0.3, 0.4, rng)
        return murmur * 0.6 + shaped(rng, n, 40, 120) * _lfo(n, 2.1, 0.8, rng) * 0.3
    if kind == "neon":  # an electric tube: a 120 Hz buzz with its harmonics and a crackle now and then
        t = np.arange(n) / RATE
        buzz = sum(np.sin(2 * np.pi * 120 * h * t) / h for h in (1, 2, 3, 4, 6)).astype(np.float32)
        flick = (rng.random(n) < 0.0004).astype(np.float32)
        return (
            buzz * 0.3 * _lfo(n, 0.5, 0.2, rng) + np.convolve(flick, np.hanning(200), "same").astype(np.float32) * 0.5
        )
    raise FoleyError(f"{kind!r} is not a bed")


def event(kind: str, seed: int) -> np.ndarray:
    """One event of ``kind`` (mono, peak 1, its own length)."""
    rng = np.random.default_rng(seed)

    def secs(s: float) -> int:
        return int(s * RATE)

    if kind == "thunder":
        n = secs(3.5)
        crack = shaped(rng, n, 300, 4000) * _env(n, 0.01, 0.12) * 0.5
        roll = shaped(rng, n, 20, 220, -3) * _env(n, 0.25, 1.1) * _lfo(n, 2.2, 0.5, rng)
        return crack + roll
    if kind == "gust":
        n = secs(1.4)
        swell = np.sin(np.pi * np.arange(n) / n) ** 2
        return shaped(rng, n, 80, 2200, -2) * swell.astype(np.float32)
    if kind == "flap":
        return _flap(rng)
    if kind == "shake":
        n = secs(0.6)
        out = np.zeros(n, np.float32)
        for k in range(6):
            f = _flap(rng, 0.6)[: secs(0.1)] * (1 - k / 7)
            at = secs(0.08 * k)
            out[at : at + len(f)] += f[: n - at]
        return out
    if kind in ("whoosh", "swell"):
        n = secs(0.9 if kind == "whoosh" else 2.0)
        shape = np.sin(np.pi * np.arange(n) / n) ** 3 if kind == "whoosh" else (np.arange(n) / n) ** 3
        return shaped(rng, n, 150, 5000, -1) * shape.astype(np.float32)
    if kind == "steps":
        n = secs(0.6)
        out = np.zeros(n, np.float32)
        for at in (0.0, 0.28):
            m = secs(0.08)
            tap = shaped(rng, m, 120, 2500) * _env(m, 0.002, 0.015)
            out[secs(at) : secs(at) + m] += tap
        return out
    if kind in ("rustle", "paper"):
        n = secs(0.5 if kind == "rustle" else 0.7)
        crackle = (rng.random(n) < (0.02 if kind == "paper" else 0.06)).astype(np.float32) * rng.uniform(
            -1, 1, n
        ).astype(np.float32)
        body = shaped(rng, n, 1200 if kind == "paper" else 900, 9000, -1)
        return (body * 0.5 + np.convolve(crackle, np.hanning(24), "same")) * np.sin(np.pi * np.arange(n) / n).astype(
            np.float32
        )
    if kind in ("block", "tap", "parcel"):
        n = secs(0.5)
        t = np.arange(n) / RATE
        f = {"block": 180, "tap": 900, "parcel": 110}[kind]
        thud = np.sin(2 * np.pi * f * t) * np.exp(-t / (0.05 if kind != "tap" else 0.02))
        click = shaped(rng, n, 500, 6000) * _env(n, 0.001, 0.008) * 0.3
        out = (thud + click).astype(np.float32)
        if kind == "block":  # the glow of the block: a soft glassy ping
            out += (np.sin(2 * np.pi * 1760 * t) * np.exp(-t / 0.18) * 0.18).astype(np.float32)
        return out
    if kind in ("hum", "ignite"):
        n = secs(1.6 if kind == "hum" else 0.9)
        t = np.arange(n) / RATE
        base = 110 if kind == "hum" else 220
        rise = 1 + (0.5 * t / t[-1] if kind == "ignite" else 0)
        tone = sum(np.sin(2 * np.pi * base * h * rise * t) / h for h in (1, 2, 3, 5))
        shimmer = shaped(rng, n, 3000, 12000) * 0.08
        return ((tone * 0.4 + shimmer) * np.sin(np.pi * t / t[-1]) ** (1 if kind == "hum" else 0.5)).astype(np.float32)
    if kind in ("chime", "coin"):
        n = secs(2.0 if kind == "chime" else 1.0)
        t = np.arange(n) / RATE
        partials = (1, 2.76, 5.4, 8.93) if kind == "chime" else (1, 2.31, 3.89, 5.6, 7.2)
        base = 1046 if kind == "chime" else 2350
        decay = 0.9 if kind == "chime" else 0.25
        return sum(np.sin(2 * np.pi * base * p * t) * np.exp(-t * p / decay) / p for p in partials).astype(np.float32)
    if kind in ("call", "chirp"):
        n = secs(1.2 if kind == "call" else 0.6)
        return _chirps(rng, n, 2 if kind == "call" else 4, 1400 if kind == "call" else 3600)
    if kind in ("splash", "splash-small"):
        n = secs(1.2 if kind == "splash" else 0.6)
        burst = shaped(rng, n, 300, 8000, -1) * _env(n, 0.005, 0.12 if kind == "splash" else 0.06)
        bubbles = _drops(rng, n, 40 if kind == "splash" else 15, 600, 2400, 0.012) * _env(n, 0.05, 0.3) * 0.5
        return burst + bubbles
    if kind == "train":  # a train passing: a rumble that swells and falls, wheel clacks, a squeal at its peak
        n = secs(3.0)
        t = np.arange(n) / RATE
        swell = np.exp(-(((t - 1.4) / 0.7) ** 2)).astype(np.float32)
        body = shaped(rng, n, 30, 900, -2) * swell
        clacks = np.zeros(n, np.float32)
        for at in np.arange(0.4, 2.6, 0.19):
            m = secs(0.04)
            clacks[secs(at) : secs(at) + m] += shaped(rng, m, 200, 2500) * _env(m, 0.001, 0.01)
        squeal = np.sin(2 * np.pi * (2400 - 300 * t) * t).astype(np.float32) * swell**3 * 0.12
        return body + clacks * swell + squeal
    if kind == "skate":  # wheels rolling on concrete, then the board's clack
        n = secs(0.9)
        roll = shaped(rng, n, 60, 900, -1) * _env(n, 0.05, 0.4)
        m = secs(0.12)
        clack = np.zeros(n, np.float32)
        clack[secs(0.7) : secs(0.7) + m] = (shaped(rng, m, 300, 4000) * _env(m, 0.001, 0.02))[: n - secs(0.7)]
        return roll * 0.7 + clack * 1.4
    if kind == "sparks":  # a spray of sparks: bright crackles that thin out
        n = secs(0.8)
        crackle = (rng.random(n) < 0.03 * np.exp(-np.arange(n) / secs(0.3))).astype(np.float32) * rng.uniform(
            -1, 1, n
        ).astype(np.float32)
        hiss = shaped(rng, n, 3000, 14000, 1) * _env(n, 0.002, 0.15) * 0.4
        sharp = np.convolve(
            crackle, np.array([1.0, -0.9], np.float32), "same"
        )  # a differentiator keeps the crackle bright
        return sharp + hiss
    if kind == "impact":  # a heavy hit: a sub thump falling in pitch and a short crack
        n = secs(1.3)
        t = np.arange(n) / RATE
        f = 70 * np.exp(-t * 2.5) + 28
        sub = np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-t * 3.2)
        crack = shaped(rng, n, 400, 5000) * _env(n, 0.001, 0.03) * 0.3
        return (sub + crack).astype(np.float32)
    if kind == "slide":  # a body sliding through water: a long wet hiss and a few splashes
        n = secs(1.0)
        hiss = shaped(rng, n, 500, 7000, -1) * np.sin(np.pi * np.arange(n) / n).astype(np.float32) ** 0.6
        return hiss * 0.8 + _drops(rng, n, 20, 600, 2200, 0.01) * 0.4
    if kind == "riser":  # noise rising in pitch and level into a cut
        n = secs(1.6)
        t = np.arange(n) / n
        out = np.zeros(n, np.float32)
        for k, lo in enumerate((300, 900, 2500)):
            band = shaped(rng, n, lo, lo * 4, 0)
            out += band * (t ** (3 - k)).astype(np.float32)
        return out
    raise FoleyError(f"{kind!r} is not an event")


def _normalise(x: np.ndarray) -> np.ndarray:
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def render(cues: list[dict], seconds: float, seed: int = 0) -> np.ndarray:
    """The foley stem of a film: ``cues`` are ``{kind, start, end, gain[, pan]}`` (seconds, dB, -1..1); stereo float32.

    A bed spans [start, end) with soft edges; an event starts at ``start``. Gains are relative: the mix sets the level.
    """
    n = int(seconds * RATE)
    out = np.zeros((n, 2), np.float32)
    for k, cue in enumerate(cues):
        kind = cue["kind"]
        if kind not in KINDS:
            raise FoleyError(f"unknown sound {kind!r}")
        start = int(cue["start"] * RATE)
        if start >= n:
            continue
        amp = 10 ** (cue.get("gain", -12) / 20)
        if kind in BEDS:
            length = max(1, int((cue["end"] - cue["start"]) * RATE))
            # A bed that continues across a cut keeps one seed per kind, so it does not jump between shots.
            sig = _normalise(bed(kind, length / RATE + 2 * BED_FADE, seed * 7 + KINDS.index(kind)))
            sig = sig[int(BED_FADE * RATE) // 2 :][:length]
            fade = min(int(BED_FADE * RATE), len(sig) // 2)
            if fade:
                sig[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                sig[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            pan = cue.get("pan", 0.0)
            width = 0.35  # a bed is wide: the two sides decorrelated by a few milliseconds
            delay = int(0.011 * RATE)
            left, right = (
                sig,
                np.concatenate([np.zeros(delay, np.float32), sig])[: len(sig)],
            )  # same length, even a sliver
            sides = np.stack([left * (1 - width * pan), right * (1 + width * pan)], axis=1)
        else:
            sig = _normalise(event(kind, seed * 131 + k))
            pan = cue.get("pan", 0.0)
            sides = np.stack([sig * np.sqrt(0.5 * (1 - pan)), sig * np.sqrt(0.5 * (1 + pan))], axis=1)
        end = min(n, start + len(sides))
        out[start:end] += sides[: end - start] * amp
    peak = np.max(np.abs(out))
    return out / peak * 0.89 if peak > 0.89 else out


def coverage(shots: list[dict], cues: list[dict]) -> dict:
    """Which shots have a sound of their own, and which character shots have an event (their action heard) — the gate's
    foley coverage. ``shots`` are ``{id, characters}``, ``cues`` carry the ``shot`` they belong to."""
    by_shot: dict[int, list[dict]] = {}
    for cue in cues:
        by_shot.setdefault(cue["shot"], []).append(cue)
    silent = [s["id"] for s in shots if not by_shot.get(s["id"])]
    unheard = [
        s["id"] for s in shots if s.get("characters") and not any(c["kind"] in EVENTS for c in by_shot.get(s["id"], []))
    ]
    covered = len(shots) - len(silent)
    return {
        "shots": len(shots),
        "covered": covered,
        "ratio": round(covered / max(1, len(shots)), 3),
        "silent": silent,
        "characterShotsUnheard": unheard,
    }
