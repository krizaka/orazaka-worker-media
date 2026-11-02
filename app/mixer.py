"""The film mix (orazaka#289): a score that follows the story, a narrator with room around him, foley, at -14 LUFS.

The v1 mix put one score at full scale under the narrator (« far too aggressive ») and normalised the sum in one pass.
A film is mixed in stems instead:

* **music** — one cue per beat of the story (hook, journey, turn, payoff), each at the level its intensity asks
  (:func:`cue_gain`: never above -6 dB under the bed's ceiling), crossfaded at the beat changes, faded in and resolved
  at the end; then **ducked** under the narrator by a sidechain (the voice keys the compressor), 6–9 dB, released slowly;
* **voice** — each line placed where the timeline says, cleaned (high-pass, gentle compression), given **room tone**
  (a quiet early-reflection reverb and a low room bed) so it sits in the picture's space instead of on top of it;
* **foley** — the stem of ``app/foley.py``, ducked a little under the voice too;
* **master** — the stems summed, measured (EBU R128), gained to **-14 LUFS integrated** and limited to a **true peak of
  -1 dBTP** (oversampled peak), then measured again: :func:`master` refuses to return a file outside ±0.5 LU / above -1.

ffmpeg does the signal work; the decisions (gains, envelopes, graphs) are pure and tested without it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass

TARGET_LUFS = -14.0
TOLERANCE_LU = 0.5
TRUE_PEAK_DBTP = -1.0
#: The limiter's ceiling sits under the target true peak: an inter-sample peak can exceed the sample peak.
LIMIT_CEILING_DBTP = -1.6
RATE = 48000

#: A cue at intensity 1 sits at MUSIC_CEILING_DB; at 0, MUSIC_FLOOR_DB (before the master gain, relative to the voice at 0).
MUSIC_CEILING_DB = -9.0
MUSIC_FLOOR_DB = -24.0
CUE_CROSSFADE_S = 1.5

#: The duck: how far the music falls under the voice, and how slowly it comes back (a film mix breathes).
DUCK_THRESHOLD = 0.03
DUCK_RATIO = 4.0
DUCK_ATTACK_MS = 60
DUCK_RELEASE_MS = 900
FOLEY_DUCK_RATIO = 2.0

VOICE_DB = 0.0
FOLEY_DB = -6.0
ROOM_TONE_DB = -46.0


class MixError(ValueError):
    """The mix cannot be made as asked, or its master missed the loudness contract."""


@dataclass(frozen=True)
class Cue:
    """One music cue: its file, where it plays in the film, its intensity (0..1)."""

    path: str
    start: float
    end: float
    intensity: float
    #: How this cue enters over the one before it: a long crossfade inside a mood, a short one on a hard genre switch
    #: cut on the picture (a journey that changes city and music at once, orazaka#289).
    fade: float = 1.5


@dataclass(frozen=True)
class Line:
    path: str
    at: float


def cue_gain(intensity: float) -> float:
    """A cue's level in dB from its intensity (pure): linear in dB between the floor and the ceiling."""
    i = min(1.0, max(0.0, intensity))
    return round(MUSIC_FLOOR_DB + (MUSIC_CEILING_DB - MUSIC_FLOOR_DB) * i, 2)


def music_graph(cues: list[Cue], seconds: float, first_input: int) -> tuple[str, list[str]]:
    """The filter graph that lays the cues end to end on the film's clock, each at its gain, crossfaded at the beat
    changes, a fade in at the start and a 3 s resolve at the end. Returns (graph ending on [music], input files)."""
    if not cues:
        raise MixError("a score needs at least one cue")
    ordered = sorted(cues, key=lambda c: c.start)
    parts, files = [], []
    for k, c in enumerate(ordered):
        following = ordered[k + 1].fade if k < len(ordered) - 1 else 0.0
        length = c.end - c.start + following / 2 + (c.fade / 2 if k else 0)
        fade_in = 2.0 if k == 0 else c.fade
        fade_out = following if k < len(ordered) - 1 else min(3.0, length / 2)
        delay = int(max(0.0, c.start - (0 if k == 0 else c.fade / 2)) * 1000)
        parts.append(
            f"[{first_input + k}:a]aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo,atrim=0:{length:.3f},"
            f"asetpts=PTS-STARTPTS,volume={cue_gain(c.intensity)}dB,afade=t=in:st=0:d={fade_in:.2f},"
            f"afade=t=out:st={max(0.0, length - fade_out):.3f}:d={fade_out:.2f},adelay={delay}|{delay}[c{k}]"
        )
        files.append(c.path)
    labels = "".join(f"[c{k}]" for k in range(len(ordered)))
    parts.append(
        f"{labels}amix=inputs={len(ordered)}:normalize=0:duration=longest,apad=whole_dur={seconds:.3f},"
        f"atrim=0:{seconds:.3f}[music]"
    )
    return ";".join(parts), files


def voice_graph(lines: list[Line], seconds: float, first_input: int) -> tuple[str, list[str]]:
    """The narrator: each line at its time, cleaned and given room (early reflections and a room bed). Ends on [voice]."""
    if not lines:
        return f"anullsrc=r={RATE}:cl=stereo,atrim=0:{seconds:.3f}[voice]", []
    parts, files = [], []
    for k, line in enumerate(lines):
        ms = int(line.at * 1000)
        parts.append(
            f"[{first_input + k}:a]aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo,"
            f"highpass=f=80,acompressor=threshold=0.125:ratio=2.5:attack=8:release=120,adelay={ms}|{ms}[v{k}]"
        )
        files.append(line.path)
    labels = "".join(f"[v{k}]" for k in range(len(lines)))
    parts.append(
        f"{labels}amix=inputs={len(lines)}:normalize=0,apad=whole_dur={seconds:.3f},atrim=0:{seconds:.3f},"
        # Room tone: short early reflections (a small room, not a hall) mixed low under the dry voice.
        f"aecho=0.9:0.35:23|37|53:0.18|0.12|0.08,volume={VOICE_DB}dB[voice_dry]"
    )
    parts.append(
        f"anoisesrc=r={RATE}:c=brown:a=1,lowpass=f=300,volume={ROOM_TONE_DB}dB,aformat=channel_layouts=stereo,"
        f"atrim=0:{seconds:.3f}[room]"
    )
    parts.append("[voice_dry][room]amix=inputs=2:normalize=0:duration=first[voice]")
    return ";".join(parts), files


def mix_graph(cues: list[Cue], lines: list[Line], foley: str | None, seconds: float) -> tuple[str, list[str]]:
    """The whole mix (pure): music ducked by the voice, foley ducked a little, the sum on [mix]. Returns (graph, inputs)."""
    files: list[str] = []
    parts: list[str] = []
    graph, used = music_graph(cues, seconds, 0)
    parts.append(graph)
    files += used
    graph, used = voice_graph(lines, seconds, len(files))
    parts.append(graph)
    files += used
    parts.append("[voice]asplit=3[vmain][vkey1][vkey2]")
    parts.append(
        f"[music][vkey1]sidechaincompress=threshold={DUCK_THRESHOLD}:ratio={DUCK_RATIO}:attack={DUCK_ATTACK_MS}"
        f":release={DUCK_RELEASE_MS}:makeup=1[ducked]"
    )
    if foley:
        parts.append(
            f"[{len(files)}:a]aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo,apad=whole_dur={seconds:.3f},"
            f"atrim=0:{seconds:.3f},volume={FOLEY_DB}dB[fx]"
        )
        files.append(foley)
        parts.append(
            f"[fx][vkey2]sidechaincompress=threshold={DUCK_THRESHOLD}:ratio={FOLEY_DUCK_RATIO}:attack=40:release=600[fxd]"
        )
        parts.append("[ducked][fxd][vmain]amix=inputs=3:normalize=0[mix]")
    else:
        parts.append("[vkey2]anullsink")
        parts.append("[ducked][vmain]amix=inputs=2:normalize=0[mix]")
    return ";".join(parts), files


_R128 = re.compile(r"I:\s+(-?[\d.]+) LUFS.*?LRA:\s+([\d.]+) LU.*?Peak:\s+(-?[\d.inf]+) dBFS", re.S)


def measure(path: str, ffmpeg: str = "ffmpeg") -> dict:
    """Integrated loudness (LUFS), loudness range (LU) and true peak (dBTP, 4x oversampled) of a file's audio."""
    err = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", path, "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stderr
    summary = err[err.rfind("Summary:") :]
    found = _R128.search(summary)
    if not found:
        raise MixError("ebur128 printed no summary")
    peak = found.group(3)
    return {
        "lufs": float(found.group(1)),
        "lra": float(found.group(2)),
        "truePeak": -120.0 if "inf" in peak else float(peak),
    }


def master_gain(measured_lufs: float) -> float:
    """The gain that brings a measured mix to the target (pure)."""
    return round(TARGET_LUFS - measured_lufs, 2)


def within_contract(m: dict) -> bool:
    """-14 LUFS ± 0.5 LU and a true peak at or under -1 dBTP (pure)."""
    return abs(m["lufs"] - TARGET_LUFS) <= TOLERANCE_LU and m["truePeak"] <= TRUE_PEAK_DBTP


def master(
    cues: list[Cue], lines: list[Line], foley: str | None, seconds: float, output: str, ffmpeg: str = "ffmpeg"
) -> dict:
    """Mixes and masters to ``output`` (wav or m4a): the stems summed, gained to -14 LUFS, true-peak limited; measured
    after, and refused if outside the contract. Returns the measurement and the gains used."""
    graph, files = mix_graph(cues, lines, foley, seconds)
    staging = output + ".mix.wav"
    command = [ffmpeg, "-v", "error", "-y"]
    for f in files:
        command += ["-i", f]
    subprocess.run(
        command + ["-filter_complex", graph, "-map", "[mix]", "-ar", str(RATE), "-c:a", "pcm_f32le", staging],
        check=True,
        capture_output=True,
        text=True,
    )
    first = measure(staging, ffmpeg)
    gain = master_gain(first["lufs"])
    ceiling = 10 ** (LIMIT_CEILING_DBTP / 20)
    final = None
    attempts = 0
    while attempts < 3:
        attempts += 1
        # The limiter takes back a little loudness on peaky mixes: measure, then nudge the gain by what was lost.
        chain = (
            f"volume={gain}dB,aresample={RATE * 4},alimiter=limit={ceiling:.4f}:attack=2:release=60:level=disabled,"
            f"aresample={RATE}"
        )
        codec = ["-c:a", "aac", "-b:a", "256k"] if output.endswith((".m4a", ".mp4")) else ["-c:a", "pcm_s24le"]
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-i", staging, "-af", chain, *codec, output],
            check=True,
            capture_output=True,
            text=True,
        )
        final = measure(output, ffmpeg)
        if within_contract(final):
            break
        gain = round(gain + (TARGET_LUFS - final["lufs"]), 2)
    os.remove(staging)
    if not final or not within_contract(final):
        raise MixError(f"the master missed the contract: {json.dumps(final)}")
    return {"premix": first, "gainDb": gain, "master": final, "attempts": attempts}
