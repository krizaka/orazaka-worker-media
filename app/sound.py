"""Sound for a video: foley, a music bed and a voice-over, mixed, levelled and muxed (orazaka-packs#25).

Every clip this platform generated was silent. This module is the step that gives it sound, and it is
deliberately two things kept apart:

* **the mix** (this file, ffmpeg only, no model): lay the tracks over the picture, duck the bed under
  the voice, normalise the loudness to -14 LUFS (what the web players target), and mux AAC into the
  mp4 *without re-encoding the picture*. The silent original is kept beside it, for muted loops.
* **the generators** (``sound_generators``): the models that make the foley and the music bed. They
  are slow, heavy and optional; the mix is none of those and is tested without them.

Nothing here reaches a network or a database.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

#: What web players normalise to (YouTube, Spotify, TikTok sit within a LU of it): one level for every clip.
TARGET_LUFS = -14.0
#: True-peak ceiling, so AAC's own overshoot cannot clip on a phone speaker.
TRUE_PEAK_DB = -1.5
LOUDNESS_RANGE = 11.0

SAMPLE_RATE = 48000

#: Gains of the tracks before the loudness pass. Only their balance matters (the pass sets the level):
#: the voice leads, foley is the picture's sound, the music is a bed that sits under both.
VOICE_DB = 0.0
FOLEY_DB = -3.0
MUSIC_DB_ALONE = -6.0
MUSIC_DB_UNDER_FOLEY = -14.0
MUSIC_DB_UNDER_VOICE = -10.0

#: The bed ducks while the voice speaks: it falls by ``ratio`` once the voice passes ``threshold`` and
#: comes back over ``release`` ms. Numbers from the usual podcast/video bed, not from a model.
DUCK_THRESHOLD = 0.02
DUCK_RATIO = 20
DUCK_ATTACK_MS = 15
DUCK_RELEASE_MS = 400

FADE_IN_S = 0.4
FADE_OUT_S = 1.2

SILENT_SUFFIX = ".silent"


class SoundError(ValueError):
    """The request cannot be honoured as given — the caller's to fix, not the worker's fault."""


@dataclass(frozen=True)
class Tracks:
    """Which audio tracks (absolute paths of readable files) go into the mix. At least one."""

    foley: str | None = None
    music: str | None = None
    voice: str | None = None

    def present(self) -> list[tuple[str, str]]:
        return [(name, path) for name, path in
                (("foley", self.foley), ("music", self.music), ("voice", self.voice)) if path]


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)


def probe(path: str) -> dict:
    """Duration, whether there is a picture, and the audio codec if there is a sound — read from the file."""
    out = _run(["ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=codec_type,codec_name,width,height",
                "-of", "json", path]).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    try:
        seconds = float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        seconds = 0.0
    return {
        "durationSeconds": round(seconds, 3),
        "video": video["codec_name"] if video else None,
        "audio": audio["codec_name"] if audio else None,
        "width": video.get("width") if video else None,
        "height": video.get("height") if video else None,
    }


def _gain(db: float) -> str:
    return f"volume={db}dB"


def _conform(label_in: str, label_out: str, seconds: float, db: float, bed: bool) -> str:
    """One track, put on the picture's clock: stereo 48 kHz, cut or padded to the clip, levelled."""
    chain = [
        f"aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo",
        "asetpts=PTS-STARTPTS",
        f"atrim=0:{seconds:.3f}",
        _gain(db),
    ]
    if bed:
        fade_out = min(FADE_OUT_S, seconds / 2)
        chain += [f"afade=t=in:st=0:d={min(FADE_IN_S, seconds / 4):.3f}",
                  f"afade=t=out:st={max(seconds - fade_out, 0):.3f}:d={fade_out:.3f}"]
    chain.append(f"apad=whole_dur={seconds:.3f}")
    return f"[{label_in}]" + ",".join(chain) + f"[{label_out}]"


def mix_graph(tracks: Tracks, seconds: float) -> tuple[str, list[str]]:
    """The ffmpeg filter graph that mixes ``tracks`` for a clip of ``seconds``, and its input order.

    Returns ``(graph, order)`` where ``order`` lists the track names in the order they are passed as
    inputs ``1..n`` (input 0 is the picture). The graph ends on the label ``[mix]``.
    """
    present = tracks.present()
    if not present:
        raise SoundError("a sound needs at least one track: foley, music or a voice-over")
    order = [name for name, _ in present]
    index = {name: i + 1 for i, name in enumerate(order)}
    has_voice = "voice" in index
    has_foley = "foley" in index
    parts: list[str] = []

    if has_foley:
        parts.append(_conform(f"{index['foley']}:a", "foley", seconds, FOLEY_DB, bed=False))
    if "music" in index:
        db = MUSIC_DB_UNDER_FOLEY if has_foley else (MUSIC_DB_UNDER_VOICE if has_voice else MUSIC_DB_ALONE)
        parts.append(_conform(f"{index['music']}:a", "music", seconds, db, bed=True))
    if has_voice:
        parts.append(_conform(f"{index['voice']}:a", "voice", seconds, VOICE_DB, bed=False))

    bed = [label for label in ("foley", "music") if label in order]
    if len(bed) == 2:
        parts.append("[foley][music]amix=inputs=2:normalize=0:duration=longest[bed]")
    elif bed:
        parts.append(f"[{bed[0]}]anull[bed]")

    if has_voice and bed:
        # The voice is the sidechain that lowers the bed, and is also heard: split it.
        parts.append("[voice]asplit=2[voice_key][voice_main]")
        parts.append(f"[bed][voice_key]sidechaincompress=threshold={DUCK_THRESHOLD}:ratio={DUCK_RATIO}"
                     f":attack={DUCK_ATTACK_MS}:release={DUCK_RELEASE_MS}[ducked]")
        parts.append("[ducked][voice_main]amix=inputs=2:normalize=0:duration=longest[mix]")
    elif has_voice:
        parts.append("[voice]anull[mix]")
    else:
        parts.append("[bed]anull[mix]")
    return ";".join(parts), order


def _loudnorm_filter(measured: dict | None) -> str:
    base = f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK_DB}:LRA={LOUDNESS_RANGE}"
    if not measured:
        return base
    return (base + f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true")


_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)


def measure_loudnorm(path: str) -> dict:
    """First loudnorm pass: the file's integrated loudness, true peak, range, threshold and offset."""
    command = ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
               f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK_DB}:LRA={LOUDNESS_RANGE}:print_format=json",
               "-f", "null", "-"]
    err = _run(command).stderr
    blocks = _JSON_BLOCK.findall(err)
    if not blocks:
        raise RuntimeError("loudnorm printed no measurement")
    return json.loads(blocks[-1])


def integrated_loudness(path: str) -> float:
    """Integrated loudness of a file's audio in LUFS (EBU R128), read from the file."""
    err = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-af", "ebur128=peak=true",
                "-f", "null", "-"]).stderr
    found = re.findall(r"I:\s+(-?\d+(?:\.\d+)?)\s+LUFS", err)
    if not found:
        raise RuntimeError("ebur128 printed no integrated loudness")
    return float(found[-1])


def split_silent(video: str, silent_path: str) -> str:
    """Keeps the silent variant: the picture alone, copied bit for bit (no re-encode), for muted loops."""
    os.makedirs(os.path.dirname(os.path.abspath(silent_path)), exist_ok=True)
    _run(["ffmpeg", "-y", "-hide_banner", "-i", video, "-map", "0:v:0", "-an", "-c:v", "copy",
          "-movflags", "+faststart", silent_path])
    return silent_path


def silent_path_for(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    return f"{base}{SILENT_SUFFIX}{ext}"


def add_sound(video: str, tracks: Tracks, output_path: str, silent_path: str | None = None) -> dict:
    """Mixes ``tracks`` over ``video`` and writes ``output_path``: the picture copied, AAC at -14 LUFS.

    :param video: a readable mp4 — its picture is kept untouched, any sound it had is replaced
    :param tracks: the generated or supplied tracks; at least one
    :param output_path: where the clip with sound is written
    :param silent_path: where the picture alone is kept; defaults to ``<output>.silent.mp4``
    :returns: measurements read from the written files (duration, loudness, codecs, tracks used)
    """
    if not os.path.isfile(video):
        raise SoundError("the video to give sound to is not readable")
    source = probe(video)
    if not source["video"]:
        raise SoundError("the file has no picture")
    seconds = source["durationSeconds"]
    if seconds <= 0:
        raise SoundError("the video's duration could not be read")
    for name, path in tracks.present():
        if not os.path.isfile(path):
            raise SoundError(f"the {name} track is not readable")

    output_path = os.path.abspath(output_path)
    if os.path.basename(output_path).startswith("-"):
        raise SoundError("Output path filename cannot start with a hyphen.")
    if os.path.abspath(video) == output_path:
        raise SoundError("the output cannot overwrite the source video")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    graph, order = mix_graph(tracks, seconds)
    staging = tempfile.mkdtemp(prefix="orz-sound-")
    try:
        mixed = os.path.join(staging, "mix.wav")
        command = ["ffmpeg", "-y", "-hide_banner", "-i", video]
        for _, path in tracks.present():
            command += ["-i", path]
        command += ["-filter_complex", graph, "-map", "[mix]", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", mixed]
        _run(command)

        measured = measure_loudnorm(mixed)
        _run(["ffmpeg", "-y", "-hide_banner", "-i", video, "-i", mixed, "-map", "0:v:0", "-map", "1:a:0",
              "-c:v", "copy", "-af", _loudnorm_filter(measured), "-ar", str(SAMPLE_RATE),
              "-c:a", "aac", "-b:a", "192k", "-t", f"{seconds:.3f}", "-movflags", "+faststart", output_path])
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    silent = split_silent(video, silent_path or silent_path_for(output_path))
    result = probe(output_path)
    result.update({
        "loudnessLufs": round(integrated_loudness(output_path), 2),
        "targetLufs": TARGET_LUFS,
        "tracks": order,
        "silentPath": silent,
        "sourceHadAudio": source["audio"] is not None,
    })
    return result
