#!/usr/bin/env python3
"""Studio media composition (ADR-034 §7.2).

Assembles what a Studio run produced — photos, an optional generated clip, an
optional voiceover — into one vertical MP4. This is a **capability**, not Studio
code: the worker learns nothing about blueprints, installations or runs, it only
knows how to turn a list of assets into a video.

ffmpeg does the work, hardware-accelerated on Apple Silicon like the rest of this
worker. Nothing here reaches a network or a database.
"""
import json
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional

from PIL import Image, ImageOps

from app.encoder import is_apple_silicon

# 9:16 at 1080p — the aspect every short-form platform wants.
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920

# How long each still is held. Short enough to feel edited, long enough to read.
DEFAULT_SECONDS_PER_PHOTO = 3.0

# The output's frame rate. REPORTED to ffmpeg for the encode, never for the meter: ADR-063 forced
# `-r 30` so that `frames / fps` would divide, which is a meter reaching into the artefact to make
# itself work. Billing now measures the file's duration directly (ADR-066), so this is an encoding
# choice again — a constant rate is what short-form platforms want — and changing it changes the
# video, not the bill.
OUTPUT_FPS = 30

# A hard ceiling on inputs: the accelerator is the scarce resource on owned
# hardware, and an unbounded photo list is a self-inflicted outage (ADR-034 §7.1).
MAX_PHOTOS = 20


def _canvas(path: str, width: int, height: int) -> Image.Image:
    """Fits one photo onto the vertical canvas without distorting it.

    Letterboxed rather than cropped: a stretched or arbitrarily cropped property
    photo is worse than a bordered one, and the professional chose that framing.
    """
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        fitted = ImageOps.contain(image, (width, height))
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(
            fitted,
            ((width - fitted.width) // 2, (height - fitted.height) // 2),
        )
        return canvas


def _resolve_assets(asset_paths: List[str]) -> List[str]:
    """Keeps the readable inputs, in order, capped."""
    resolved = []
    for candidate in asset_paths[:MAX_PHOTOS]:
        if candidate and os.path.isfile(candidate):
            resolved.append(candidate)
    return resolved


def compose(
    photo_paths: List[str],
    output_path: str,
    audio_path: Optional[str] = None,
    seconds_per_photo: float = DEFAULT_SECONDS_PER_PHOTO,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> dict:
    """Composes photos (and an optional voiceover) into one vertical MP4.

    :param photo_paths: absolute paths of the stills, in the order they appear
    :param output_path: absolute path of the MP4 to write
    :param audio_path: absolute path of a voiceover to lay over it, if any
    :param seconds_per_photo: how long each still is held
    :param width: output width in pixels
    :param height: output height in pixels
    :returns: the measurements billing settles the hold against
    :raises ValueError: when no readable photo was supplied
    """
    photos = _resolve_assets(photo_paths)
    if not photos:
        raise ValueError("compose requires at least one readable photo")

    output_path = os.path.abspath(os.path.realpath(output_path))
    if os.path.basename(output_path).startswith("-"):
        raise ValueError("Output path filename cannot start with a hyphen.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    staging = tempfile.mkdtemp()
    try:
        listing = os.path.join(staging, "sequence.txt")
        with open(listing, "w", encoding="utf-8") as sequence:
            for index, photo in enumerate(photos):
                staged = os.path.join(staging, f"still_{index:04d}.png")
                _canvas(photo, width, height).save(staged, format="PNG")
                # ffmpeg's concat demuxer needs the last entry repeated without a
                # duration, or it drops the final still.
                sequence.write(f"file '{staged}'\nduration {seconds_per_photo}\n")
            sequence.write(f"file '{staged}'\n")

        codec = "h264_videotoolbox" if is_apple_silicon() else "libx264"
        command = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", listing,
        ]
        if audio_path and os.path.isfile(audio_path):
            command += ["-i", audio_path, "-c:a", "aac", "-shortest"]
        command += [
            "-c:v", codec,
            "-b:v", "8M",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={width}:{height}",
            "-r", str(OUTPUT_FPS),
            # The slideshow's length is a ceiling on the work, not the bill: at a constant rate
            # the repeated last entry would otherwise hold the final still one extra period.
            "-t", str(len(photos) * seconds_per_photo),
            "-movflags", "+faststart",
            output_path,
        ]

        print(f"[composer] {' '.join(command)}", flush=True)
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Measurements only, and measured on the file — which unit this bills in belongs to the
        # pricebook row the hold was pinned to, never to the executor (ADR-033). The request's
        # length is not the work: `-shortest` ends the file at a shorter voiceover, and billing
        # photos x seconds_per_photo billed 9.00 output seconds for a 2.02 s file (ADR-063).
        measured = _measure(output_path)
        measured["images"] = len(photos)
        return measured
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _measure(path: str) -> dict:
    """What the produced file is, read from the file.

    The DURATION is the measurement (ADR-066). It used to be a frame count over a rate, which is
    exact only when the rate is constant and whole — so the composer was made to force one, and
    an artefact was shaped to suit a meter. The container's duration is the same quantity without
    that condition, and a file whose duration ffprobe cannot read is reported as unmeasured: the
    hold is released rather than billed at a figure the file does not support, as an undecodable
    image already is.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration",
         "-of", "json", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True,
    )
    probed = json.loads(probe.stdout)
    stream = probed["streams"][0]
    try:
        seconds = float(probed["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        seconds = 0.0
    return {
        "durationSeconds": round(seconds, 3) if seconds > 0 else None,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }
