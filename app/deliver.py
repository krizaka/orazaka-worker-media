"""A film's deliverables, cut from its master (orazaka#259): the 1080p file a player streams and the 9:16 cut of the
short-form platforms, each with its own language's captions and label.

The master is made once, at its best (``cut.py`` with ``master: "2160p"`` and ``text: false``: every shot upscaled to
4K by SeedVR2, no text drawn). A deliverable is then a cheap pass over it:

- ``2160p``: the master itself with its language's texts (laid out at 4K);
- ``1080p``: the master scaled down (lanczos);
- ``9:16``: a 9:16 window of the master's full height that follows each shot's subject — one horizontal centre per shot,
  given by the edit, clamped inside the frame, switching on the shot's first frame — scaled to 1080x1920.

Texts are drawn **after** the reframe, at the deliverable's own size: a caption laid out for 16:9 and then cropped to 9:16
would lose its ends. The texts and the label are the language's (the author's words go through files, never the filter
string, as in ``cut.text_filters``); the file's metadata says AI-generated like every cut.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

from app import cut

#: The deliverables this module makes: the aspect their texts are laid out for, and their frame.
DELIVERABLES = {
    "2160p": {"aspect": "16:9", "size": (3840, 2160), "master": "2160p"},
    "1080p": {"aspect": "16:9", "size": (1920, 1080)},
    "9:16": {"aspect": "9:16", "size": (1080, 1920)},
}


class DeliverError(ValueError):
    """A deliverable that cannot be made as asked — the caller's to fix."""


def _x(centre: float, master_width: int, window: int) -> int:
    return int(min(max(centre * master_width - window / 2, 0), master_width - window))


def plan(kind: str, master_size: tuple[int, int], spans: list[tuple[float, float]], centres: list[float]) -> dict:
    """How a deliverable is cut from the master: the crop (``None`` for none) and the frame (pure).

    ``spans`` are the shots' places on the film (``cut.timeline``); ``centres`` one horizontal centre per shot, 0..1 of the
    master's width. A 9:16 window switches to a shot's centre on the shot's first frame.
    """
    if kind not in DELIVERABLES:
        raise DeliverError(f"deliverable '{kind}' is not one of {', '.join(DELIVERABLES)}")
    spec = DELIVERABLES[kind]
    if spec["aspect"] == "16:9":
        return {"crop": None, "size": spec["size"], "aspect": spec["aspect"], "master": spec.get("master", "1080p")}
    if len(centres) != len(spans):
        raise DeliverError("a vertical cut needs one centre per shot")
    width, height = master_size
    window = int(height * 9 / 16)
    xs = [_x(c, width, window) for c in centres]
    expression = str(xs[-1])
    for (start, _end), x in reversed(list(zip(spans[1:], xs[:-1], strict=True))):
        expression = f"if(lt(t\\,{start:.3f})\\,{x}\\,{expression})"
    return {
        "crop": f"crop={window}:{height}:{expression}:0",
        "size": spec["size"],
        "aspect": spec["aspect"],
        "master": "1080p",
    }


def text_chain(
    layout: "cut.Edit", font: str, scratch: str, titles_ass: str | None = None, fontsdir: str | None = None
) -> list[str]:
    """The filters that set a deliverable's text (pure): the film's titles script (``app/titles.py``, libass) when one is
    given — set for this deliverable's own frame —, else the cut's drawtext layout of ``texts``."""
    if titles_ass:
        from app.titles import _filter_path  # noqa: PLC0415

        folder = f":fontsdir='{_filter_path(fontsdir)}'" if fontsdir else ""
        return [f"ass=filename='{_filter_path(titles_ass)}'{folder}"]
    return cut.text_filters(layout, [], font, scratch)


def deliver(
    master: str,
    output: str,
    kind: str,
    spans: list[tuple[float, float]],
    centres: list[float],
    texts: list[tuple[str, float, float, str]],
    label: str,
    font: str,
    provenance: dict | None = None,
    titles_ass: str | None = None,
    fontsdir: str | None = None,
    burn_label: bool = False,
) -> dict:
    """Makes one deliverable of ``master`` at ``output``; ``texts`` are (text, start, end, style) in the language's words.

    ``titles_ass`` (orazaka#289): an ASS script written for this deliverable's frame by ``titles.script`` replaces the
    drawtext layout — film titles, subtitles and the label set by libass, its faces in ``fontsdir``. The label is burnt
    only with ``burn_label`` (the players draw the disclosure; the metadata always carries it)."""
    probe = (
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                master,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .split(",")
    )
    step = plan(kind, (int(probe[0]), int(probe[1])), spans, centres)
    width, height = step["size"]
    # The texts reuse the cut's own layout at the deliverable's frame: an Edit of that aspect, its titles the texts.
    layout = cut.Edit(
        aspect=step["aspect"],
        fps=24,
        grade="none",
        shots=(),
        master=step["master"],
        titles=tuple(cut.Title(text=t, start=a, end=b, style=s) for t, a, b, s in texts),
        label=label,
        burn_label=burn_label,
    )
    scratch = tempfile.mkdtemp(prefix="orz-deliver-")
    try:
        chain = [step["crop"]] if step["crop"] else []
        chain += [f"scale={width}:{height}:flags=lanczos", "setsar=1"]
        chain += text_chain(layout, font, scratch, titles_ass, fontsdir)
        chain.append("format=yuv420p")
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            master,
            "-vf",
            ",".join(chain),
            "-map",
            "0:v",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-c:a",
            "copy",
            *cut.metadata_args(layout, provenance),
            "-movflags",
            "+faststart",
            os.path.abspath(output),
        ]
        subprocess.run(command, check=True, cwd=scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    duration = float(
        json.loads(
            subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", output],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )["format"]["duration"]
    )
    return {"kind": kind, "width": width, "height": height, "durationSeconds": round(duration, 3)}
