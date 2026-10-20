"""Film titles and subtitles (orazaka#289): what a cinematic short writes on its picture, set like a film, not a caption bar.

The v1 cut drew its text with ``drawtext``: one face, one size, a hard box, words that popped on and off. A film sets its
text instead — this module writes an **ASS** script (rendered by libass through ffmpeg's ``ass`` filter):

* **The main title** — a pairing: the film's name in a tracked, uppercase display sans (Plus Jakarta Sans, the Krizaka
  display face) and its subtitle in a light italic serif (Cormorant Garamond, SIL OFL), centred in the title-safe area.
  It fades in slowly, the tracking opens a little while it holds (slow tracking), and it fades out before the cut.
* **Subtitles** — sentence case in the text face (Inter), at most two balanced lines of 42 characters, no box: a soft dark
  halo; a short fade; never under 1 s on screen; a start or an end within a quarter second of a cut lands on the cut.
* **Safe areas** — everything sits inside the title-safe area (10 % of each side); a vertical (9:16) frame also keeps
  clear of the platforms' own interface: subtitles stand above its bottom quarter, titles below its top tenth.
* **The label** — « AI-generated » in the language, small and quiet, top right inside the safe area (ADR-080 §5).

Everything but :func:`render` is pure. Faces are passed as a fonts folder (``fontsdir``): the worker ships none.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

#: The pairing of the film's text (family names as the font files declare them).
DISPLAY_FACE = "Plus Jakarta Sans SemiBold"
SERIF_FACE = "Cormorant Garamond Light"
TEXT_FACE = "Inter Medium"
LABEL_FACE = "Inter"

TITLE_SAFE = 0.10
#: A vertical frame's bottom quarter belongs to the platforms' buttons and captions; its top tenth to their header.
VERTICAL_BOTTOM_CLEAR = 0.24
VERTICAL_TOP_CLEAR = 0.10

MAX_LINE = 42
MIN_ON_SCREEN = 1.0
SNAP_TO_CUT = 0.25
SUB_FADE_MS = 160
TITLE_FADE_IN_MS = 1100
TITLE_FADE_OUT_MS = 800

KINDS = ("main", "subtitle", "label")


class TitleError(ValueError):
    """The text cannot be set as asked (the caller's to fix)."""


@dataclass(frozen=True)
class Text:
    start: float
    end: float
    text: str
    kind: str = "subtitle"
    #: The main title's second line (the serif subtitle), if any.
    sub: str = ""


@dataclass(frozen=True)
class Frame:
    width: int
    height: int

    @property
    def vertical(self) -> bool:
        return self.height > self.width

    def safe(self) -> dict:
        """The title-safe box: left, right, top and bottom margins in pixels."""
        side = round(self.width * TITLE_SAFE)
        top = round(self.height * (VERTICAL_TOP_CLEAR if self.vertical else TITLE_SAFE))
        bottom = round(self.height * (VERTICAL_BOTTOM_CLEAR if self.vertical else TITLE_SAFE))
        return {"left": side, "right": side, "top": top, "bottom": bottom}


def wrap(text: str, limit: int = MAX_LINE) -> list[str]:
    """One or two lines, balanced: the break that makes the two lines closest in length (pure)."""
    words = text.split()
    if len(text) <= limit or len(words) < 2:
        return [text.strip()]
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) > limit or len(b) > limit:
            continue
        score = abs(len(a) - len(b)) - (6 if re.search(r"[,.:;…]$", a) else 0)  # a break after punctuation reads best
        if best is None or score < best[0]:
            best = (score, [a, b])
    if best is None:
        raise TitleError(f"too long for two lines of {limit}: {text!r}")
    return best[1]


def on_cuts(texts: list[Text], cuts: list[float], tolerance: float = SNAP_TO_CUT) -> list[Text]:
    """Each subtitle's start and end moved onto a cut within ``tolerance``, and kept on screen at least 1 s (pure)."""
    out = []
    for t in texts:
        start, end = t.start, t.end
        if t.kind == "subtitle":
            near = [c for c in cuts if abs(c - start) <= tolerance]
            start = min(near, key=lambda c: abs(c - start)) if near else start
            near = [c for c in cuts if abs(c - end) <= tolerance and c > start]
            end = min(near, key=lambda c: abs(c - end)) - 0.04 if near else end
            end = max(end, start + MIN_ON_SCREEN)
        out.append(Text(round(start, 3), round(end, 3), t.text, t.kind, t.sub))
    return out


def _stamp(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


#: The main title's tracking (em): it opens slowly from START to END while the title holds.
TRACK_START = 0.30
TRACK_END = 0.38
#: The display face's mean advance of an uppercase letter, in em (Plus Jakarta Sans SemiBold, measured).
CAPS_ADVANCE = 0.68


def title_size(frame: Frame, text: str, wanted: int) -> int:
    """The main title's size: as asked, made smaller until its widest moment (full tracking) fits the safe width (pure)."""
    m = frame.safe()
    room = frame.width - m["left"] - m["right"]
    n = len(text)
    per_px = n * CAPS_ADVANCE + max(0, n - 1) * TRACK_END
    return max(8, min(wanted, int(room / per_px)))


def sizes(frame: Frame) -> dict:
    """Type sizes in pixels, from the frame's short side (a phone and a cinema screen read the same proportion)."""
    short = min(frame.width, frame.height)
    return {"title": round(short * 0.050), "serif": round(short * 0.105), "subtitle": round(short * (0.052 if frame.vertical else 0.044)),
            "label": round(short * 0.022)}


def script(frame: Frame, texts: list[Text], label: str = "") -> str:
    """The ASS script of a film's text for a frame (pure)."""
    for t in texts:
        if t.kind not in KINDS:
            raise TitleError(f"unknown text kind {t.kind!r}")
        if t.end <= t.start:
            raise TitleError(f"a text ends before it starts: {t.text!r}")
    s = sizes(frame)
    m = frame.safe()
    w, h = frame.width, frame.height
    styles = [
        # Name, Font, Size, Primary, Secondary, Outline, Back, Bold, Italic, Underline, Strike, ScaleX, ScaleY, Spacing,
        # Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
        f"Style: Title,{DISPLAY_FACE},{s['title']},&H00F4F1EA,&H00F4F1EA,&H64000000,&H00000000,0,0,0,0,100,100,"
        f"{round(s['title'] * 0.35)},0,1,0,0,5,{m['left']},{m['right']},0,1",
        f"Style: Serif,{SERIF_FACE},{s['serif']},&H00F4F1EA,&H00F4F1EA,&H64000000,&H00000000,0,1,0,0,100,100,0,0,1,0,0,5,"
        f"{m['left']},{m['right']},0,1",
        # The subtitle halo: a thin dark outline blurred into a soft shadow (\\blur in each event), never a box.
        f"Style: Sub,{TEXT_FACE},{s['subtitle']},&H00F2F2F2,&H00F2F2F2,&H8C000000,&H00000000,0,0,0,0,100,100,"
        f"{round(s['subtitle'] * 0.01, 2)},0,1,{max(1.5, s['subtitle'] * 0.06):.1f},0,2,{m['left']},{m['right']},{m['bottom']},1",
        f"Style: Label,{LABEL_FACE},{s['label']},&H59FFFFFF,&H59FFFFFF,&HB4000000,&H00000000,0,0,0,0,100,100,"
        f"{round(s['label'] * 0.08, 2)},0,1,1,0,9,{m['left']},{m['right']},{m['top']},1",
    ]
    events = []
    cx, cy = w // 2, (m["top"] + (h - m["bottom"])) // 2
    for t in sorted(texts, key=lambda x: x.start):
        start, end = _stamp(t.start), _stamp(t.end)
        if t.kind == "main":
            length_ms = round((t.end - t.start) * 1000)
            size = title_size(frame, t.text, s["title"])
            track = round(size * TRACK_END)
            gap = round(s["serif"] * 0.55)
            events.append(f"Dialogue: 1,{start},{end},Title,,0,0,0,,{{\\an5\\pos({cx},{cy - gap})\\fs{size}\\fsp{round(size * TRACK_START)}\\fad({TITLE_FADE_IN_MS},{TITLE_FADE_OUT_MS})"
                          f"\\blur0.6\\t(0,{length_ms},\\fsp{track})}}{_escape(t.text.upper())}")
            if t.sub:
                delay = 450  # the serif arrives after the name, as a title card breathes
                events.append(f"Dialogue: 1,{_stamp(t.start + delay / 1000)},{end},Serif,,0,0,0,,{{\\an5\\pos({cx},{cy + round(gap * 0.6)})"
                              f"\\fad({TITLE_FADE_IN_MS},{TITLE_FADE_OUT_MS})\\blur0.5}}{_escape(t.sub)}")
        elif t.kind == "subtitle":
            body = "\\N".join(_escape(line) for line in wrap(t.text))
            events.append(f"Dialogue: 0,{start},{end},Sub,,0,0,0,,{{\\fad({SUB_FADE_MS},{SUB_FADE_MS})\\blur3}}{body}")
        else:
            events.append(f"Dialogue: 0,{start},{end},Label,,0,0,0,,{_escape(t.text)}")
    if label:
        last = max((t.end for t in texts), default=1.0)
        events.append(f"Dialogue: 0,{_stamp(0)},{_stamp(last)},Label,,0,0,0,,{_escape(label)}")
    return "\n".join([
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 2", "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
        "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        *styles, "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text", *events, "",
    ])


def _filter_path(path: str) -> str:
    """A path inside an ffmpeg filter argument: its colons, quotes and backslashes escaped."""
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def render(video: str, ass_text: str, output: str, fontsdir: str, ffmpeg: str = "ffmpeg", crf: int = 16) -> dict:
    """Burns the script into the picture (libass), the sound copied. Returns the output's size and duration."""
    if not os.path.isdir(fontsdir):
        raise TitleError(f"the fonts folder {fontsdir} does not exist")
    ass_path = output + ".ass"
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_text)
    try:
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", video, "-vf",
                        f"ass=filename='{_filter_path(ass_path)}':fontsdir='{_filter_path(fontsdir)}'",
                        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "copy",
                        "-movflags", "+faststart", output], check=True, capture_output=True, text=True)
    finally:
        os.remove(ass_path)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
                            "-of", "csv=p=0:s=x", output], capture_output=True, text=True, check=True).stdout.split()
    width, height = (int(v) for v in probe[0].split("x")[:2])
    return {"width": width, "height": height, "durationSeconds": round(float(probe[-1]), 3)}
