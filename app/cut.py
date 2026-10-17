"""The cut of a cinematic short (ADR-080, krizaka/orazaka): shots in, one edited film out.

A **capability**, like ``composer.py``: the worker knows nothing of shot lists, briefs or Studios. It is handed
clips (files this host can read, in order) and an **edit decision list** — which part of each clip, at what speed,
how it hands over to the next one, the colour grade, the titles and captions — and it makes the film with ffmpeg:

1. each shot is trimmed, slowed or ramped, interpolated to the output rate (Wan makes 16 fps; motion-compensated
   interpolation makes 30 without doubling frames), pushed in when asked, scaled to 1080p and graded through a 3D LUT;
2. the shots are joined by ``xfade`` (a hard cut is a one-frame fade, so one chain does both);
3. titles, captions and the **AI-generated label** are drawn on the timeline, and the label is also written in the
   file's metadata (``comment``, ``description``), which a later mux (the soundtrack) carries over.

The edit decision list is checked here, once, at the edge: an edit this module cannot make is refused with what is
wrong, never made differently. Everything that decides a shape (the filter graph, the LUT, the timeline) is a pure
function, tested without ffmpeg.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

#: The output frame size per aspect: 1080p, the size every short-form platform and a 16:9 player expect.
OUTPUT_SIZES = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}

#: Technical ceilings of one cut on one accelerator (the worker's budget, like composer.MAX_PHOTOS), not business rules.
MAX_SHOTS = 24
MAX_SECONDS = 120.0
MIN_SHOT_SECONDS = 0.3
MAX_TRANSITION_SECONDS = 1.5
MAX_TITLES = 12
MAX_TEXT = 160

#: The handovers this cut knows, and the ffmpeg ``xfade`` transition each one is (ffmpeg 4.4 has them all).
TRANSITIONS = {
    "cut": "fade",  # a one-frame fade: a hard cut inside the same chain
    "fade": "fade",
    "dissolve": "dissolve",
    "black": "fadeblack",
    "flash": "fadewhite",
    "whip": "hblur",
    "slide": "smoothleft",
    "radial": "radial",
    "iris": "circleopen",
}

#: The grades. Each is a parametric look baked into a 17-point 3D LUT (``grade_cube``), applied with ``lut3d``.
GRADES = ("none", "teal-orange", "neon-night", "warm-film", "clean-news", "nordic")

#: Text styles: size as a fraction of the frame's short side, and where the block sits (fraction of the height).
TEXT_STYLES = {
    "hero": {"size": 0.085, "y": 0.42, "box": False, "border": 4},
    "kicker": {"size": 0.034, "y": 0.10, "box": True, "border": 0},
    "caption": {"size": 0.045, "y": 0.74, "box": False, "border": 3},
    "lower-third": {"size": 0.038, "y": 0.80, "box": True, "border": 0},
    "end": {"size": 0.055, "y": 0.45, "box": False, "border": 3},
}

DEFAULT_LABEL = "AI-generated"

#: The upscalers the cut knows: SeedVR2 3B (Apache-2.0), the one measured best by the showcase comparison.
UPSCALERS = ("none", "seedvr2")
SEEDVR2_DIT = "seedvr2_ema_3b_fp16.safetensors"

#: Synthesized ambience beds (ffmpeg ``anoisesrc`` through filters — no model, no licence): what each sounds like.
AMBIENCES = {
    "none": None,
    "rain": "anoisesrc=color=pink:amplitude=0.5,highpass=f=500,lowpass=f=9000,volume=-14dB",
    "storm": "anoisesrc=color=pink:amplitude=0.6,highpass=f=300,lowpass=f=8000,volume=-12dB",
    "wind": "anoisesrc=color=brown:amplitude=0.7,lowpass=f=700,tremolo=f=0.25:d=0.6,volume=-16dB",
    "city": "anoisesrc=color=brown:amplitude=0.6,lowpass=f=400,volume=-18dB",
    "room": "anoisesrc=color=pink:amplitude=0.2,lowpass=f=2000,volume=-30dB",
    "nature": "anoisesrc=color=pink:amplitude=0.3,highpass=f=1500,lowpass=f=6000,tremolo=f=0.6:d=0.4,volume=-24dB",
}

#: Fonts tried in order when ``MEDIA_FONT_FILE`` names none: the GPU pod's DejaVu, then the Mac's.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
)


class EditError(ValueError):
    """The edit decision list cannot be made as written — the caller's to fix."""


@dataclass(frozen=True)
class Shot:
    clip: int
    start: float
    end: float
    speed: float = 1.0
    ramp_at: float | None = None
    ramp_speed: float | None = None
    zoom: float = 1.0
    transition: str = "cut"
    transition_seconds: float = 0.0
    caption: str = ""

    @property
    def seconds(self) -> float:
        """How long the shot lasts on the timeline, once slowed or ramped."""
        if self.ramp_at is None:
            return (self.end - self.start) / self.speed
        before = (self.ramp_at - self.start) / self.speed
        return before + (self.end - self.ramp_at) / float(self.ramp_speed)


@dataclass(frozen=True)
class Title:
    text: str
    start: float
    end: float
    style: str = "hero"


@dataclass(frozen=True)
class Edit:
    aspect: str
    fps: int
    grade: str
    shots: tuple
    titles: tuple = field(default_factory=tuple)
    label: str = DEFAULT_LABEL
    interpolate: bool = True
    #: Super-resolution of each source clip before the cut (``seedvr2``: SeedVR2 3B, Apache-2.0) or ``none`` (lanczos).
    upscale: str = "none"
    #: The synthesized foley bed under the film (no model: filtered noise and tones): ``none`` or an ambience.
    ambience: str = "none"
    #: Whooshes on transitions and impacts on ramps and flashes, synthesized.
    sfx: bool = False

    @property
    def size(self) -> tuple[int, int]:
        return OUTPUT_SIZES[self.aspect]


def _number(value, name: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise EditError(f"{name} must be a number") from None
    if not low <= number <= high:
        raise EditError(f"{name} must be between {low:g} and {high:g}, not {number:g}")
    return number


def _text(value, name: str) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > MAX_TEXT:
        raise EditError(f"{name} is longer than {MAX_TEXT} characters")
    return text


def parse_edit(raw, clip_count: int) -> Edit:
    """The edit decision list, checked: a dict or its JSON text in, an :class:`Edit` out.

    Raises:
        EditError: naming the first thing that cannot be made.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as bad:
            raise EditError(f"the edit is not JSON ({bad.msg})") from None
    if not isinstance(raw, dict):
        raise EditError("the edit must be an object")
    aspect = str(raw.get("aspect") or "9:16")
    if aspect not in OUTPUT_SIZES:
        raise EditError(f"aspect must be one of {', '.join(OUTPUT_SIZES)}")
    fps = int(_number(raw.get("fps", 30), "fps", 12, 60))
    grade = str(raw.get("grade") or "none")
    if grade not in GRADES:
        raise EditError(f"grade must be one of {', '.join(GRADES)}")
    shots_raw = raw.get("shots")
    if not isinstance(shots_raw, list) or not shots_raw:
        raise EditError("the edit has no shots")
    if len(shots_raw) > MAX_SHOTS:
        raise EditError(f"a cut holds at most {MAX_SHOTS} shots")
    shots = []
    for i, item in enumerate(shots_raw):
        if not isinstance(item, dict):
            raise EditError(f"shot {i} must be an object")
        where = f"shot {i}"
        clip = int(_number(item.get("clip", i), f"{where}.clip", 0, clip_count - 1))
        start = _number(item.get("in", 0), f"{where}.in", 0, 600)
        end = _number(item.get("out", start + 5), f"{where}.out", start + MIN_SHOT_SECONDS, 600)
        speed = _number(item.get("speed", 1), f"{where}.speed", 0.25, 4)
        ramp_at = item.get("rampAt")
        ramp_speed = None
        if ramp_at is not None:
            ramp_at = _number(ramp_at, f"{where}.rampAt", start + 0.1, end - 0.1)
            ramp_speed = _number(item.get("rampSpeed", 0.5), f"{where}.rampSpeed", 0.25, 4)
        transition = str(item.get("transition") or "cut")
        if transition not in TRANSITIONS:
            raise EditError(f"{where}.transition must be one of {', '.join(TRANSITIONS)}")
        seconds = 0.0 if transition == "cut" else _number(
            item.get("transitionSeconds", 0.4), f"{where}.transitionSeconds", 0.1, MAX_TRANSITION_SECONDS)
        shots.append(Shot(clip=clip, start=start, end=end, speed=speed, ramp_at=ramp_at, ramp_speed=ramp_speed,
                          zoom=_number(item.get("zoom", 1), f"{where}.zoom", 1, 1.5), transition=transition,
                          transition_seconds=seconds, caption=_text(item.get("caption"), f"{where}.caption")))
    # A transition is the handover INTO a shot; it must fit inside both shots it joins.
    for i, shot in enumerate(shots[1:], start=1):
        if shot.transition_seconds >= min(shot.seconds, shots[i - 1].seconds) / 2:
            raise EditError(f"shot {i}.transitionSeconds is longer than half of a shot it joins")
    titles_raw = raw.get("titles") or []
    if not isinstance(titles_raw, list) or len(titles_raw) > MAX_TITLES:
        raise EditError(f"titles must be a list of at most {MAX_TITLES}")
    titles = []
    for i, item in enumerate(titles_raw):
        style = str(item.get("style") or "hero")
        if style not in TEXT_STYLES:
            raise EditError(f"title {i}.style must be one of {', '.join(TEXT_STYLES)}")
        start = _number(item.get("start", 0), f"title {i}.start", 0, MAX_SECONDS)
        titles.append(Title(text=_text(item.get("text"), f"title {i}.text"), start=start,
                            end=_number(item.get("end", start + 2), f"title {i}.end", start + 0.2, MAX_SECONDS),
                            style=style))
    upscale = str(raw.get("upscale") or "none")
    if upscale not in UPSCALERS:
        raise EditError(f"upscale must be one of {', '.join(UPSCALERS)}")
    ambience = str(raw.get("ambience") or "none")
    if ambience not in AMBIENCES:
        raise EditError(f"ambience must be one of {', '.join(AMBIENCES)}")
    edit = Edit(aspect=aspect, fps=fps, grade=grade, shots=tuple(shots), titles=tuple(titles),
                label=_text(raw.get("label") or DEFAULT_LABEL, "label") or DEFAULT_LABEL,
                interpolate=bool(raw.get("interpolate", True)), upscale=upscale, ambience=ambience,
                sfx=bool(raw.get("sfx", False)))
    if timeline(edit)[-1][1] > MAX_SECONDS:
        raise EditError(f"the cut is longer than {MAX_SECONDS:g} s")
    return edit


def timeline(edit: Edit) -> list[tuple[float, float]]:
    """Where each shot starts and ends on the finished film, transitions overlapping (pure)."""
    spans, length = [], 0.0
    for i, shot in enumerate(edit.shots):
        overlap = _handover(edit, i)
        start = max(0.0, length - overlap)
        spans.append((round(start, 3), round(start + shot.seconds, 3)))
        length = start + shot.seconds
    return spans


def _handover(edit: Edit, i: int) -> float:
    """Seconds shot ``i`` overlaps the one before it (a hard cut overlaps one frame)."""
    if i == 0:
        return 0.0
    shot = edit.shots[i]
    return shot.transition_seconds if shot.transition != "cut" else 1.0 / edit.fps


def shot_filter(edit: Edit, shot: Shot) -> str:
    """The filter chain that turns one trimmed clip into one finished shot (pure).

    Speed first (a ramp is a time map: unchanged until ``rampAt``, then slowed or sped), then motion-compensated
    interpolation to the output rate, then the push-in, the 1080p frame, the grade and a light vignette.
    """
    width, height = edit.size
    parts = []
    if shot.ramp_at is not None:
        at = shot.ramp_at - shot.start
        parts.append(
            f"setpts='if(lt(T,{at:.3f}),PTS/{shot.speed:g},({at:.3f}/{shot.speed:g}+(T-{at:.3f})/{shot.ramp_speed:g})/TB)'"
        )
    elif shot.speed != 1:
        parts.append(f"setpts=PTS/{shot.speed:g}")
    if edit.interpolate:
        parts.append(f"minterpolate=fps={edit.fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    else:
        parts.append(f"fps={edit.fps}")
    if shot.zoom > 1:
        grow = shot.zoom - 1
        parts.append(
            f"scale=w='{width}*(1+{grow:g}*t/{shot.seconds:.3f})':h='{height}*(1+{grow:g}*t/{shot.seconds:.3f})'"
            ":eval=frame:flags=lanczos"
        )
        parts.append(f"crop={width}:{height}")
    else:
        parts.append(f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos")
        parts.append(f"crop={width}:{height}")
    if edit.grade != "none":
        parts.append("lut3d=file=grade.cube:interp=tetrahedral")
        parts.append("vignette=angle=PI/5")
    parts.append("format=yuv420p")
    parts.append("setsar=1")
    return ",".join(parts)


def join_filter(edit: Edit, durations: list[float]) -> str:
    """The ``xfade`` chain that joins the shot files ``[0:v]…[n:v]`` into ``[joined]`` (pure)."""
    if len(durations) == 1:
        return "[0:v]settb=AVTB,fps={0}[joined]".format(edit.fps)
    parts = [f"[{i}:v]settb=AVTB,fps={edit.fps}[s{i}]" for i in range(len(durations))]
    current, length = "s0", durations[0]
    for i in range(1, len(durations)):
        shot = edit.shots[i]
        overlap = _handover(edit, i)
        offset = max(0.0, length - overlap)
        out = "joined" if i == len(durations) - 1 else f"j{i}"
        parts.append(
            f"[{current}][s{i}]xfade=transition={TRANSITIONS[shot.transition]}:duration={overlap:.3f}"
            f":offset={offset:.3f}[{out}]"
        )
        current, length = out, offset + durations[i]
    return ";".join(parts)


def _wrap(text: str, width: int, size: int) -> list[str]:
    """Lines short enough for the frame: about 0.55 em per character of a bold sans."""
    per_line = max(8, int(width * 0.86 / (size * 0.55)))
    return textwrap.wrap(text, per_line)[:4]


def text_filters(edit: Edit, spans: list[tuple[float, float]], font: str, scratch: str) -> list[str]:
    """``drawtext`` filters for the titles, the captions and the label, each line its own centred filter.

    Texts go through files (``textfile``), never the filter string: a caption is the author's words and an
    apostrophe must not become filter syntax.
    """
    width, height = edit.size
    short = min(width, height)
    items = [(t.text, t.start, t.end, t.style) for t in edit.titles if t.text]
    items += [(s.caption, a, b, "caption") for s, (a, b) in zip(edit.shots, spans, strict=True) if s.caption]
    filters, n = [], 0
    for text, start, end, style in items:
        spec = TEXT_STYLES[style]
        size = max(18, int(short * spec["size"]))
        lines = _wrap(text, width, size)
        top = int(height * spec["y"]) - (len(lines) * int(size * 1.25)) // 2
        for k, line in enumerate(lines):
            path = os.path.join(scratch, f"text-{n}.txt")
            n += 1
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(line)
            look = (f":box=1:boxcolor=black@0.55:boxborderw={max(6, size // 4)}" if spec["box"]
                    else f":borderw={spec['border']}:bordercolor=black@0.85")
            filters.append(
                f"drawtext=fontfile='{font}':textfile='{path}':fontcolor=white:fontsize={size}"
                f":x=(w-text_w)/2:y={top + k * int(size * 1.25)}{look}"
                f":enable='between(t,{start:.3f},{end:.3f})'"
            )
    label_path = os.path.join(scratch, "label.txt")
    with open(label_path, "w", encoding="utf-8") as handle:
        handle.write(edit.label)
    label_size = max(14, int(short * 0.022))
    filters.append(
        f"drawtext=fontfile='{font}':textfile='{label_path}':fontcolor=white@0.85:fontsize={label_size}"
        f":x=w-text_w-{label_size}:y={label_size}:box=1:boxcolor=black@0.35:boxborderw={label_size // 3}"
    )
    return filters


def _clamp(value: float) -> float:
    return 0.0 if value < 0 else 1.0 if value > 1 else value


#: Each grade: (shadow tint RGB, highlight tint RGB, contrast, saturation, lift). Tints are added, scaled by how
#: dark or bright the pixel is; contrast is an S-curve around mid-grey; lift raises the blacks (a filmic toe).
_LOOKS = {
    "none": ((0, 0, 0), (0, 0, 0), 1.0, 1.0, 0.0),
    "teal-orange": ((-0.05, 0.03, 0.07), (0.07, 0.02, -0.06), 1.12, 1.08, 0.01),
    "neon-night": ((0.02, -0.03, 0.09), (0.05, -0.01, 0.05), 1.18, 1.22, 0.0),
    "warm-film": ((0.03, 0.01, -0.02), (0.06, 0.03, -0.04), 0.94, 0.95, 0.035),
    "clean-news": ((0.0, 0.0, 0.01), (0.01, 0.01, 0.0), 1.05, 1.04, 0.0),
    "nordic": ((-0.02, 0.0, 0.04), (0.0, 0.01, 0.03), 1.06, 0.88, 0.02),
}


def grade_rgb(name: str, r: float, g: float, b: float) -> tuple[float, float, float]:
    """One colour through a grade (pure): split-toning, S-curve contrast, saturation, lifted blacks."""
    shadow, light, contrast, saturation, lift = _LOOKS[name]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    ws, wh = (1 - luma) ** 2, luma**2
    rgb = [c + ws * s + wh * h for c, s, h in zip((r, g, b), shadow, light, strict=True)]
    rgb = [0.5 + (c - 0.5) * contrast for c in rgb]
    grey = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    rgb = [grey + (c - grey) * saturation for c in rgb]
    rgb = [lift + c * (1 - lift) for c in rgb]
    return tuple(_clamp(c) for c in rgb)


def grade_cube(name: str, points: int = 17) -> str:
    """The grade as a ``.cube`` 3D LUT (pure): ``LUT_3D_SIZE`` then one RGB line per point, red fastest."""
    if name not in _LOOKS:
        raise EditError(f"unknown grade '{name}'")
    lines = [f'TITLE "orazaka {name}"', f"LUT_3D_SIZE {points}"]
    step = 1.0 / (points - 1)
    for bi in range(points):
        for gi in range(points):
            for ri in range(points):
                r, g, b = grade_rgb(name, ri * step, gi * step, bi * step)
                lines.append(f"{r:.6f} {g:.6f} {b:.6f}")
    return "\n".join(lines) + "\n"


def missing_filters() -> list[str]:
    """The filters this host's ffmpeg lacks for a cut (Homebrew's default build has no ``drawtext``)."""
    listed = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, check=False).stdout
    return [name for name in ("drawtext", "xfade", "lut3d", "minterpolate") if f" {name} " not in listed]


def sfx_events(edit: Edit, spans: list[tuple[float, float]]) -> list[tuple[str, float]]:
    """Where the synthesized hits fall on the timeline (pure): a whoosh into every non-cut transition, an impact on
    every flash and on every speed ramp (the hero moment)."""
    events: list[tuple[str, float]] = []
    if not edit.sfx:
        return events
    for shot, (start, _end) in zip(edit.shots, spans, strict=True):
        if shot.transition not in ("cut", "fade", "black"):
            events.append(("whoosh", max(0.0, start - 0.3)))
        if shot.transition == "flash":
            events.append(("impact", start))
        if shot.ramp_at is not None:
            events.append(("impact", start + (shot.ramp_at - shot.start) / shot.speed))
    return events


#: The synthesized hits: a filtered noise swell (whoosh) and a sub-bass thump with a noise crack (impact).
_HITS = {
    "whoosh": ("anoisesrc=color=pink:amplitude=0.8:duration=0.7,highpass=f=800,lowpass=f=6000,"
               "afade=t=in:st=0:d=0.45,afade=t=out:st=0.45:d=0.25,volume=-8dB"),
    "impact": ("sine=frequency=48:duration=0.9,volume=6dB,afade=t=out:st=0.05:d=0.85[s];"
               "anoisesrc=color=brown:amplitude=1:duration=0.35,lowpass=f=1800,afade=t=out:st=0.02:d=0.3,volume=-4dB[n];"
               "[s][n]amix=inputs=2:normalize=0"),
}


def foley_graph(edit: Edit, spans: list[tuple[float, float]], seconds: float) -> str | None:
    """The ffmpeg filter graph of the synthesized foley track, ending on ``[fx]`` — or None when there is none (pure)."""
    parts, labels = [], []
    bed = AMBIENCES[edit.ambience]
    if bed:
        parts.append(f"{bed},atrim=0:{seconds:.3f},afade=t=in:d=1,afade=t=out:st={max(0.0, seconds - 1.5):.3f}:d=1.5[bed]")
        labels.append("[bed]")
    for i, (kind, at) in enumerate(sfx_events(edit, spans)):
        chain = _HITS[kind]
        delay = int(at * 1000)
        if ";" in chain:  # a multi-source hit: its inner labels made unique, its last stage labelled, then delayed
            chain = chain.replace("[s]", f"[s{i}]").replace("[n]", f"[n{i}]")
            parts.append(f"{chain}[h{i}raw]")
            parts.append(f"[h{i}raw]adelay={delay}|{delay}[h{i}]")
        else:
            parts.append(f"{chain},adelay={delay}|{delay}[h{i}]")
        labels.append(f"[h{i}]")
    if not labels:
        return None
    parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest,"
                 f"aformat=channel_layouts=stereo,atrim=0:{seconds:.3f},apad=whole_dur={seconds:.3f}[fx]")
    return ";".join(parts)


def upscale_command(edit: Edit, source: str, output: str, env: dict) -> list[str]:
    """The SeedVR2 command for one source clip: its short side to 1080 (9:16 → 1080x1920, 16:9 → 1920x1080) (pure)."""
    home, python = env.get("SEEDVR2_HOME"), env.get("SEEDVR2_PYTHON")
    if not home or not python:
        raise EditError("this worker has no video upscaler installed (SEEDVR2_HOME / SEEDVR2_PYTHON)")
    models = os.path.join(env.get("MEDIA_MODELS_DIR", "/models"), "seedvr2")
    return [python, os.path.join(home, "inference_cli.py"), source, "--output", output, "--resolution",
            str(min(edit.size)), "--dit_model", SEEDVR2_DIT, "--model_dir", models, "--batch_size", "5",
            "--attention_mode", "sdpa", "--color_correction", "lab", "--output_format", "mp4"]


def segment_workers(shots: int, cpus: int | None = None) -> int:
    """How many shots are made at once: one per two cores (x264 threads too), never more than the shots, at least one."""
    cores = cpus if cpus is not None else (os.cpu_count() or 1)
    return max(1, min(shots, cores // 2))


def find_font(env: dict | None = None) -> str:
    source = os.environ if env is None else env
    for candidate in (source.get("MEDIA_FONT_FILE"), *FONT_CANDIDATES):
        if candidate and os.path.isfile(candidate):
            return candidate
    raise RuntimeError("no font for the titles: install fonts-dejavu-core or set MEDIA_FONT_FILE")


def metadata_args(edit: Edit, provenance: dict | None = None) -> list[str]:
    """The AI-generated label in the file's own metadata, beside the one drawn on the picture (pure)."""
    note = {"aiGenerated": True, "label": edit.label, "generator": "Orazaka cinematic short", **(provenance or {})}
    return ["-metadata", f"comment={edit.label} — made with Orazaka (AI-generated video)",
            "-metadata", "description=" + json.dumps(note, separators=(",", ":"), ensure_ascii=False),
            "-metadata", "copyright=AI-generated content"]


def _probe_seconds(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def cut(clips: list[str], edit: Edit, output_path: str, provenance: dict | None = None,
        font: str | None = None) -> dict:
    """Makes the film of ``edit`` from ``clips`` (readable files, in the order the edit's ``clip`` indexes).

    Returns what the file is, measured on it: duration, size, shot count — what billing settles against.
    """
    output_path = os.path.abspath(output_path)
    if os.path.basename(output_path).startswith("-"):
        raise EditError("the output file name cannot start with a hyphen")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    missing = missing_filters()
    if missing:
        raise RuntimeError(f"this host's ffmpeg cannot cut a film (missing {', '.join(missing)})")
    font = font or find_font()
    staging = tempfile.mkdtemp(prefix="orz-cut-")
    try:
        if edit.grade != "none":
            with open(os.path.join(staging, "grade.cube"), "w", encoding="utf-8") as handle:
                handle.write(grade_cube(edit.grade))
        upscaled = {}
        if edit.upscale == "seedvr2":
            for index in sorted({shot.clip for shot in edit.shots}):
                target = os.path.join(staging, f"up-{index:02d}.mp4")
                _run_in(upscale_command(edit, os.path.abspath(clips[index]), target, dict(os.environ)),
                        os.environ.get("SEEDVR2_HOME", staging))
                upscaled[index] = target
        clips = [upscaled.get(i, path) for i, path in enumerate(clips)]
        segments = [os.path.join(staging, f"shot-{i:02d}.mp4") for i in range(len(edit.shots))]

        def make(i: int) -> None:
            shot = edit.shots[i]
            _run_in(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{shot.start:.3f}",
                     "-to", f"{shot.end:.3f}", "-i", os.path.abspath(clips[shot.clip]), "-an",
                     "-vf", shot_filter(edit, shot), "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                     "-r", str(edit.fps), segments[i]], staging)

        # Motion-compensated interpolation is single-threaded: the shots are made side by side, one per spare core.
        with ThreadPoolExecutor(max_workers=segment_workers(len(edit.shots))) as pool:
            list(pool.map(make, range(len(edit.shots))))
        durations = [_probe_seconds(segment) for segment in segments]
        spans = timeline(edit)
        graph = join_filter(edit, durations) + ";[joined]" + ",".join(text_filters(edit, spans, font, staging)) + "[out]"
        length = spans[-1][1]
        fx = foley_graph(edit, spans, length)
        audio = ["-map", "[fx]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"] if fx else []
        if fx:
            graph += ";" + fx
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for segment in segments:
            command += ["-i", segment]
        command += ["-filter_complex", graph, "-map", "[out]", *audio, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-r", str(edit.fps), "-t", f"{length:.3f}", *metadata_args(edit, provenance),
                    "-movflags", "+faststart", output_path]
        _run_in(command, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    width, height = edit.size
    return {"durationSeconds": round(_probe_seconds(output_path), 3), "width": width, "height": height,
            "shots": len(edit.shots), "grade": edit.grade, "fps": edit.fps, "upscale": edit.upscale,
            "ambience": edit.ambience, "sfx": len(sfx_events(edit, timeline(edit)))}


def _run_in(command: list[str], cwd: str) -> None:
    """ffmpeg, run from the staging folder: the grade's LUT is named relatively in the shot filters."""
    done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, cwd=cwd)
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg failed: {' | '.join(tail)}")
