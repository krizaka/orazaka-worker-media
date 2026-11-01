"""A narrator directed like an actor (orazaka#289): a line's direction — emotion, pace, emphasis, the pause after — turned
into what an expressive speech model takes (pure).

The v1 narrator was Kokoro-82M at one speed: every line the same flat contour, the « GPS effect ». The A/B of
orazaka#289 (Kokoro, Kokoro with phrasing, Chatterbox, Zonos; ADR-080 addendum v2) chose **Zonos v0.1 transformer**
(Zyphra, Apache-2.0): it takes an emotion vector, a speaking rate and a pitch variation per utterance, in English and in
French, and its takes moved 12–19 semitones where Kokoro's moved 9–11, every word understood (Whisper, WER 0).

A line is said in **phrases** (split at its ellipses, commas and colons — where a storyteller breathes). Each phrase gets
the line's emotion and rate; a phrase that holds an emphasised word is said a little slower with more pitch movement;
each break gets the pause its punctuation asks; the line ends on its ``pauseAfter``. :func:`plan` is the whole decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Zonos's emotion axes, in its order.
AXES = ("happiness", "sadness", "disgust", "fear", "surprise", "anger", "other", "neutral")

#: Words of a direction and the axes they lean on (several words add up; the vector is then normalised).
LEXICON = {
    "suspense": {"fear": 0.25, "surprise": 0.1, "other": 0.15},
    "hushed": {"neutral": 0.2, "sadness": 0.1, "fear": 0.1},
    "tense": {"fear": 0.3, "anger": 0.05},
    "fear": {"fear": 0.5},
    "calm": {"neutral": 0.4, "happiness": 0.05},
    "resolve": {"neutral": 0.3, "anger": 0.05, "happiness": 0.1},
    "warm": {"happiness": 0.35, "neutral": 0.2},
    "smile": {"happiness": 0.4},
    "pride": {"happiness": 0.35, "neutral": 0.2},
    "wonder": {"surprise": 0.3, "happiness": 0.2},
    "awe": {"surprise": 0.35, "happiness": 0.15},
    "tender": {"happiness": 0.25, "sadness": 0.2},
    "hope": {"happiness": 0.3, "sadness": 0.1},
    "joy": {"happiness": 0.55},
    "joyful": {"happiness": 0.55},
    "light": {"happiness": 0.3},
    "sad": {"sadness": 0.5},
    "satisfied": {"happiness": 0.35, "neutral": 0.2},
    "full": {"happiness": 0.3, "neutral": 0.2},
}

#: Speaking rate per pace (Zonos's ``speaking_rate``, roughly phonemes a second): a film narrator sits under talk.
RATES = {"slow": 11.0, "measured": 13.0, "rhythmic": 14.0, "lively": 15.0}
PITCH_STD = 60.0
EMPHASIS_RATE = 0.9
EMPHASIS_PITCH = 1.3
#: A phrase this short (in words) is joined to the next.
MIN_PHRASE_WORDS = 2
#: The breath each break asks for, in seconds.
BREAKS = {"…": 0.55, "...": 0.55, ",": 0.18, ":": 0.3, ";": 0.3, "—": 0.3}


@dataclass(frozen=True)
class Phrase:
    text: str
    emotion: tuple[float, ...]
    rate: float
    pitch_std: float
    pause_after: float


def emotion_vector(emotion: str) -> tuple[float, ...]:
    """The emotion axes a direction asks for (pure): its words looked up, summed with a neutral floor, normalised."""
    vector = dict.fromkeys(AXES, 0.0)
    vector["neutral"] = 0.15
    for word in re.findall(r"[a-z]+", emotion.lower()):
        for axis, weight in LEXICON.get(word, {}).items():
            vector[axis] += weight
    total = sum(vector.values())
    return tuple(round(vector[a] / total, 3) for a in AXES)


def phrases(text: str) -> list[tuple[str, float]]:
    """The line cut where a storyteller breathes: (phrase, pause after it), the last pause left to the line (pure)."""
    out: list[tuple[str, float]] = []
    for part in re.split(r"(\.\.\.|…|[,:;—])", text):
        if part in BREAKS:
            if out:
                out[-1] = (out[-1][0] + ("," if part == "," else ""), BREAKS[part])
        elif part.strip():
            out.append((part.strip(), 0.0))
    # A phrase of one or two words is said with the next one, its comma kept: Zonos drops such a short utterance
    # (« In Québec, … » lost its first words in every take, orazaka#289).
    merged: list[tuple[str, float]] = []
    for phrase, pause in out:
        if merged and len(merged[-1][0].split()) <= MIN_PHRASE_WORDS and merged[-1][1] < BREAKS["…"]:
            merged[-1] = (f"{merged[-1][0]} {phrase}", pause)
        else:
            merged.append((phrase, pause))
    return merged


def plan(text: str, direction: dict) -> list[Phrase]:
    """A line and its direction as the phrases a speech model says (pure)."""
    text = text.replace(
        "’", "'"
    )  # the phonemizer reads a typographic apostrophe as a break (« on t’aide » → « on tête »)
    emotion = emotion_vector(direction.get("emotion", ""))
    rate = RATES.get(direction.get("pace", "measured"), RATES["measured"])
    emphasis = [e.lower() for e in direction.get("emphasis", [])]
    parts = phrases(text)
    out = []
    for k, (phrase, pause) in enumerate(parts):
        stressed = any(e in phrase.lower() for e in emphasis)
        last = k == len(parts) - 1
        out.append(
            Phrase(
                phrase,
                emotion,
                round(rate * (EMPHASIS_RATE if stressed else 1.0), 2),
                round(PITCH_STD * (EMPHASIS_PITCH if stressed else 1.0), 1),
                float(direction.get("pauseAfter", 0.3)) if last else pause,
            )
        )
    return out
