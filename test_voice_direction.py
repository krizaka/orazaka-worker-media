"""A narrator directed like an actor (app/voice_direction.py, orazaka#289): emotion words become Zonos's emotion axes,
pace a speaking rate, emphasis a slower and more melodic phrase, punctuation a breath."""
from __future__ import annotations

import unittest

from app import voice_direction as vd


class TestDirection(unittest.TestCase):
    def test_emotion_words_lean_on_their_axes_and_sum_to_one(self):
        warm = dict(zip(vd.AXES, vd.emotion_vector("warm, a smile"), strict=True))
        tense = dict(zip(vd.AXES, vd.emotion_vector("hushed suspense"), strict=True))
        self.assertAlmostEqual(sum(warm.values()), 1.0, places=2)
        self.assertEqual(max(warm, key=warm.get), "happiness")
        self.assertGreater(tense["fear"], warm["fear"])
        self.assertEqual(max(dict(zip(vd.AXES, vd.emotion_vector(""), strict=True)).items(), key=lambda x: x[1])[0], "neutral")

    def test_a_line_is_said_in_phrases_with_the_breaths_its_punctuation_asks(self):
        self.assertEqual(vd.phrases("The night the storm came… one parcel still had to cross the valley."),
                         [("The night the storm came", 0.55), ("one parcel still had to cross the valley.", 0.0)])
        self.assertEqual(vd.phrases("The falcon keeps watch: security comes first."),
                         [("The falcon keeps watch", 0.3), ("security comes first.", 0.0)])

    def test_pace_and_emphasis(self):
        plan = vd.plan("And when the night is longest… no one flies alone.",
                       {"emotion": "tender, rising hope", "pace": "slow", "emphasis": ["alone"], "pauseAfter": 0.6})
        self.assertEqual([p.rate for p in plan], [11.0, 9.9])
        self.assertEqual([p.pitch_std for p in plan], [60.0, 78.0])
        self.assertEqual([p.pause_after for p in plan], [0.55, 0.6])
        self.assertEqual(vd.plan("Hi.", {"pace": "lively"})[0].rate, 15.0)


if __name__ == "__main__":
    unittest.main()
