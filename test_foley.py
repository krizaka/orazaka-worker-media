"""Foley by synthesis (app/foley.py, orazaka#289): every sound of the list is made, finite, bounded and deterministic,
laid where the timeline says, and the coverage says which shots are silent or whose action is not heard."""
from __future__ import annotations

import unittest

import numpy as np

from app import foley


class TestEverySoundIsMade(unittest.TestCase):
    def test_each_kind_renders_audible_bounded_and_deterministic(self):
        for kind in foley.BEDS:
            a, b = foley.bed(kind, 1.0, 7), foley.bed(kind, 1.0, 7)
            self.assertTrue(np.isfinite(a).all(), kind)
            self.assertGreater(np.abs(a).max(), 0.05, kind)
            np.testing.assert_array_equal(a, b)
        for kind in foley.EVENTS:
            e = foley.event(kind, 3)
            self.assertTrue(np.isfinite(e).all(), kind)
            self.assertGreater(np.abs(e).max(), 0.01, kind)
            self.assertLess(len(e), foley.RATE * 4, kind)

    def test_the_city_at_night_is_heard(self):
        """v3 (ADR-080 v3): the underground films need their sounds — a tunnel, a crowd, a neon buzz; a train, a
        skateboard, sparks, an impact, a wet slide, a riser."""
        for kind in ("tunnel", "crowd", "neon"):
            self.assertIn(kind, foley.BEDS)
        for kind in ("train", "skate", "sparks", "impact", "slide", "riser"):
            self.assertIn(kind, foley.EVENTS)
        # An impact is low (its energy under 200 Hz), sparks are bright (over 2 kHz).
        def low_share(x):
            spec = np.abs(np.fft.rfft(x))
            freqs = np.fft.rfftfreq(len(x), 1 / foley.RATE)
            return spec[freqs < 200].sum() / spec.sum()
        self.assertGreater(low_share(foley.event("impact", 1)), 0.4)
        self.assertLess(low_share(foley.event("sparks", 1)), 0.05)

    def test_an_unknown_sound_is_refused_by_name(self):
        with self.assertRaisesRegex(foley.FoleyError, "explosion"):
            foley.render([{"kind": "explosion", "start": 0, "end": 1}], 1)


class TestTheStem(unittest.TestCase):
    def test_a_bed_shorter_than_its_stereo_delay_still_renders(self):
        # A sliver of a span (a frame where two shots meet) once made the two sides different lengths.
        for seconds in (0.001, 0.005, 0.011, 0.02):
            out = foley.render([{"kind": "wind", "start": 1.0, "end": 1.0 + seconds}], 2.0)
            self.assertEqual(out.shape, (2 * foley.RATE, 2))
            self.assertTrue(np.isfinite(out).all())

    def test_events_land_on_their_second_and_beds_span_their_shot(self):
        stem = foley.render([{"kind": "rain", "start": 1.0, "end": 2.0, "gain": -12},
                             {"kind": "coin", "start": 2.5, "gain": -6}], 4.0, seed=1)
        self.assertEqual(stem.shape, (4 * foley.RATE, 2))
        energy = lambda a, b: float(np.abs(stem[int(a * foley.RATE):int(b * foley.RATE)]).mean())  # noqa: E731
        self.assertLess(energy(0.0, 0.9), 1e-6)  # nothing before the rain
        self.assertGreater(energy(1.2, 1.8), 1e-3)  # the rain
        self.assertLess(energy(2.1, 2.45), 1e-6)  # silence between
        self.assertGreater(energy(2.5, 2.7), energy(1.2, 1.8))  # the coin, louder
        self.assertLessEqual(np.abs(stem).max(), 0.89 + 1e-6)

    def test_coverage_names_silent_shots_and_unheard_characters(self):
        shots = [{"id": 1, "characters": []}, {"id": 2, "characters": ["pigeon"]}, {"id": 3, "characters": ["owl"]}]
        cues = [{"shot": 1, "kind": "rain"}, {"shot": 2, "kind": "wind"}, {"shot": 2, "kind": "rain"}]
        self.assertEqual(foley.coverage(shots, cues), {"shots": 3, "covered": 2, "ratio": 0.667, "silent": [3],
                                                       "characterShotsUnheard": [2, 3]})


if __name__ == "__main__":
    unittest.main()
