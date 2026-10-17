"""The Krizaka cinematic kit, as worker defaults: what a starting image costs the frame, and the negatives every Wan clip carries (no GPU)."""
from __future__ import annotations

import unittest

from PIL import Image

from app import video_models, wan_generator


class TestKept(unittest.TestCase):
    def test_a_still_at_the_frames_aspect_keeps_everything(self):
        self.assertEqual(1.0, video_models.kept_fraction(Image.new("RGB", (1280, 720)), 1280, 720))

    def test_a_portrait_still_cut_to_16_9_keeps_a_third_of_its_pixels_at_most(self):
        kept = video_models.kept_fraction(Image.new("RGB", (1024, 1024)), 1280, 720)
        self.assertAlmostEqual(0.5625, kept, places=2)  # 1024x576 of 1024x1024
        self.assertLess(video_models.kept_fraction(Image.new("RGB", (768, 1360)), 1280, 720), 0.35)

    def test_a_wide_still_cut_to_9_16_keeps_little(self):
        self.assertLess(video_models.kept_fraction(Image.new("RGB", (1360, 768)), 720, 1280), 0.35)

    def test_a_still_a_little_off_the_aspect_keeps_almost_everything(self):
        self.assertGreater(video_models.kept_fraction(Image.new("RGB", (1360, 768)), 1280, 720), 0.99)


class TestNegatives(unittest.TestCase):
    def test_every_wan_clip_carries_the_artefacts_of_image_to_video_in_its_negatives(self):
        for word in ("morphing", "warping", "flicker", "jitter", "distorted", "extra limbs", "watermark", "text"):
            self.assertIn(word, wan_generator.DEFAULT_NEGATIVE_PROMPT)

    def test_the_wan_examples_negatives_are_kept(self):
        for word in ("blurred details", "worst quality", "extra fingers", "walking backwards"):
            self.assertIn(word, wan_generator.DEFAULT_NEGATIVE_PROMPT)


if __name__ == "__main__":
    unittest.main()
