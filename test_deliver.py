"""A film's master and its deliverables (orazaka#259): the cut can make a 4K master (SeedVR2 to 2160p, tiled) without
text, and app/deliver.py makes the 1080p and 9:16 cuts from it, reframed shot by shot, with the captions and the label
drawn per language after the reframe, so no caption is ever cropped."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest

from app import cut, deliver

EDIT = {
    "aspect": "16:9",
    "fps": 24,
    "master": "2160p",
    "text": False,
    "shots": [
        {"clip": 0, "in": 0, "out": 2, "caption": "The falcon keeps watch: security comes first."},
        {"clip": 1, "in": 0, "out": 2, "transition": "dissolve", "transitionSeconds": 0.4},
    ],
}


class TestTheMaster(unittest.TestCase):
    def test_a_2160p_master_is_3840_wide_and_upscales_to_2160_with_tiled_vae(self):
        edit = cut.parse_edit(EDIT, 2)
        self.assertEqual(edit.size, (3840, 2160))
        command = cut.upscale_command(edit, "in.mp4", "out.mp4", {"SEEDVR2_HOME": "/s", "SEEDVR2_PYTHON": "/p"})
        self.assertEqual(command[command.index("--resolution") + 1], "2160")
        self.assertIn("--vae_encode_tiled", command)
        self.assertIn("--vae_decode_tiled", command)

    def test_the_default_master_stays_1080p_without_tiling(self):
        edit = cut.parse_edit({**EDIT, "master": None, "text": None}, 2)
        self.assertEqual(edit.size, (1920, 1080))
        self.assertTrue(edit.text)
        command = cut.upscale_command(edit, "in.mp4", "out.mp4", {"SEEDVR2_HOME": "/s", "SEEDVR2_PYTHON": "/p"})
        self.assertNotIn("--vae_encode_tiled", command)

    def test_an_unknown_master_is_refused_by_name(self):
        with self.assertRaisesRegex(cut.EditError, "master must be one of 1080p, 2160p"):
            cut.parse_edit({**EDIT, "master": "8k"}, 2)

    def test_a_clean_master_draws_no_text_at_all(self):
        edit = cut.parse_edit(EDIT, 2)
        self.assertEqual(cut.overlay_filters(edit, cut.timeline(edit), "/f.ttf", "/tmp"), ["null"])
        with_text = cut.parse_edit({**EDIT, "text": True}, 2)
        self.assertTrue(any("drawtext" in f for f in cut.overlay_filters(with_text, cut.timeline(with_text), "/f.ttf", "/tmp")))


class TestTheDeliverables(unittest.TestCase):
    def test_the_known_deliverables(self):
        self.assertEqual(deliver.DELIVERABLES["1080p"], {"aspect": "16:9", "size": (1920, 1080)})
        self.assertEqual(deliver.DELIVERABLES["9:16"], {"aspect": "9:16", "size": (1080, 1920)})
        with self.assertRaisesRegex(deliver.DeliverError, "deliverable '4:3'"):
            deliver.plan("4:3", (3840, 2160), [(0, 2)], [0.5])

    def test_a_vertical_cut_follows_each_shot_centre_and_stays_inside_the_frame(self):
        spans = [(0.0, 2.0), (1.6, 3.6), (3.55, 6.0)]
        graph = deliver.plan("9:16", (3840, 2160), spans, [0.5, 0.05, 0.95])["crop"]
        # The crop is 9:16 of the master's height: 1215 x 2160, its x jumps at each shot's start.
        self.assertTrue(graph.startswith("crop=1215:2160:"))
        self.assertIn("lt(t\\,1.600)", graph)
        self.assertIn("lt(t\\,3.550)", graph)
        self.assertIn("1312", graph)  # 0.5 * 3840 - 1215 / 2 = 1312
        self.assertIn(":0", graph.replace("crop=1215:2160:", ""))  # 0.05 clamps to the left edge
        self.assertIn("2625", graph)  # 0.95 clamps to the right edge (3840 - 1215)

    def test_the_4k_cut_keeps_the_master_frame_and_lays_its_texts_out_at_4k(self):
        step = deliver.plan("2160p", (3840, 2160), [(0, 2)], [0.5])
        self.assertEqual((step["crop"], step["size"], step["master"]), (None, (3840, 2160), "2160p"))
        layout = cut.Edit(aspect="16:9", fps=24, grade="none", shots=(), master=step["master"])
        self.assertEqual(layout.size, (3840, 2160))

    def test_the_1080p_cut_is_a_scale_without_crop(self):
        step = deliver.plan("1080p", (3840, 2160), [(0, 2)], [0.5])
        self.assertIsNone(step["crop"])
        self.assertEqual(step["size"], (1920, 1080))

    def test_the_centres_must_match_the_shots(self):
        with self.assertRaisesRegex(deliver.DeliverError, "one centre per shot"):
            deliver.plan("9:16", (3840, 2160), [(0, 2), (2, 4)], [0.5])


@unittest.skipUnless(shutil.which("ffmpeg") and not cut.missing_filters(), "this ffmpeg cannot draw text (drawtext)")
class TestTheDeliverablesAreMade(unittest.TestCase):
    def test_a_master_becomes_a_captioned_vertical_cut(self):
        scratch = tempfile.mkdtemp()
        try:
            master = os.path.join(scratch, "master.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:d=4:r=24", "-pix_fmt", "yuv420p", master], check=True)
            out = os.path.join(scratch, "v.mp4")
            facts = deliver.deliver(master, out, "9:16", spans=[(0, 2), (2, 4)], centres=[0.3, 0.7],
                                    texts=[("Le pigeon livre, à chaque fois.", 0.2, 1.8, "caption")], label="Généré par IA",
                                    font=cut.find_font())
            self.assertEqual((facts["width"], facts["height"]), (1080, 1920))
            self.assertAlmostEqual(facts["durationSeconds"], 4.0, delta=0.1)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


class TestFilmTitlesReplaceDrawtext(unittest.TestCase):
    """orazaka#289: a deliverable can take the film's ASS titles (libass) instead of the drawtext layout."""

    def test_an_ass_script_is_burnt_with_its_fonts_folder(self):
        layout = cut.Edit(aspect="16:9", fps=24, grade="none", shots=(), titles=(), label="AI-generated")
        chain = deliver.text_chain(layout, "font.ttf", "/tmp", titles_ass="/w/titles: en.ass", fontsdir="/w/fonts")
        self.assertEqual(chain, ["ass=filename='/w/titles\\: en.ass':fontsdir='/w/fonts'"])
