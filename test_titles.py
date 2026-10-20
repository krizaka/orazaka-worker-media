"""Film titles and subtitles (app/titles.py, orazaka#289): set inside the safe areas of 16:9 and 9:16, two balanced
lines at most, timed on the cut, the title pairing faded and tracked; burnt in by libass when ffmpeg has it."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest

from app import titles


class TestTheScriptIsPure(unittest.TestCase):
    def test_safe_areas_of_a_landscape_and_a_vertical_frame(self):
        self.assertEqual(titles.Frame(1920, 1080).safe(), {"left": 192, "right": 192, "top": 108, "bottom": 108})
        # A vertical frame keeps clear of the platforms' buttons (bottom quarter) and header (top tenth).
        self.assertEqual(titles.Frame(1080, 1920).safe(), {"left": 108, "right": 108, "top": 192, "bottom": 461})

    def test_two_balanced_lines_at_most_breaking_after_punctuation(self):
        self.assertEqual(titles.wrap("The pigeon delivers, every time."), ["The pigeon delivers, every time."])
        self.assertEqual(titles.wrap("And when the night is longest… no one flies alone."),
                         ["And when the night is longest…", "no one flies alone."])
        with self.assertRaises(titles.TitleError):
            titles.wrap("word " * 30)

    def test_subtitles_land_on_a_near_cut_and_stay_a_second(self):
        out = titles.on_cuts([titles.Text(3.1, 3.4, "Hi."), titles.Text(0.2, 5.0, "Title", "main")], cuts=[3.0, 6.0])
        self.assertEqual((out[0].start, out[0].end), (3.0, 4.0))
        self.assertEqual((out[1].start, out[1].end), (0.2, 5.0))  # a title keeps its own timing
        out = titles.on_cuts([titles.Text(1.0, 5.85, "Line.")], cuts=[6.0])
        self.assertEqual(out[0].end, 5.96)

    def test_the_title_fits_the_safe_width_at_its_widest_tracking(self):
        for frame in (titles.Frame(1920, 1080), titles.Frame(1080, 1920)):
            text = "THE WORLD OF KRIZAKA"
            size = titles.title_size(frame, text, titles.sizes(frame)["title"])
            widest = size * (len(text) * titles.CAPS_ADVANCE + (len(text) - 1) * titles.TRACK_END)
            safe = frame.safe()
            self.assertLessEqual(widest, frame.width - safe["left"] - safe["right"])

    def test_the_script_sets_the_pairing_fades_and_tracking(self):
        ass = titles.script(titles.Frame(1920, 1080), [titles.Text(1, 6, "The World of Krizaka", "main", sub="Dawn"),
                                                        titles.Text(2, 4, "The falcon keeps watch.")], label="AI-generated")
        self.assertIn(f"Style: Title,{titles.DISPLAY_FACE}", ass)
        self.assertIn(f"Style: Serif,{titles.SERIF_FACE}", ass)
        self.assertIn(f"Style: Sub,{titles.TEXT_FACE}", ass)
        self.assertIn("THE WORLD OF KRIZAKA", ass)
        self.assertRegex(ass, r"\\t\(0,5000,\\fsp\d+\)")  # the slow tracking over the title's life
        self.assertIn(f"\\fad({titles.TITLE_FADE_IN_MS},{titles.TITLE_FADE_OUT_MS})", ass)
        self.assertIn("Dialogue: 1,0:00:01.45,0:00:06.00,Serif", ass)  # the serif breathes in after the name
        self.assertIn("\\blur3}The falcon keeps watch.", ass)  # a soft halo, never a box (BorderStyle 1)
        self.assertIn("AI-generated", ass)

    def test_a_text_that_ends_before_it_starts_is_refused(self):
        with self.assertRaises(titles.TitleError):
            titles.script(titles.Frame(1920, 1080), [titles.Text(3, 2, "x")])


def _ffmpeg_has_libass() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    listed = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, check=False).stdout
    return " ass " in listed


FONTS = os.environ.get("TITLES_FONTS_DIR", "")


@unittest.skipUnless(_ffmpeg_has_libass() and os.path.isdir(FONTS), "ffmpeg without libass, or no TITLES_FONTS_DIR")
class TestTheTextIsBurntIn(unittest.TestCase):
    def test_16x9_and_9x16_render_with_text_on_the_picture(self):
        tmp = tempfile.mkdtemp()
        try:
            for w, h in ((1920, 1080), (1080, 1920)):
                src = os.path.join(tmp, f"src-{w}.mp4")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x203040:s={w}x{h}:d=3:r=24", src], check=True)
                out = os.path.join(tmp, f"out-{w}.mp4")
                ass = titles.script(titles.Frame(w, h), [titles.Text(0, 3, "The World of Krizaka", "main", sub="Dawn")])
                facts = titles.render(src, ass, out, FONTS)
                self.assertEqual((facts["width"], facts["height"]), (w, h))
                frames = [subprocess.run(["ffmpeg", "-v", "error", "-ss", "2", "-i", p, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt",
                                          "gray", "-"], capture_output=True, check=True).stdout for p in (src, out)]
                self.assertGreater(sum(1 for a, b in zip(*frames, strict=False) if abs(a - b) > 40), 500)  # letters on the picture
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
