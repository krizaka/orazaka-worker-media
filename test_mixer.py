"""The film mix (app/mixer.py, orazaka#289): cue levels follow the story, the score ducks under the voice, the master
lands on -14 LUFS ± 0.5 with a true peak at or under -1 dBTP — measured, not assumed."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest

from app import mixer


class TestTheDecisionsArePure(unittest.TestCase):
    def test_cue_levels_follow_intensity_and_never_reach_full_scale(self):
        self.assertEqual(mixer.cue_gain(0), mixer.MUSIC_FLOOR_DB)
        self.assertEqual(mixer.cue_gain(1), mixer.MUSIC_CEILING_DB)
        self.assertLess(mixer.cue_gain(0.3), mixer.cue_gain(0.75))
        self.assertLessEqual(mixer.cue_gain(5), -9.0)

    def test_the_music_graph_lays_each_cue_at_its_time_and_gain_and_resolves_at_the_end(self):
        graph, files = mixer.music_graph([mixer.Cue("b.wav", 10, 20, 0.75), mixer.Cue("a.wav", 0, 10, 0.3)], 20.0, 0)
        self.assertEqual(files, ["a.wav", "b.wav"])
        self.assertIn(f"volume={mixer.cue_gain(0.3)}dB", graph)
        self.assertIn("adelay=9250|9250", graph)  # the second cue starts half a crossfade early
        self.assertIn(
            "afade=t=out:st=7.750:d=3.00", graph
        )  # the last cue resolves over its last 3 s, ending on the film's end
        self.assertTrue(graph.endswith("[music]"))

    def test_a_hard_genre_switch_crossfades_short_on_the_cut(self):
        graph, _ = mixer.music_graph(
            [mixer.Cue("soft.wav", 0, 6, 0.3), mixer.Cue("hiphop.wav", 6, 12, 0.6, fade=0.3)], 12.0, 0
        )
        self.assertIn("adelay=5850|5850", graph)  # the hip-hop cue starts 0.15 s before the cut
        self.assertIn("afade=t=in:st=0:d=0.30", graph)
        self.assertIn("afade=t=out:st=5.850:d=0.30", graph)  # the soft cue leaves as fast

    def test_the_voice_keys_the_duck_and_gets_room(self):
        graph, files = mixer.mix_graph(
            [mixer.Cue("m.wav", 0, 8, 0.5)], [mixer.Line("v0.wav", 1.2), mixer.Line("v1.wav", 4)], "fx.wav", 8
        )
        self.assertEqual(files, ["m.wav", "v0.wav", "v1.wav", "fx.wav"])
        self.assertIn("[music][vkey1]sidechaincompress", graph)
        self.assertIn("[fx][vkey2]sidechaincompress", graph)
        self.assertIn("adelay=1200|1200", graph)
        self.assertIn("aecho=", graph)
        self.assertIn("anoisesrc", graph)

    def test_the_contract(self):
        self.assertTrue(mixer.within_contract({"lufs": -14.3, "truePeak": -1.2}))
        self.assertFalse(mixer.within_contract({"lufs": -13.2, "truePeak": -1.2}))
        self.assertFalse(mixer.within_contract({"lufs": -14.0, "truePeak": -0.6}))
        self.assertEqual(mixer.master_gain(-20.5), 6.5)

    def test_a_score_needs_a_cue(self):
        with self.assertRaises(mixer.MixError):
            mixer.music_graph([], 5, 0)


def _ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg"))


@unittest.skipUnless(_ffmpeg(), "ffmpeg is not installed")
class TestTheMasterIsMeasured(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _make(self, name: str, source: str, seconds: float) -> str:
        path = os.path.join(self.dir, name)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", source, "-t", str(seconds), "-ar", "48000", path],
            check=True,
        )
        return path

    def test_a_loud_score_and_a_voice_master_to_minus_14_lufs_under_minus_1_dbtp(self):
        loud = self._make("score.wav", "sine=f=220:sample_rate=48000,volume=0.95", 8)
        voice = self._make("voice.wav", "anoisesrc=c=pink:a=0.5,bandpass=f=1000:w=800", 2.5)
        fx = self._make("fx.wav", "anoisesrc=c=brown:a=0.3", 8)
        out = os.path.join(self.dir, "master.wav")
        facts = mixer.master(
            [mixer.Cue(loud, 0, 4, 0.3), mixer.Cue(loud, 4, 8, 0.7)], [mixer.Line(voice, 2.0)], fx, 8, out
        )
        self.assertTrue(mixer.within_contract(facts["master"]), facts)
        self.assertEqual(mixer.measure(out), facts["master"])

    def test_the_score_is_lower_while_the_voice_speaks(self):
        tone = self._make("tone.wav", "sine=f=440:sample_rate=48000,volume=0.5", 6)
        voice = self._make("voice.wav", "anoisesrc=c=pink:a=0.6", 2.0)
        graph, files = mixer.mix_graph([mixer.Cue(tone, 0, 6, 0.6)], [mixer.Line(voice, 3.0)], None, 6)
        ducked = os.path.join(self.dir, "ducked.wav")
        cmd = ["ffmpeg", "-v", "error", "-y"]
        for f in files:
            cmd += ["-i", f]
        # The ducked music alone: the same graph, its [ducked] output instead of the mix.
        graph = graph.replace("[ducked][vmain]amix=inputs=2:normalize=0[mix]", "[vmain]anullsink")
        subprocess.run(cmd + ["-filter_complex", graph, "-map", "[ducked]", ducked], check=True)

        def rms(start: float) -> float:
            err = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-ss",
                    str(start),
                    "-t",
                    "1",
                    "-i",
                    ducked,
                    "-af",
                    "astats=metadata=0",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
            ).stderr
            return float(err.split("RMS level dB:")[-1].split()[0])

        self.assertGreater(rms(1.0) - rms(3.8), 3.0)  # at least 3 dB down under the voice


if __name__ == "__main__":
    unittest.main()
