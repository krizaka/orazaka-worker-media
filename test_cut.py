"""The cut of a cinematic short (app/cut.py): the edit is checked, the graph and the LUT are pure, ffmpeg makes it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest

from app import cut

TWO = {
    "aspect": "9:16",
    "grade": "teal-orange",
    "shots": [
        {"clip": 0, "in": 0, "out": 2, "caption": "Night falls on the city"},
        {
            "clip": 1,
            "in": 0.5,
            "out": 2.5,
            "transition": "flash",
            "transitionSeconds": 0.3,
            "rampAt": 1.5,
            "rampSpeed": 0.5,
            "zoom": 1.1,
        },
    ],
    "titles": [{"text": "NIGHTFALL", "start": 0.2, "end": 1.5, "style": "hero"}],
}


class TestTheEditIsCheckedOnce(unittest.TestCase):
    def test_a_valid_edit_parses_with_its_defaults(self):
        edit = cut.parse_edit(json.dumps(TWO), 2)
        self.assertEqual((1080, 1920), edit.size)
        self.assertEqual(30, edit.fps)
        self.assertEqual("AI-generated", edit.label)
        self.assertEqual("flash", edit.shots[1].transition)

    def test_what_cannot_be_made_is_refused_by_name(self):
        cases = {
            "aspect must be": {**TWO, "aspect": "4:3"},
            "grade must be": {**TWO, "grade": "sepia-dream"},
            "no shots": {**TWO, "shots": []},
            "shot 1.clip": {**TWO, "shots": [TWO["shots"][0], {**TWO["shots"][1], "clip": 5}]},
            "shot 0.out": {**TWO, "shots": [{"clip": 0, "in": 2, "out": 2.1}]},
            "transition must be": {**TWO, "shots": [TWO["shots"][0], {**TWO["shots"][1], "transition": "star"}]},
            "longer than half": {**TWO, "shots": [TWO["shots"][0], {**TWO["shots"][1], "transitionSeconds": 1.4}]},
            "speed": {**TWO, "shots": [{"clip": 0, "speed": 9}]},
            "longer than 160": {**TWO, "shots": [{"clip": 0, "caption": "x" * 161}]},
        }
        for message, raw in cases.items():
            with self.subTest(message), self.assertRaises(cut.EditError) as refused:
                cut.parse_edit(raw, 2)
            self.assertIn(message, str(refused.exception))

    def test_not_json_is_refused(self):
        with self.assertRaises(cut.EditError):
            cut.parse_edit("{nope", 1)

    def test_a_ramp_lasts_longer_than_the_clip_it_slows(self):
        shot = cut.parse_edit(TWO, 2).shots[1]
        self.assertAlmostEqual(1.0 + 1.0 / 0.5, shot.seconds)  # 1 s at speed 1, then 1 s at half speed


class TestTheTimeline(unittest.TestCase):
    def test_a_transition_overlaps_the_shots_it_joins(self):
        edit = cut.parse_edit(TWO, 2)
        (a0, a1), (b0, b1) = cut.timeline(edit)
        self.assertEqual((0.0, 2.0), (a0, a1))
        self.assertAlmostEqual(2.0 - 0.3, b0)
        self.assertAlmostEqual(b0 + 3.0, b1)

    def test_the_join_is_one_xfade_chain_with_the_offsets_of_the_timeline(self):
        edit = cut.parse_edit(TWO, 2)
        graph = cut.join_filter(edit, [2.0, 3.0])
        self.assertIn("xfade=transition=fadewhite:duration=0.300:offset=1.700[joined]", graph)

    def test_a_hard_cut_is_a_one_frame_fade(self):
        edit = cut.parse_edit({**TWO, "shots": [TWO["shots"][0], {"clip": 1, "in": 0, "out": 2}]}, 2)
        self.assertIn(f"duration={1 / 30:.3f}", cut.join_filter(edit, [2.0, 2.0]))

    def test_one_shot_needs_no_join(self):
        edit = cut.parse_edit({"shots": [{"clip": 0, "out": 2}]}, 1)
        self.assertEqual("[0:v]settb=AVTB,fps=30[joined]", cut.join_filter(edit, [2.0]))


class TestTheShotFilter(unittest.TestCase):
    def test_speed_interpolation_push_in_and_grade_in_that_order(self):
        edit = cut.parse_edit(TWO, 2)
        chain = cut.shot_filter(edit, edit.shots[1])
        order = [chain.index(k) for k in ("setpts=", "minterpolate=fps=30", "eval=frame", "crop=1080:1920", "lut3d=")]
        self.assertEqual(sorted(order), order)

    def test_no_grade_no_lut(self):
        edit = cut.parse_edit({**TWO, "grade": "none"}, 2)
        self.assertNotIn("lut3d", cut.shot_filter(edit, edit.shots[0]))


class TestTheGrade(unittest.TestCase):
    def test_the_cube_has_every_point_in_range(self):
        cube = cut.grade_cube("neon-night", points=5).splitlines()
        self.assertEqual("LUT_3D_SIZE 5", cube[1])
        values = [float(v) for line in cube[2:] for v in line.split()]
        self.assertEqual(5**3 * 3, len(values))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))

    def test_none_is_the_identity_and_teal_orange_cools_the_shadows(self):
        self.assertEqual((0.2, 0.4, 0.6), tuple(round(c, 6) for c in cut.grade_rgb("none", 0.2, 0.4, 0.6)))
        r, _g, b = cut.grade_rgb("teal-orange", 0.15, 0.15, 0.15)
        self.assertGreater(b, r)
        r, _g, b = cut.grade_rgb("teal-orange", 0.85, 0.85, 0.85)
        self.assertGreater(r, b)


class TestTheLabel(unittest.TestCase):
    def test_the_label_is_in_the_metadata_and_not_burnt_in_by_default(self):
        # orazaka#289: the players draw the AI disclosure themselves (a quiet chip from the file's
        # provenance); a label burnt into the picture doubled it. It is drawn only when an edit asks (`burnLabel`).
        edit = cut.parse_edit(TWO, 2)
        self.assertFalse(edit.burn_label)
        with tempfile.TemporaryDirectory() as scratch:
            filters = cut.text_filters(edit, cut.timeline(edit), "/font.ttf", scratch)
            self.assertFalse(os.path.exists(os.path.join(scratch, "label.txt")))
        self.assertFalse(any("label.txt" in f for f in filters))
        args = cut.metadata_args(edit)
        self.assertIn("AI-generated", args[args.index("-metadata") + 1])
        self.assertTrue(json.loads(args[3].split("=", 1)[1])["aiGenerated"])

    def test_the_label_is_drawn_when_the_edit_asks(self):
        edit = cut.parse_edit({**TWO, "burnLabel": True}, 2)
        with tempfile.TemporaryDirectory() as scratch:
            filters = cut.text_filters(edit, cut.timeline(edit), "/font.ttf", scratch)
            with open(os.path.join(scratch, "label.txt"), encoding="utf-8") as handle:
                self.assertEqual("AI-generated", handle.read())
        self.assertIn("label.txt", filters[-1])
        args = cut.metadata_args(edit)
        self.assertIn("AI-generated", args[args.index("-metadata") + 1])
        self.assertTrue(json.loads(args[3].split("=", 1)[1])["aiGenerated"])

    def test_a_caption_is_read_from_a_file_never_spliced_into_the_filter(self):
        edit = cut.parse_edit({"shots": [{"clip": 0, "out": 2, "caption": "It's a bird: ok"}]}, 1)
        with tempfile.TemporaryDirectory() as scratch:
            filters = cut.text_filters(edit, cut.timeline(edit), "/font.ttf", scratch)
        self.assertFalse(any("It's" in f for f in filters))


def _ffmpeg_can_draw_text() -> bool:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return False
    listed = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, check=False).stdout
    return all(f" {name} " in listed for name in ("drawtext", "xfade", "lut3d", "minterpolate"))


class TestParallelSegments(unittest.TestCase):
    def test_shots_are_made_side_by_side_within_the_cores(self):
        self.assertEqual(8, cut.segment_workers(10, 16))
        self.assertEqual(3, cut.segment_workers(3, 64))
        self.assertEqual(1, cut.segment_workers(10, 1))


class TestUpscaleAndFoley(unittest.TestCase):
    """Hollywood finishing (ADR-080): SeedVR2 per source clip, aspect kept; a synthesized foley bed and hits, no model."""

    def test_seedvr2_takes_the_short_side_to_1080_whatever_the_aspect(self):
        env = {"SEEDVR2_HOME": "/up/seedvr2", "SEEDVR2_PYTHON": "/up/venv/bin/python", "MEDIA_MODELS_DIR": "/models"}
        vertical = cut.parse_edit({**TWO, "upscale": "seedvr2"}, 2)
        command = cut.upscale_command(vertical, "/in.mp4", "/out.mp4", env)
        self.assertEqual("1080", command[command.index("--resolution") + 1])
        self.assertEqual("/models/seedvr2", command[command.index("--model_dir") + 1])
        self.assertEqual("/up/seedvr2/inference_cli.py", command[1])
        with self.assertRaises(cut.EditError):
            cut.upscale_command(vertical, "/in.mp4", "/out.mp4", {})

    def test_unknown_upscaler_or_ambience_is_refused(self):
        for key, value in (("upscale", "esrgan-x9"), ("ambience", "jungle-party")):
            with self.subTest(key), self.assertRaises(cut.EditError):
                cut.parse_edit({**TWO, key: value}, 2)

    def test_hits_fall_on_the_transitions_flashes_and_ramps(self):
        edit = cut.parse_edit({**TWO, "sfx": True}, 2)
        events = cut.sfx_events(edit, cut.timeline(edit))
        kinds = [k for k, _ in events]
        self.assertEqual(["whoosh", "impact", "impact"], kinds)  # flash: whoosh + impact; the ramp: impact
        _, flash_at = events[1]
        self.assertAlmostEqual(cut.timeline(edit)[1][0], flash_at)
        self.assertEqual([], cut.sfx_events(cut.parse_edit(TWO, 2), cut.timeline(cut.parse_edit(TWO, 2))))

    def test_the_foley_graph_has_a_bed_unique_labels_and_ends_on_fx(self):
        edit = cut.parse_edit({**TWO, "sfx": True, "ambience": "storm"}, 2)
        graph = cut.foley_graph(edit, cut.timeline(edit), 5.0)
        self.assertTrue(graph.endswith("[fx]"))
        self.assertIn("[bed]", graph)
        self.assertIn("[s1]", graph)
        self.assertIn("[s2]", graph)
        self.assertNotIn("[s]", graph)
        self.assertIsNone(cut.foley_graph(cut.parse_edit(TWO, 2), cut.timeline(cut.parse_edit(TWO, 2)), 5.0))


@unittest.skipUnless(_ffmpeg_can_draw_text(), "ffmpeg without drawtext/xfade/lut3d (e.g. Homebrew's default build)")
class TestFfmpegMakesTheFilm(unittest.TestCase):
    def test_two_clips_become_one_graded_labelled_1080p_film(self):
        with tempfile.TemporaryDirectory() as work:
            clips = []
            for i, colour in enumerate(("red", "blue")):
                path = os.path.join(work, f"c{i}.mp4")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        f"testsrc2=size=360x640:rate=16:duration=3,hue=s=1,drawbox=c={colour}:t=40",
                        "-pix_fmt",
                        "yuv420p",
                        path,
                    ],
                    check=True,
                )
                clips.append(path)
            out = os.path.join(work, "film.mp4")
            edit = cut.parse_edit({**TWO, "interpolate": False, "sfx": True, "ambience": "rain"}, 2)
            measured = cut.cut(clips, edit, out, font=cut.find_font())
            self.assertEqual((1080, 1920), (measured["width"], measured["height"]))
            self.assertAlmostEqual(cut.timeline(edit)[-1][1], measured["durationSeconds"], delta=0.25)
            probe = json.loads(
                subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", out],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )
            self.assertIn("AI-generated", probe["format"]["tags"]["comment"])
            streams = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", out],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
            self.assertEqual(["video", "audio"], streams)  # the synthesized foley is the film's own sound


if __name__ == "__main__":
    unittest.main()


class TestTheCutFetchesUploadedClips(unittest.TestCase):
    def test_an_asset_id_clip_is_fetched_from_the_job_service_on_a_remote_host(self):
        import sys
        import types
        from unittest.mock import patch

        sys.modules.setdefault("pika", types.ModuleType("pika"))
        from app import consumer

        asset = "0b6f2a52-1c1e-4c84-9a8e-2f6b2b1d9a10"
        job = {
            "jobId": "j1",
            "userId": "u1",
            "payload": {"clips": [asset], "edit": json.dumps({"shots": [{"clip": 0, "out": 2}]})},
        }
        fetched = []

        def fake_fetch(job_arg, target, asset=None):
            fetched.append((job_arg["jobId"], asset))
            return target

        with (
            tempfile.TemporaryDirectory() as root,
            patch.object(consumer.asset_sink, "sink_url", return_value="http://job-service"),
            patch.object(consumer.asset_sink, "fetch_image", side_effect=fake_fetch),
            patch.object(consumer, "_upload_root", return_value=root),
            patch.object(consumer, "_materialise", side_effect=lambda path, scratch: path),
            patch.object(consumer.film, "cut", return_value={"durationSeconds": 2.0, "width": 1080, "height": 1920}),
            patch.object(consumer, "_store", return_value="/api/v1/assets/j1/video.mp4"),
            patch.object(consumer, "send_progress_update"),
        ):
            result, _ = consumer._cut(job)
        self.assertEqual([("j1", asset)], fetched)
        self.assertEqual("/api/v1/assets/j1/video.mp4", result["url"])
