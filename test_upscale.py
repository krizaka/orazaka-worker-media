"""The optional video upscale: what is asked, what is refused, the commands it runs (no GPU, no model)."""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from app import upscale, video_models

ENV = {"SEEDVR2_HOME": "/opt/seedvr2", "SEEDVR2_PYTHON": "/opt/up/bin/python", "MEDIA_MODELS_DIR": "/models"}


class Runner:
    """Records the commands; ffprobe answers a 1280x704 clip, ffmpeg and the upscaler write their output file."""

    def __init__(self, root=None):
        self.calls = []
        self.root = os.path.realpath(root) if root else None

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if cmd[0] == "ffprobe":
            return SimpleNamespace(stdout="1280,704\n")
        out = cmd[cmd.index("--output") + 1] if "--output" in cmd else cmd[-1]
        resolved = os.path.realpath(out)
        if self.root is None or resolved.startswith(self.root + os.sep):  # only ever inside the scratch folders of the test
            with open(resolved, "wb") as handle:
                handle.write(b"upscaled")
        return SimpleNamespace(stdout="")


class TestRequested(unittest.TestCase):
    def test_nothing_asked_means_no_upscale(self):
        for value in (None, "", "none", False):
            self.assertIsNone(upscale.requested({"upscale": value}))
        self.assertIsNone(upscale.requested({}))

    def test_1080p_and_4k_are_the_two_targets(self):
        self.assertEqual("1080p", upscale.requested({"upscale": "1080p"}))
        self.assertEqual("2160p", upscale.requested({"upscale": "4K"}))
        self.assertEqual("2160p", upscale.requested({"upscale": "2160p"}))

    def test_anything_else_is_refused_by_name_as_the_callers_mistake(self):
        with self.assertRaises(video_models.InvalidVideoRequest) as raised:
            upscale.requested({"upscale": "8k"})
        self.assertIn("8k", str(raised.exception))


class TestUpscale(unittest.TestCase):
    def _clip(self):
        folder = tempfile.mkdtemp()
        self.folder = folder
        path = os.path.join(folder, "clip.mp4")
        with open(path, "wb") as handle:
            handle.write(b"original")
        return path

    def test_a_worker_without_the_upscaler_refuses_by_name(self):
        with self.assertRaises(upscale.InvalidUpscale):
            upscale.upscale(self._clip(), "1080p", [tempfile.gettempdir()], run=Runner(), env={})

    def test_1080p_runs_seedvr2_at_the_target_height_then_cuts_to_the_exact_frame_and_replaces_the_clip(self):
        clip = self._clip()
        runner = Runner(tempfile.gettempdir())
        ticks = iter([10.0, 71.5])
        facts = upscale.upscale(clip, "1080p", [os.path.dirname(clip)], run=runner, env=ENV, clock=lambda: next(ticks))
        probe, seedvr, ffmpeg = runner.calls
        self.assertEqual("ffprobe", probe[0])
        self.assertEqual(["/opt/up/bin/python", "/opt/seedvr2/inference_cli.py", os.path.realpath(clip)], seedvr[:3])
        self.assertEqual("1080", seedvr[seedvr.index("--resolution") + 1])
        self.assertEqual(upscale.DIT, seedvr[seedvr.index("--dit_model") + 1])
        self.assertEqual("/models/seedvr2", seedvr[seedvr.index("--model_dir") + 1])
        self.assertNotIn("--vae_encode_tiled", seedvr)  # 1080p fits without tiling
        self.assertIn("scale=-2:1080:flags=lanczos,crop=1920:1080", ffmpeg)
        self.assertEqual(b"upscaled", open(clip, "rb").read())  # replaced in place
        self.assertEqual({"upscaleModel": "seedvr2-3b", "upscaleFrom": "1280x704", "upscaleTo": "1920x1080", "upscaleSeconds": 61.5}, facts)

    def test_4k_tiles_the_vae_and_leaves_in_3840x2160(self):
        runner = Runner(tempfile.gettempdir())
        facts = upscale.upscale(self._clip(), "2160p", [self.folder], run=runner, env=ENV)
        seedvr, ffmpeg = runner.calls[1], runner.calls[2]
        self.assertEqual("2160", seedvr[seedvr.index("--resolution") + 1])
        self.assertIn("--vae_encode_tiled", seedvr)
        self.assertIn("--vae_decode_tiled", seedvr)
        self.assertIn("scale=-2:2160:flags=lanczos,crop=3840:2160", ffmpeg)
        self.assertEqual("3840x2160", facts["upscaleTo"])


class TestPathGuard(unittest.TestCase):
    def test_a_clip_outside_the_workers_output_directories_is_refused_before_anything_runs(self):
        runner = Runner()
        with self.assertRaises(upscale.InvalidUpscale):
            upscale.upscale("/etc/passwd", "1080p", [tempfile.gettempdir()], run=runner, env=ENV)
        self.assertEqual([], runner.calls)

    def test_a_path_that_climbs_out_of_the_root_is_refused(self):
        root = tempfile.mkdtemp()
        with self.assertRaises(upscale.InvalidUpscale):
            upscale.upscale(os.path.join(root, "..", "x.mp4"), "1080p", [root], run=Runner(), env=ENV)


class TestWiring(unittest.TestCase):
    """main.apply_upscale: a clip with no `upscale` is untouched; with one, it is replaced and the metrics say so."""

    def _main(self):
        import app.main as main  # imported late: test_main stubs pika before app.telemetry loads

        return main

    def test_without_upscale_nothing_runs(self):
        from unittest.mock import patch

        metrics = {"width": 1280, "height": 704}
        with patch("app.upscale.upscale") as run:
            self._main().apply_upscale({"prompt": "x"}, "/tmp/clip.mp4", metrics)
        run.assert_not_called()
        self.assertEqual({"width": 1280, "height": 704}, metrics)

    def test_with_1080p_the_clip_is_upscaled_and_the_metrics_carry_the_new_size_and_the_facts(self):
        from unittest.mock import patch

        facts = {"upscaleModel": "seedvr2-3b", "upscaleFrom": "1280x704", "upscaleTo": "1920x1080", "upscaleSeconds": 66.0}
        metrics = {"width": 1280, "height": 704}
        with patch("app.upscale.upscale", return_value=facts) as run:
            self._main().apply_upscale({"upscale": "1080p"}, "/tmp/clip.mp4", metrics)
        self.assertEqual(("/tmp/clip.mp4", "1080p"), run.call_args.args[:2])
        self.assertEqual((1920, 1080, "seedvr2-3b"), (metrics["width"], metrics["height"], metrics["upscaleModel"]))


if __name__ == "__main__":
    unittest.main()
