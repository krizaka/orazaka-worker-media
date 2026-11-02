"""Wan 2.2 A14B on DigitalOcean serverless inference: what is asked of the provider, and how it fails (no network)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from app import main, remote_video, video_models

MODEL = "wan2-2-t2v-a14b"


class Recorder:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, method, url, key, body=None, timeout=60):
        self.calls.append((method, url, key, body))
        status, payload = self.answers.pop(0)
        return status, payload if isinstance(payload, bytes) else json.dumps(payload).encode()


ENV = {
    "AI_INFERENCE_KEY": "k-test",
    "AI_INFERENCE_URL": "https://provider.test",
    "AI_VIDEO_DAILY_RUNS": "-1",
    "AI_SPEND_DAILY_BUDGET_USD": "-1",
}


class TestCatalogue(unittest.TestCase):
    def test_the_model_is_a_provider_model_with_a_cost(self):
        model = video_models.find(MODEL)
        self.assertEqual("remote", model.pipeline)
        self.assertEqual("do-inference", model.provider)
        self.assertEqual(0.60, model.cost_usd)
        self.assertEqual("Remote", main.MODEL_REGISTRY[MODEL])

    def test_aspect_picks_the_providers_sizes(self):
        model = video_models.find(MODEL)
        # Measured: the gateway returns 832x480 whatever size is asked, so only that shape is declared.
        self.assertEqual((832, 480), model.size_for("16:9"))
        for refused in ("9:16", "1:1"):
            with self.assertRaises(video_models.UnsupportedAspect):
                model.size_for(refused)


class TestGenerate(unittest.TestCase):
    def test_submits_polls_and_downloads(self):
        model = video_models.find(MODEL)
        call = Recorder(
            [
                (200, {"id": "job_1", "status": "processing"}),
                (200, {"id": "video_1", "status": "processing"}),
                (200, {"id": "video_1", "status": "completed", "output": None}),
                (200, b"MP4BYTES"),
            ]
        )
        video, facts = remote_video.generate(model, "tea", 832, 480, 20261011, env=ENV, sleep=lambda s: None, call=call)
        self.assertEqual(b"MP4BYTES", video)
        method, url, key, body = call.calls[0]
        self.assertEqual(("POST", "https://provider.test/v1/videos", "k-test"), (method, url, key))
        self.assertEqual({"model": MODEL, "prompt": "tea", "size": "832x480", "fps": 16, "seed": 20261011}, body)
        self.assertEqual("https://provider.test/v1/videos/job_1", call.calls[1][1])
        self.assertEqual("https://provider.test/v1/videos/video_1/content", call.calls[3][1])
        self.assertEqual(
            {"provider": "do-inference", "providerJobId": "video_1", "costUsd": 0.6},
            {k: facts[k] for k in ("provider", "providerJobId", "costUsd")},
        )

    def test_a_failed_job_raises(self):
        call = Recorder(
            [(200, {"id": "j", "status": "processing"}), (200, {"id": "j", "status": "failed", "error": "nsfw"})]
        )
        with self.assertRaisesRegex(remote_video.RemoteVideoError, "failed"):
            remote_video.generate(
                video_models.find(MODEL), "x", 720, 1280, None, env=ENV, sleep=lambda s: None, call=call
            )

    def test_no_seed_is_not_sent(self):
        body = remote_video.request_body(video_models.find(MODEL), "x", 1280, 720, None)
        self.assertNotIn("seed", body)

    def test_a_missing_key_is_refused_before_any_call(self):
        with self.assertRaisesRegex(remote_video.RemoteVideoError, "AI_INFERENCE_KEY"):
            remote_video.generate(video_models.find(MODEL), "x", 720, 1280, None, env={}, call=Recorder([]))

    def test_a_job_that_never_ends_times_out(self):
        ticks = iter(range(0, 100000, 400))
        call = Recorder([(200, {"id": "j", "status": "processing"})] * 10)
        with self.assertRaisesRegex(remote_video.RemoteVideoError, "did not finish"):
            remote_video.generate(
                video_models.find(MODEL),
                "x",
                720,
                1280,
                None,
                env=ENV,
                sleep=lambda s: None,
                clock=lambda: next(ticks),
                call=call,
            )


class TestRun(unittest.TestCase):
    def test_a_run_writes_the_clip_with_its_provenance(self):
        def fake(model, prompt, w, h, seed, progress=None, **_):
            return b"VIDEO", {
                "provider": "do-inference",
                "providerJobId": "v9",
                "providerSeconds": 41.0,
                "costUsd": 0.6,
            }

        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"UPLOAD_DIR": root}):
            out = os.path.join(root, "u", "j", "output", "video.mp4")
            body = main.run_remote(
                {"prompt": "tea", "aspect": "16:9", "seed": "7", "output_path": out}, MODEL, generate=fake
            )
            self.assertEqual(b"VIDEO", open(out, "rb").read())
        metrics = body["metrics"]
        self.assertEqual(
            (832, 480, 16, "7", "16:9"),
            (metrics["width"], metrics["height"], metrics["fps"], metrics["seed"], metrics["aspect"]),
        )
        self.assertEqual("do-inference", metrics["provider"])

    def test_a_starting_image_is_refused_not_ignored(self):
        with self.assertRaises(video_models.InvalidVideoRequest):
            main.run_remote({"prompt": "x", "image": "abc"}, MODEL, generate=lambda *a, **k: (b"", {}))


if __name__ == "__main__":
    unittest.main()
