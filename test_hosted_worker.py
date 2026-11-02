"""The hosted-only worker: a cheap CPU consumer for the models a provider serves (no GPU, no torch), on its own queue."""

from __future__ import annotations

import unittest


class TestHostedWorker(unittest.TestCase):
    def test_the_mode_is_read_from_the_environment(self):
        import app.consumer as consumer

        self.assertTrue(consumer.hosted_only({"MEDIA_WORKER_HOSTED_ONLY": "true"}))
        self.assertTrue(consumer.hosted_only({"MEDIA_WORKER_HOSTED_ONLY": " TRUE "}))
        self.assertFalse(consumer.hosted_only({}))
        self.assertFalse(consumer.hosted_only({"MEDIA_WORKER_HOSTED_ONLY": "false"}))

    def test_it_has_its_own_queue_name_and_registers_itself_as_another_worker(self):
        import app.consumer as consumer

        base = {
            "name": "orazaka-worker-media",
            "family": "media",
            "bindings": ["job.video.*", "job.compose.*"],
            "version": "1.0.0",
            "concurrency": 1,
        }
        self.assertEqual("orazaka.jobs.video", consumer.video_queue({}))
        self.assertEqual("orazaka.jobs.video.hosted", consumer.video_queue({"MEDIA_WORKER_HOSTED_ONLY": "true"}))
        self.assertEqual(base, consumer.apply_hosted(base, {}))
        hosted = consumer.apply_hosted(base, {"MEDIA_WORKER_HOSTED_ONLY": "true"})
        # its bindings never match the GPU worker's `job.video.*`: a hosted job is not copied to the GPU queue
        self.assertEqual(["job.hostedvideo.*"], hosted["bindings"])
        self.assertEqual("orazaka-worker-media-hosted", hosted["name"])
        self.assertEqual("1.0.0", hosted["version"])

    def test_the_binding_never_matches_the_gpu_workers(self):
        import re

        gpu = re.compile(r"^job\.video\.[^.]+$")
        self.assertIsNone(gpu.match("job.hostedvideo.generate"))
        self.assertIsNotNone(gpu.match("job.video.generate"))

    def test_a_model_run_here_is_refused_by_name_never_started(self):
        import app.consumer as consumer

        class Remote:
            remote_model = "alibaba-cloud/wan2.2-t2v-a14b"

        class Local:
            remote_model = None

        models = {"wan2-2-t2v-a14b": Remote(), "wan2.2-t2v-a14b": Local()}
        env = {"MEDIA_WORKER_HOSTED_ONLY": "true"}
        self.assertIsNone(consumer.hosted_refusal("wan2-2-t2v-a14b", env, find=models.__getitem__))
        message = consumer.hosted_refusal("wan2.2-t2v-a14b", env, find=models.__getitem__)
        self.assertIn("wan2.2-t2v-a14b", message)
        self.assertIn("hosted", message)
        self.assertIn("GPU", consumer.hosted_refusal("", env, find=models.__getitem__))
        self.assertIsNone(
            consumer.hosted_refusal("wan2.2-t2v-a14b", {}, find=models.__getitem__), "the GPU worker runs anything"
        )


if __name__ == "__main__":
    unittest.main()


class TestLeanRequirements(unittest.TestCase):
    def test_the_hosted_worker_needs_no_gpu_library(self):
        import pathlib

        text = (pathlib.Path(__file__).parent / "requirements-hosted.txt").read_text()
        names = [l.split("#")[0].strip().lower() for l in text.splitlines() if l.strip() and not l.startswith("#")]
        for heavy in ("torch", "torchvision", "diffusers", "mlx", "mflux", "transformers", "accelerate"):
            self.assertNotIn(heavy, names)
        for needed in ("pika", "pyyaml", "cryptography"):
            self.assertIn(needed, names)
