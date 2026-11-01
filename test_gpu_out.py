"""A clip goes straight to the pod's own output zone (krizaka-internal#2149), never with a site credential."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from app import consumer, gpu_out

ENV = {
    "GPU_OUT_URL": "https://storage.bunnycdn.com/krizaka-gpu-out/2026-11-01/run9",
    "GPU_OUT_HEADER": "AccessKey",
    "GPU_OUT_KEY": "k-test",
    "RUNPOD_POD_ID": "pod123",
}
JOB = {"jobId": "job-1"}


class _Response:
    status = 201

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestTheZone(unittest.TestCase):
    def _clip(self):
        handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        handle.write(b"\x00\x00\x00\x18ftypmp42clip")
        handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name) and os.remove(handle.name))
        return handle.name

    def test_without_a_zone_nothing_is_sent(self):
        self.assertFalse(gpu_out.configured({}))
        self.assertIsNone(gpu_out.put(self._clip(), JOB, "video.mp4", "video/mp4", env={}))

    def test_a_clip_is_put_under_the_pod_with_the_zone_key_only(self):
        with mock.patch("urllib.request.urlopen", return_value=_Response()) as opened:
            key = gpu_out.put(self._clip(), JOB, "video.mp4", "video/mp4", env=ENV)
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, f"{ENV['GPU_OUT_URL']}/pod123/job-1-video.mp4")
        self.assertEqual(request.get_method(), "PUT")
        self.assertEqual(request.get_header("Accesskey"), "k-test")
        self.assertEqual(key, "2026-11-01/run9/pod123/job-1-video.mp4")

    def test_a_refused_write_fails_the_job(self):
        import urllib.error

        refused = urllib.error.HTTPError(ENV["GPU_OUT_URL"], 401, "no", {}, None)
        with mock.patch("urllib.request.urlopen", side_effect=refused):
            with self.assertRaises(gpu_out.GpuOutError):
                gpu_out.put(self._clip(), JOB, "video.mp4", "video/mp4", env=ENV)

    def test_an_unsafe_name_is_refused(self):
        with self.assertRaises(gpu_out.GpuOutError):
            gpu_out.put(self._clip(), JOB, "../x.mp4", "video/mp4", env=ENV)

    def test_the_store_writes_the_zone_before_the_sink(self):
        order = []
        path = self._clip()
        with mock.patch.object(consumer.gpu_out, "put", side_effect=lambda *a, **k: order.append("zone")), \
                mock.patch.object(consumer.asset_sink, "sink_url", return_value="http://core"), \
                mock.patch.object(consumer.asset_sink, "upload", side_effect=lambda *a, **k: order.append("sink") or "/api/v1/assets/j/video.mp4"):
            url = consumer._store(path, JOB, "video.mp4", "video/mp4", "/uploads/x")
        self.assertEqual(order, ["zone", "sink"])
        self.assertEqual(url, "/api/v1/assets/j/video.mp4")


if __name__ == "__main__":
    unittest.main()
