"""The daily ceilings on the paid video provider: clips per day and USD per day, refused before any call."""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest

from app import remote_video, spend, video_models

MODEL = "wan2-2-t2v-a14b"
DAY = dt.datetime(2026, 10, 15, 12, 0, tzinfo=dt.timezone.utc)


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, method, url, key, body=None, timeout=60):
        self.calls.append(url)
        return 200, json.dumps({"id": "j1", "status": "completed", "output": None}).encode()


class TestSpend(unittest.TestCase):
    def setUp(self):
        handle, self.state = tempfile.mkstemp(suffix=".json")
        os.close(handle)
        os.remove(self.state)
        self.addCleanup(lambda: os.path.exists(self.state) and os.remove(self.state))

    def env(self, **extra):
        return {"AI_SPEND_STATE_FILE": self.state, **extra}

    def test_the_n_plus_one_th_clip_of_the_day_is_refused(self):
        env = self.env(AI_VIDEO_DAILY_RUNS="2", AI_SPEND_DAILY_BUDGET_USD="100")
        spend.reserve(0.6, env, DAY)
        spend.reserve(0.6, env, DAY)
        with self.assertRaisesRegex(spend.SpendRefused, "limit of 2"):
            spend.reserve(0.6, env, DAY)

    def test_the_budget_trips_the_breaker(self):
        env = self.env(AI_VIDEO_DAILY_RUNS="100", AI_SPEND_DAILY_BUDGET_USD="1.0")
        spend.reserve(0.6, env, DAY)
        with self.assertRaisesRegex(spend.SpendRefused, "budget"):
            spend.reserve(0.6, env, DAY)

    def test_the_next_day_starts_again_and_a_restart_does_not_give_the_day_back(self):
        env = self.env(AI_VIDEO_DAILY_RUNS="1", AI_SPEND_DAILY_BUDGET_USD="100")
        spend.reserve(0.6, env, DAY)
        with self.assertRaises(spend.SpendRefused):  # counters come from the file, not from memory
            spend.reserve(0.6, env, DAY)
        spend.reserve(0.6, env, DAY + dt.timedelta(days=1))

    def test_a_negative_ceiling_switches_it_off(self):
        env = self.env(AI_VIDEO_DAILY_RUNS="-1", AI_SPEND_DAILY_BUDGET_USD="-1")
        for _ in range(20):
            spend.reserve(0.6, env, DAY)

    def test_a_refused_clip_never_reaches_the_provider(self):
        env = self.env(AI_INFERENCE_KEY="k", AI_INFERENCE_URL="https://provider.test", AI_VIDEO_DAILY_RUNS="0")
        call = Recorder()
        with self.assertRaisesRegex(remote_video.RemoteVideoError, "daily limit"):
            remote_video.generate(video_models.find(MODEL), "x", 720, 1280, None, env=env, call=call)
        self.assertEqual(call.calls, [])


if __name__ == "__main__":
    unittest.main()
