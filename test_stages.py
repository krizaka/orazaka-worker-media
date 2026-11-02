"""Deterministic progress (orazaka-worker-media#64): the named stages, in order, with real values."""

import unittest

from app import stages
from app.telemetry import progress_payload


class Clock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


def reporter(history=None, clock=None):
    sent = []
    clock = clock or Clock()
    rep = stages.StageReporter(
        "job-1", "wan2.2-ti2v-5b", "cuda", history or stages.EtaHistory(), send=sent.append, clock=clock
    )
    return rep, sent, clock


class TestPayload(unittest.TestCase):
    def test_the_stage_travels_with_its_real_values(self):
        payload = progress_payload("j", 40, stage="STREAMING_WEIGHTS_NVME", weights_percent=40, eta_seconds=7, at=12)
        self.assertEqual(
            {
                "jobId": "j",
                "progress": 40,
                "stage": "STREAMING_WEIGHTS_NVME",
                "weightsPercent": 40,
                "etaSeconds": 7,
                "at": 12,
            },
            payload,
        )

    def test_an_unknown_stage_or_a_bad_value_is_left_out(self):
        payload = progress_payload("j", 5, stage="DAYDREAMING", weights_percent=-3, eta_seconds=None)
        self.assertEqual({"jobId": "j", "progress": 5}, payload)

    def test_an_old_consumer_still_reads_the_core_keys(self):
        payload = progress_payload("j", 5, "generating", 1, 4, stage="INFERENCE_STEP")
        self.assertEqual(("j", 5, "generating"), (payload["jobId"], payload["progress"], payload["phase"]))


class TestOrder(unittest.TestCase):
    def test_the_stages_come_in_order_once_each(self):
        rep, sent, _ = reporter()
        rep.wake()
        rep.wake()
        rep.weights(10, 100)
        rep.step(1, 4)
        rep.step(2, 4)
        rep.verify()
        rep.verify()
        rep.completed()
        rep.completed()
        names = [s["stage"] for s in sent]
        self.assertEqual(
            [
                "COLD_START_WAKEUP",
                "STREAMING_WEIGHTS_NVME",
                "INFERENCE_STEP",
                "INFERENCE_STEP",
                "SANDBOX_VERIFICATION",
                "COMPLETED",
            ],
            names,
        )

    def test_a_stage_never_goes_back(self):
        rep, sent, _ = reporter()
        rep.step(1, 4)
        rep.weights(50, 100)  # late: the job is already generating
        rep.wake()
        self.assertEqual(["INFERENCE_STEP"], [s["stage"] for s in sent])

    def test_the_timestamps_are_monotonic(self):
        rep, sent, clock = reporter()
        rep.wake()
        clock.tick(2)
        rep.weights(1, 2)
        clock.tick(-5)  # a clock stepping back must not reorder the events
        rep.step(1, 2)
        ats = [s["at"] for s in sent]
        self.assertEqual(sorted(ats), ats)


class TestWeights(unittest.TestCase):
    def test_the_percentage_is_the_bytes_read_and_never_goes_down(self):
        rep, sent, _ = reporter()
        rep.weights(25, 100)
        rep.weights(10, 100)  # a parallel reader reporting late
        rep.weights(100, 100)
        self.assertEqual([25, 25, 100], [s["weightsPercent"] for s in sent if "weightsPercent" in s][:3])

    def test_an_unknown_total_says_nothing_instead_of_inventing_a_percentage(self):
        rep, sent, _ = reporter()
        rep.weights(10, 0)
        self.assertEqual([], sent)


class TestEta(unittest.TestCase):
    def test_no_eta_without_history(self):
        rep, sent, _ = reporter()
        rep.weights(10, 100)
        self.assertNotIn("etaSeconds", sent[-1])

    def test_the_load_eta_comes_from_the_last_loads_of_that_model_and_hardware(self):
        history = stages.EtaHistory()
        for seconds in (10, 12, 14):
            history.record("wan2.2-ti2v-5b", "cuda", "load", seconds)
        history.record("wan2.2-ti2v-5b", "mps", "load", 999)
        rep, sent, _ = reporter(history)
        rep.wake()
        self.assertEqual(12, sent[-1]["etaSeconds"])

    def test_the_history_keeps_only_the_last_n(self):
        history = stages.EtaHistory(keep=3)
        for seconds in (100, 1, 1, 1):
            history.record("m", "cuda", "load", seconds)
        self.assertEqual(1, history.estimate("m", "cuda", "load"))

    def test_the_step_eta_slides_over_the_latest_steps(self):
        rep, sent, clock = reporter()
        rep.step(1, 10)  # no previous step: no duration yet
        self.assertNotIn("etaSeconds", sent[-1])
        for _ in range(4):  # four slow steps of 10 s
            clock.tick(10)
            rep.step(len(sent) + 1, 10)
        for _ in range(stages.STEP_WINDOW):  # then the window fills with 2 s steps
            clock.tick(2)
            rep.step(len(sent) + 1, 10)
        last = sent[-1]
        self.assertEqual(last["steps"] - last["step"], 10 - last["step"])
        self.assertEqual(2 * (10 - last["step"]), last["etaSeconds"])

    def test_completing_records_the_durations_for_next_time(self):
        history = stages.EtaHistory()
        rep, _, clock = reporter(history)
        rep.wake()
        clock.tick(30)
        rep.step(1, 2)  # the load ended when generation began
        self.assertEqual(30, history.estimate("wan2.2-ti2v-5b", "cuda", "load"))


class TestConsumerWiring(unittest.TestCase):
    def test_a_video_job_wakes_then_completes_before_it_is_done(self):
        import json
        from unittest.mock import MagicMock, patch

        import app.consumer as consumer

        consumer._SEEN.clear()
        order = []
        rep = stages.StageReporter("job-w", "m", "cuda", stages.EtaHistory(), send=lambda p: order.append(p["stage"]))
        body = json.dumps({"jobId": "job-w", "userId": "u", "payload": {"prompt": "x"}}).encode()
        with (
            patch.object(stages, "reporter_for", return_value=rep),
            patch.object(consumer, "_generate", return_value=({"url": "u"}, {})),
            patch.object(consumer, "send_progress_update"),
            patch.object(consumer, "send_job_done", side_effect=lambda *a, **k: order.append("done")),
        ):
            consumer._on_message(MagicMock(), MagicMock(delivery_tag=1, routing_key="job.video.generate"), None, body)
        self.assertEqual(["COLD_START_WAKEUP", "COMPLETED", "done"], order)


if __name__ == "__main__":
    unittest.main()
