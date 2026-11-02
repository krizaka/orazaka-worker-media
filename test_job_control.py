"""Cancelling a job on the worker (orazaka-web-client#85): the registry, the routing key, the listener loop."""

from __future__ import annotations

import unittest

from app import job_control


class TestRegistry(unittest.TestCase):
    def setUp(self):
        job_control._CANCELLED.clear()

    def test_a_cancelled_job_raises_at_its_next_step(self):
        job_control.request_cancel("job-1")
        self.assertTrue(job_control.is_cancelled("job-1"))
        with self.assertRaises(job_control.JobCancelled):
            job_control.raise_if_cancelled("job-1")

    def test_another_job_is_not_touched(self):
        job_control.request_cancel("job-1")
        job_control.raise_if_cancelled("job-2")  # no raise
        self.assertFalse(job_control.is_cancelled(None))

    def test_a_cancel_is_forgotten_after_its_ttl(self):
        job_control.request_cancel("old", now=1000.0)
        job_control.request_cancel("new", now=1000.0 + job_control._TTL_SECONDS + 1)
        self.assertFalse(job_control.is_cancelled("old"))
        self.assertTrue(job_control.is_cancelled("new"))


class TestRoutingKey(unittest.TestCase):
    def test_the_id_is_the_last_segment(self):
        self.assertEqual("abc-123", job_control.job_id_from_routing_key("job.cancel.abc-123"))

    def test_other_keys_are_not_cancellations(self):
        self.assertIsNone(job_control.job_id_from_routing_key("job.video.generate"))
        self.assertIsNone(job_control.job_id_from_routing_key("job.cancel."))
        self.assertIsNone(job_control.job_id_from_routing_key(""))


class TestListener(unittest.TestCase):
    def test_a_lost_link_is_retried_with_a_growing_wait(self):
        calls, waits = [], []

        def listen():
            calls.append(1)
            if len(calls) < 3:
                raise OSError("tunnel")

        job_control.listen_for_cancels(retry_seconds=5, listen=listen, sleep=waits.append)
        self.assertEqual(3, len(calls))
        self.assertEqual([5, 10], waits)

    def test_the_callback_records_and_acknowledges(self):
        job_control._CANCELLED.clear()

        class Channel:
            acked = None

            def basic_ack(self, delivery_tag):
                self.acked = delivery_tag

        class Method:
            routing_key = "job.cancel.j9"
            delivery_tag = 7

        channel = Channel()
        job_control._on_cancel(channel, Method(), None, b"")
        self.assertTrue(job_control.is_cancelled("j9"))
        self.assertEqual(7, channel.acked)


if __name__ == "__main__":
    unittest.main()


class TestConsumerCancel(unittest.TestCase):
    """The consumer's two ways of stopping: a job dropped from the queue, a running job that is told to stop."""

    def setUp(self):
        import app.consumer as consumer

        consumer._SEEN.clear()
        job_control._CANCELLED.clear()

    def _call(self, job_id, generate=None):
        import json
        from unittest.mock import MagicMock, patch

        import app.consumer as consumer

        channel = MagicMock()
        method = MagicMock(delivery_tag=1, routing_key="job.video.generate")
        body = json.dumps({"jobId": job_id, "userId": "u-1", "payload": {"prompt": "x", "holdId": "h-1"}}).encode()
        with (
            patch.object(consumer, "_generate", **({"side_effect": generate} if generate else {})) as render,
            patch.object(consumer, "send_job_cancelled") as cancelled,
            patch.object(consumer, "send_job_error") as errored,
            patch.object(consumer, "send_progress_update"),
            patch.object(consumer, "send_job_done") as done,
        ):
            consumer._on_message(channel, method, None, body)
        return render, cancelled, errored, done

    def test_a_job_cancelled_while_queued_is_dropped_without_a_render(self):
        job_control.request_cancel("job-q")
        render, cancelled, errored, done = self._call("job-q")
        render.assert_not_called()
        cancelled.assert_called_once_with("job-q", hold_id="h-1")
        errored.assert_not_called()
        done.assert_not_called()

    def test_a_running_job_told_to_stop_ends_as_cancelled_not_failed(self):
        def stop(job):
            raise job_control.JobCancelled("stop")

        _, cancelled, errored, done = self._call("job-r", generate=stop)
        cancelled.assert_called_once_with("job-r", hold_id="h-1")
        errored.assert_not_called()
        done.assert_not_called()


class TestProgressPayload(unittest.TestCase):
    def test_phase_and_step_travel_with_the_percentage(self):
        from app.telemetry import progress_payload

        self.assertEqual(
            {"jobId": "j", "progress": 40, "phase": "generating", "step": 4, "steps": 10},
            progress_payload("j", 40, "generating", 4, 10),
        )

    def test_an_unknown_phase_or_a_bad_step_is_left_out(self):
        from app.telemetry import progress_payload

        self.assertEqual({"jobId": "j", "progress": 5}, progress_payload("j", 5, "daydreaming", 0, 0))

    def test_the_wan_generator_lets_a_cancellation_through(self):
        import inspect

        import app.wan_generator as wan

        self.assertIn("except job_control.JobCancelled", inspect.getsource(wan.generate_wan_frames))
