"""The consumer survives a dropped tunnel: it reconnects, and a redelivered job is never rendered twice (no broker, no GPU)."""

from __future__ import annotations

import json
import types
import unittest
from unittest.mock import MagicMock, patch


class FakeAMQPError(Exception):
    """Stands for pika.exceptions.AMQPError (test_main replaces the pika module by a mock for the whole run)."""


def fake_pika():
    return types.SimpleNamespace(exceptions=types.SimpleNamespace(AMQPError=FakeAMQPError))


class TestReconnect(unittest.TestCase):
    def test_a_lost_connection_is_retried_with_a_growing_wait_until_it_holds(self):
        import app.consumer as consumer

        calls, waits = [], []

        def consume():
            calls.append(1)
            if len(calls) < 3:
                raise FakeAMQPError("stream lost")

        with patch.object(consumer, "pika", fake_pika()):
            consumer.run_consumer(retry_seconds=5, consume=consume, sleep=waits.append)
        self.assertEqual(3, len(calls))
        self.assertEqual([5, 10], waits)

    def test_the_wait_is_capped(self):
        import app.consumer as consumer

        waits, calls = [], []

        def consume():
            calls.append(1)
            if len(calls) < 12:
                raise OSError("tunnel")

        with patch.object(consumer, "pika", fake_pika()):
            consumer.run_consumer(retry_seconds=5, consume=consume, sleep=waits.append)
        self.assertEqual(30.0, max(waits))

    def test_a_clean_stop_ends_the_loop(self):
        import app.consumer as consumer

        consume = MagicMock()
        with patch.object(consumer, "pika", fake_pika()):
            consumer.run_consumer(consume=consume, sleep=lambda s: self.fail("no wait on a clean stop"))
        consume.assert_called_once()


class TestRedelivery(unittest.TestCase):
    def setUp(self):
        import app.consumer as consumer

        consumer._SEEN.clear()

    def test_a_job_is_first_delivered_once(self):
        import app.consumer as consumer

        self.assertTrue(consumer._first_delivery("job-1"))
        self.assertFalse(consumer._first_delivery("job-1"))
        self.assertTrue(consumer._first_delivery("job-2"))
        self.assertTrue(consumer._first_delivery(None))  # no id: nothing to dedupe, the job is refused later

    def test_a_job_seen_long_ago_is_forgotten(self):
        import app.consumer as consumer

        consumer._first_delivery("old")
        consumer._SEEN["old"] -= consumer._SEEN_TTL_SECONDS + 1
        self.assertTrue(consumer._first_delivery("old"))

    def test_a_redelivered_job_is_acknowledged_once_and_not_rendered_again(self):
        import app.consumer as consumer

        channel = MagicMock()
        method = MagicMock(delivery_tag=7, routing_key="job.video.generate")
        body = json.dumps({"jobId": "job-9", "userId": "u-1", "payload": {"prompt": "x"}}).encode()
        consumer._first_delivery("job-9")  # the first delivery is still rendering
        with patch.object(consumer, "pika", fake_pika()), patch.object(consumer, "_generate") as render:
            consumer._on_message(channel, method, None, body)
        render.assert_not_called()
        self.assertEqual(1, channel.connection.add_callback_threadsafe.call_count)

    def test_an_ack_on_a_dead_connection_is_not_a_crash(self):
        import app.consumer as consumer

        channel = MagicMock()
        channel.connection.add_callback_threadsafe.side_effect = FakeAMQPError("gone")
        method = MagicMock(delivery_tag=1, routing_key="job.video.generate")
        body = json.dumps({"jobId": "job-10", "userId": "u-1", "payload": {"prompt": "x"}}).encode()
        with (
            patch.object(consumer, "pika", fake_pika()),
            patch.object(consumer, "_generate", side_effect=ValueError("boom")),
            patch.object(consumer, "send_job_error"),
        ):
            consumer._on_message(channel, method, None, body)  # must not raise


if __name__ == "__main__":
    unittest.main()
