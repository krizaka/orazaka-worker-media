import json
import tempfile
import unittest
from pathlib import Path

from app import measure


class TestMeasure(unittest.TestCase):
    def test_the_result_line_says_seconds_size_and_rate(self):
        line = json.loads(measure.result_line("stage", "wan", 12.0, 34_000_000_000))
        self.assertEqual({"mode": "stage", "model": "wan", "seconds": 12.0, "gb": 34.0, "gbPerSecond": 2.83}, line)

    def test_extra_facts_travel_with_it(self):
        line = json.loads(measure.result_line("render", "wan", 80.0, 0, {"steps": 20}))
        self.assertEqual(20, line["steps"])

    def test_evict_counts_the_bytes_and_never_fails(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "a.bin").write_bytes(b"x" * 100)
        self.assertEqual(100, measure.evict(tmp))
        self.assertEqual(0, measure.evict(tmp / "absent"))


class FakeTransformer:
    def __init__(self):
        self.calls = []

    def enable_layerwise_casting(self, **kwargs):
        self.calls.append(("casting", kwargs))

    def enable_cache(self, config):
        self.calls.append(("cache", config))


class FakePipe:
    def __init__(self):
        self.transformer = FakeTransformer()


class TestAcceleration(unittest.TestCase):
    def test_none_is_the_baseline_and_changes_nothing(self):
        pipe = FakePipe()
        self.assertEqual((), measure.parse_accel("none"))
        measure.apply_acceleration(pipe, (), 0.2, fp8_dtype="f8", cache_config=lambda threshold: threshold)
        self.assertEqual([], pipe.transformer.calls)

    def test_fp8_stores_the_weights_in_fp8_and_computes_in_bf16(self):
        pipe = FakePipe()
        measure.apply_acceleration(pipe, measure.parse_accel("fp8"), 0.2, fp8_dtype="f8", cache_config=lambda threshold: threshold, compute_dtype="bf16")
        self.assertEqual([("casting", {"storage_dtype": "f8", "compute_dtype": "bf16"})], pipe.transformer.calls)

    def test_firstblock_takes_the_threshold(self):
        pipe = FakePipe()
        measure.apply_acceleration(pipe, measure.parse_accel("firstblock"), 0.15, fp8_dtype="f8", cache_config=lambda threshold: ("cfg", threshold))
        self.assertEqual([("cache", ("cfg", 0.15))], pipe.transformer.calls)

    def test_both_apply_in_a_fixed_order_fp8_first(self):
        pipe = FakePipe()
        measure.apply_acceleration(pipe, measure.parse_accel("firstblock+fp8"), 0.2, fp8_dtype="f8", cache_config=lambda threshold: threshold, compute_dtype="bf16")
        self.assertEqual(["casting", "cache"], [name for name, _ in pipe.transformer.calls])

    def test_an_unknown_acceleration_is_refused_by_name(self):
        with self.assertRaises(SystemExit) as refused:
            measure.parse_accel("teacache")
        self.assertIn("teacache", str(refused.exception))

    def test_the_result_names_the_acceleration(self):
        line = json.loads(measure.result_line("render", "wan", 50.0, 0, {"accel": "fp8+firstblock"}))
        self.assertEqual("fp8+firstblock", line["accel"])


if __name__ == "__main__":
    unittest.main()
