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


if __name__ == "__main__":
    unittest.main()
