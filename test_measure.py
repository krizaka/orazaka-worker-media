import json
import os
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


class TestLoadModes(unittest.TestCase):
    def test_the_load_modes_are_the_four_the_decision_compares(self):
        import re
        source = Path(measure.__file__).read_text()
        modes = re.search(r'choices=\[([^\]]+)\], default="volume"', source).group(1)
        self.assertEqual(['"volume"', '"stage"', '"warm-stage"', '"prefetch"'], [m.strip() for m in modes.split(",")])


class FakeModel:
    name = "fake"
    pipeline = "wan"
    text_encoder_from = None


def run_load(mode, stage_dir, events, model=None, donor=None):
    """measure_load with the GPU world replaced: records the order of resolve, stage, evict and load."""
    from unittest import mock

    model = model or FakeModel()
    tmp = Path(tempfile.mkdtemp())
    snapshot = tmp / "snap"
    snapshot.mkdir()
    (snapshot / "w.bin").write_bytes(b"x" * 10)
    env = {"MEDIA_MODELS_STAGE_DIR": stage_dir} if stage_dir else {}
    snapshots = {id(model): snapshot}
    if donor:
        snapshots[id(donor)] = tmp / "donor"
        (tmp / "donor").mkdir()
    fake_cache = mock.Mock()
    fake_cache.stage.side_effect = lambda src, dst: events.append(("stage", src.name, dst.parent))
    with mock.patch.dict(os.environ, env, clear=False), \
            mock.patch.object(measure, "_resolve", lambda module, names: events.append(("resolve", module, names))), \
            mock.patch.object(measure, "_snapshot", lambda m: snapshots[id(m)]), \
            mock.patch.object(measure, "evict", lambda path: events.append(("evict", Path(path).name)) or 0), \
            mock.patch("app.video_models.find", lambda name, *a: donor if donor and name == "donor" else model), \
            mock.patch("app.model_cache.stage", fake_cache.stage), \
            mock.patch("app.wan_generator._load", lambda m, from_image=False: events.append(("load",))):
        if not stage_dir:
            os.environ.pop("MEDIA_MODELS_STAGE_DIR", None)
        line = measure.measure_load("fake", mode)
        return json.loads(line), os.environ.get("MEDIA_MODELS_STAGE_DIR")


class TestLoadBehaviour(unittest.TestCase):
    def test_the_classes_the_loader_resolves_are_resolved_before_the_clock(self):
        events = []
        run_load("volume", None, events)
        self.assertIn(("resolve", "diffusers", ("AutoencoderKLWan", "WanImageToVideoPipeline", "WanPipeline")), events)
        self.assertLess(max(i for i, e in enumerate(events) if e[0] == "resolve"), events.index(("load",)))

    def test_the_text_encoder_class_is_resolved_when_a_donor_is_configured(self):
        model = FakeModel()
        model.text_encoder_from = "donor"
        events = []
        run_load("volume", None, events, model=model, donor=FakeModel())
        self.assertIn(("resolve", "transformers", ("UMT5EncoderModel",)), events)

    def test_ltx_resolves_its_own_pipeline_class(self):
        model = FakeModel()
        model.pipeline = "ltx"
        events = []
        run_load("volume", None, events, model=model)
        self.assertIn(("resolve", "diffusers", ("LTXConditionPipeline",)), events)
        self.assertNotIn(("resolve", "diffusers", ("AutoencoderKLWan", "WanImageToVideoPipeline", "WanPipeline")), events)

    def test_warm_stage_completes_the_copy_before_the_timer_and_keeps_staging_on(self):
        events = []
        _, stage_dir = run_load("warm-stage", "/nvme", events)
        self.assertEqual("/nvme", stage_dir)
        self.assertEqual(("stage", "snap", Path("/nvme")), next(e for e in events if e[0] == "stage"))
        self.assertLess(next(i for i, e in enumerate(events) if e[0] == "stage"), events.index(("load",)))

    def test_warm_stage_stages_the_donor_too(self):
        model = FakeModel()
        model.text_encoder_from = "donor"
        events = []
        run_load("warm-stage", "/nvme", events, model=model, donor=FakeModel())
        self.assertEqual(["snap", "donor"], [e[1] for e in events if e[0] == "stage"])

    def test_stage_starts_from_no_copy_and_prepares_nothing(self):
        events = []
        stage_dir = tempfile.mkdtemp()
        (Path(stage_dir) / "snap").mkdir()
        run_load("stage", stage_dir, events)
        self.assertFalse((Path(stage_dir) / "snap").exists())
        self.assertEqual([], [e for e in events if e[0] == "stage"])

    def test_both_staged_modes_refuse_a_missing_stage_directory(self):
        for mode in ("stage", "warm-stage"):
            with self.assertRaises(SystemExit) as refused:
                run_load(mode, None, [])
            self.assertIn("MEDIA_MODELS_STAGE_DIR", str(refused.exception))

    def test_volume_drops_the_stage_directory(self):
        _, stage_dir = run_load("volume", "/nvme", [])
        self.assertIsNone(stage_dir)


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
