"""LTX-Video 0.9.8 13B distilled: a model of the catalogue that makes a 5 s clip in seconds, from a still or from text (no GPU)."""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from app import video_models, wan_generator

NAME = "ltx-video-0.9.8-13b-distilled"


class TestCatalogue(unittest.TestCase):
    def test_it_is_an_open_weights_image_to_video_model_pinned_by_commit(self):
        model = video_models.find(NAME)
        self.assertEqual("ltx", model.pipeline)
        self.assertEqual("Lightricks/LTX-Video-0.9.8-13B-distilled", model.repo)
        self.assertRegex(model.revision, r"^[0-9a-f]{40}$")
        self.assertTrue(model.image_to_video)
        self.assertFalse(model.requires_image)  # text to video works too
        self.assertEqual((8, 1.0), (model.steps, model.guidance))  # distilled: 8 steps, no guidance

    def test_it_makes_the_sizes_it_was_trained_at_in_multiples_of_32(self):
        model = video_models.find(NAME)
        self.assertEqual((1216, 704), model.size_for("16:9"))
        self.assertEqual((704, 1216), model.size_for("9:16"))
        for width, height in model.sizes.values():
            self.assertEqual((0, 0), (width % 32, height % 32))

    def test_frames_are_counted_8k_plus_1_at_24_fps(self):
        model = video_models.find(NAME)
        self.assertEqual(24, model.fps)
        self.assertEqual(121, model.frames_for(5))  # 5 s
        self.assertEqual(73, model.frames_for(3))
        self.assertEqual(25, model.frames_for(1))
        self.assertEqual(121, model.frames_for(20))  # never past what the model holds

    def test_the_wan_models_keep_their_4k_plus_1(self):
        self.assertEqual(81, video_models.find("wan2.2-t2v-a14b").frames_for(5))


class TestHighQualityWan(unittest.TestCase):
    """The bake-off's winner for people in 9:16: the same I2V A14B checkpoint, undistilled (40 steps, guidance 3.5)."""

    def test_it_shares_the_checkpoint_of_the_distilled_model_and_runs_the_full_schedule(self):
        hq, fast = video_models.find("wan2.2-i2v-a14b-hq"), video_models.find("wan2.2-i2v-a14b")
        self.assertEqual((fast.repo, fast.revision), (hq.repo, hq.revision))  # one 115 GB fetch serves both
        self.assertEqual((40, 3.5, 3.5), (hq.steps, hq.guidance, hq.guidance_two))
        self.assertEqual((16, 81), (hq.fps, hq.max_frames))
        self.assertTrue(hq.image_to_video and hq.requires_image)
        self.assertEqual((1280, 720), hq.size_for("16:9"))
        self.assertEqual((720, 1280), hq.size_for("9:16"))

    def test_the_distilled_model_keeps_its_four_steps_and_its_loras(self):
        fast = video_models.find("wan2.2-i2v-a14b")
        self.assertEqual(4, fast.steps)
        self.assertIsNotNone(fast.lora)

    def test_the_hq_model_loads_no_lora_and_is_on_the_cinema_profile(self):
        self.assertIsNone(video_models.find("wan2.2-i2v-a14b-hq").lora)
        self.assertIn("wan2.2-i2v-a14b-hq", video_models.load_manifest()["profiles"]["cinema"]["video"])


class TestScratchCache(unittest.TestCase):
    """LTX's weights are on the host's own disk (`cache: scratch`): the loader must look there, like the I2V A14B's."""

    def test_the_ltx_pipeline_is_loaded_from_the_scratch_cache(self):
        fake = types.SimpleNamespace(LTXConditionPipeline=MagicMock())
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"diffusers": fake}), patch.object(wan_generator, "_torch", return_value=torch), patch.object(
            wan_generator, "_device", return_value="cpu"
        ), patch.object(wan_generator, "hub_cache", return_value="/var/lib/orazaka-scratch/hf/hub"):
            wan_generator._pipe = None
            wan_generator._load(video_models.find(NAME), False)
        wan_generator._pipe = None
        wan_generator._pipe_name = None
        kwargs = fake.LTXConditionPipeline.from_pretrained.call_args.kwargs
        self.assertEqual("/var/lib/orazaka-scratch/hf/hub", kwargs["cache_dir"])
        self.assertTrue(kwargs["local_files_only"])


class TestKitProfile(unittest.TestCase):
    def test_the_kit_profile_serves_the_bake_off_models_and_flux_stills(self):
        from app import fetch_models

        manifest = video_models.load_manifest()
        kit = manifest["profiles"]["kit"]
        self.assertEqual(NAME, kit["defaultVideo"])
        self.assertEqual({NAME, "wan2.2-i2v-a14b-hq", "wan2.2-i2v-a14b"}, set(kit["video"]))
        self.assertEqual("flux-2-klein-4b", kit["image"])
        work = fetch_models.plan("kit")
        by_repo = {repo: (patterns, cache) for repo, _rev, patterns, cache in work["hf"]}
        patterns, cache = by_repo["Lightricks/LTX-Video-0.9.8-13B-distilled"]
        self.assertEqual("scratch", cache)  # the volume is full
        self.assertIn("text_encoder/*", patterns)
        self.assertNotIn("transformer_2/*", patterns)  # one transformer, not Wan's two experts
        self.assertIn("transformer_2/*", by_repo["Wan-AI/Wan2.2-I2V-A14B-Diffusers"][0])


class TestGenerate(unittest.TestCase):
    def _run(self, image):
        frames = [Image.new("RGB", (1216, 704)) for _ in range(121)]
        pipe = MagicMock()
        pipe.return_value = types.SimpleNamespace(frames=[frames])
        fake_ltx = types.SimpleNamespace(LTXVideoCondition=MagicMock(side_effect=lambda **kw: ("condition", kw)))
        fake_utils = types.SimpleNamespace(export_to_video=MagicMock(), load_video=MagicMock(return_value=["frame"]))
        modules = {
            "diffusers.pipelines.ltx.pipeline_ltx_condition": fake_ltx,
            "diffusers.utils": fake_utils,
        }
        torch = MagicMock()
        torch.Generator.return_value.manual_seed.return_value = "generator"
        with patch.dict(sys.modules, modules), patch.object(wan_generator, "_torch", return_value=torch), patch.object(
            wan_generator, "_pipe", pipe
        ), patch.object(wan_generator, "_pipe_name", NAME):
            out = wan_generator.generate_wan_frames(NAME, "slow dolly-in", 1216, 704, 121, 8, 7, image=image)
        return out, pipe, fake_ltx

    def test_a_clip_from_a_still_conditions_the_first_frame_and_passes_the_distilled_settings(self):
        out, pipe, fake_ltx = self._run(Image.new("RGB", (1360, 768)))
        self.assertEqual(121, len(out))
        kwargs = pipe.call_args.kwargs
        self.assertEqual((1216, 704, 121, 8, 1.0), (kwargs["width"], kwargs["height"], kwargs["num_frames"], kwargs["num_inference_steps"], kwargs["guidance_scale"]))
        self.assertEqual("slow dolly-in", kwargs["prompt"])
        self.assertIn("morphing", kwargs["negative_prompt"])  # the kit's default negatives
        self.assertEqual("pil", kwargs["output_type"])
        self.assertEqual(1, len(kwargs["conditions"]))
        self.assertEqual(0, fake_ltx.LTXVideoCondition.call_args.kwargs["frame_index"])

    def test_a_clip_from_text_has_no_condition(self):
        _out, pipe, _fake = self._run(None)
        self.assertIsNone(pipe.call_args.kwargs["conditions"])


if __name__ == "__main__":
    unittest.main()
