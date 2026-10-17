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
