"""Wan 2.2 A14B starts from an image: the same catalogue name, the image-to-video checkpoint of the same family (no GPU)."""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from app import video_models, wan_generator

TEXT = "wan2.2-t2v-a14b"
FIVE_B = "wan2.2-ti2v-5b"


class TestCatalogue(unittest.TestCase):
    def test_a14b_takes_a_starting_image_from_its_own_i2v_checkpoint_pinned_by_commit(self):
        model = video_models.find(TEXT)
        self.assertTrue(model.image_to_video)
        self.assertEqual("Wan-AI/Wan2.2-I2V-A14B-Diffusers", model.image_repo)
        self.assertRegex(model.image_revision, r"^[0-9a-f]{40}$")
        self.assertNotEqual(model.revision, model.image_revision)
        self.assertEqual((1280, 720), model.size_for("16:9"))

    def test_only_the_5b_conditions_the_first_latent_frame(self):
        self.assertTrue(video_models.find(FIVE_B).expand_timesteps)
        self.assertFalse(video_models.find(TEXT).expand_timesteps)  # the I2V A14B checkpoint's own config, never forced

    def test_the_text_model_without_an_image_repo_has_none(self):
        self.assertIsNone(video_models.find("wan2.1-t2v-1.3b").image_repo)


class TestLoad(unittest.TestCase):
    """_load picks the repository, the revision and the timestep mode from the model, never from the caller."""

    def _load(self, model_name, from_image):
        fake = types.SimpleNamespace(
            AutoencoderKLWan=MagicMock(), WanImageToVideoPipeline=MagicMock(), WanPipeline=MagicMock()
        )
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"diffusers": fake}), patch.object(wan_generator, "_torch", return_value=torch), patch.object(
            wan_generator, "_device", return_value="cpu"
        ):
            wan_generator._pipe = None
            wan_generator._load(video_models.find(model_name), from_image)
        wan_generator._pipe = None
        wan_generator._pipe_name = None
        return fake

    def test_an_image_for_a14b_loads_the_i2v_checkpoint_without_expand_timesteps(self):
        fake = self._load(TEXT, True)
        model = video_models.find(TEXT)
        fake.AutoencoderKLWan.from_pretrained.assert_called_once()
        self.assertEqual(model.image_repo, fake.AutoencoderKLWan.from_pretrained.call_args.args[0])
        args, kwargs = fake.WanImageToVideoPipeline.from_pretrained.call_args
        self.assertEqual(model.image_repo, args[0])
        self.assertEqual(model.image_revision, kwargs["revision"])
        self.assertNotIn("expand_timesteps", kwargs)

    def test_a14b_without_an_image_still_loads_the_text_checkpoint(self):
        fake = self._load(TEXT, False)
        model = video_models.find(TEXT)
        self.assertEqual(model.repo, fake.WanPipeline.from_pretrained.call_args.args[0])
        self.assertEqual(model.revision, fake.WanPipeline.from_pretrained.call_args.kwargs["revision"])

    def test_the_5b_keeps_its_one_checkpoint_and_expand_timesteps(self):
        fake = self._load(FIVE_B, True)
        args, kwargs = fake.WanImageToVideoPipeline.from_pretrained.call_args
        self.assertEqual(video_models.find(FIVE_B).repo, args[0])
        self.assertTrue(kwargs["expand_timesteps"])


if __name__ == "__main__":
    unittest.main()
