"""Image to video with Wan 2.2 TI2V 5B: the photo's crop, the size ladder, the refusals, the hand-over (no GPU)."""
from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from app import video_models, wan_generator

MODEL = "wan2.2-ti2v-5b"


def _main():
    """Imported late: test_main stubs pika before app.main (and so app.telemetry) loads."""
    import app.main as main

    return main


def photo(width=1200, height=800, mark=None):
    image = Image.new("RGB", (width, height), (10, 20, 30))
    if mark:
        x, y = mark
        for dx in range(-20, 20):
            for dy in range(-20, 20):
                image.putpixel((x + dx, y + dy), (250, 0, 0))
    return image


class TestCrop(unittest.TestCase):
    def test_a_wide_frame_cuts_top_and_bottom_never_letterboxes(self):
        out = video_models.crop_to_aspect(photo(1000, 1000), 1280, 704)
        self.assertEqual((1280, 704), out.size)
        self.assertNotIn((0, 0, 0), {out.getpixel((0, 0)), out.getpixel((1279, 703))})  # no black bars

    def test_a_vertical_frame_cuts_the_sides_around_the_subject(self):
        source = photo(1600, 900, mark=(1300, 450))  # the subject sits at the right
        centred = video_models.crop_to_aspect(source, 704, 1280)
        subject = video_models.crop_to_aspect(source, 704, 1280, focus=(1300 / 1600, 0.5))
        def reds(im):
            return sum(1 for x in range(im.width) for y in range(im.height) if im.getpixel((x, y))[0] > 200 and im.getpixel((x, y))[1] < 50)

        self.assertEqual(0, reds(centred))
        self.assertGreater(reds(subject), 0)

    def test_the_focus_is_clamped_to_the_photo(self):
        self.assertEqual((480, 832), video_models.crop_to_aspect(photo(), 480, 832, focus=(9, -3)).size)


class TestModelShape(unittest.TestCase):
    def test_only_the_5b_starts_from_a_photo(self):
        self.assertTrue(video_models.find(MODEL).image_to_video)
        self.assertFalse(video_models.find("wan2.1-t2v-1.3b").image_to_video)

    def test_the_ladder_goes_down_in_the_same_aspect(self):
        model = video_models.find(MODEL)
        self.assertEqual([(1280, 704), (960, 544), (832, 480)], model.size_ladder("16:9"))
        self.assertEqual([(704, 1280), (544, 960), (480, 832)], model.size_ladder("9:16"))
        self.assertEqual([(640, 352)], model.size_ladder("16:9", 640, 352))  # an explicit size has none


class TestRefusals(unittest.TestCase):
    def test_a_photo_for_a_text_only_model_is_refused_by_name(self):
        with self.assertRaises(video_models.InvalidVideoRequest):
            wan_generator.generate_wan_frames("wan2.1-t2v-1.3b", "x", 832, 480, 81, 30, 1, image=photo())

    def test_an_unreadable_photo_is_refused_never_replaced_by_grey(self):
        with self.assertRaises(video_models.InvalidVideoRequest):
            _main().starting_image({"image": base64.b64encode(b"not an image").decode()})

    def test_blank_inputs_are_no_image(self):
        self.assertFalse(_main()._asked_for_an_image({"image": "", "image_path": "  "}))
        self.assertTrue(_main()._asked_for_an_image({"image_path": "/tmp/a.jpg"}))  # noqa: S108

    def test_the_photo_is_read_from_base64_or_a_file(self):
        buffer = io.BytesIO()
        photo(64, 48).save(buffer, "PNG")
        self.assertEqual((64, 48), _main().starting_image({"image": base64.b64encode(buffer.getvalue()).decode()}).size)
        path = os.path.join(tempfile.mkdtemp(prefix="orz-i2v-"), "p.png")
        photo(32, 32).save(path)
        self.assertEqual((32, 32), _main().starting_image({"image_path": path}).size)

    def test_the_focus_defaults_to_the_centre(self):
        self.assertEqual((0.5, 0.5), _main()._focus({}))
        self.assertEqual((0.25, 1.0), _main()._focus({"focusX": "0.25", "focusY": 7}))


class FakePipe:
    """A pipeline that runs out of memory above a size, and records what it was given."""

    def __init__(self, fits):
        self.fits, self.calls = fits, []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["width"] > self.fits:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        frames = [Image.new("RGB", (kwargs["width"], kwargs["height"]))] * 5
        return MagicMock(frames=[frames])


class TestResidentGuard(unittest.TestCase):
    def test_a_resident_pipeline_has_paid_for_its_memory(self):
        with patch.object(wan_generator, "_pipe", object()), patch.object(wan_generator, "_pipe_name", f"{MODEL}:image"):
            self.assertTrue(wan_generator.is_resident(MODEL, True))
            self.assertFalse(wan_generator.is_resident(MODEL, False))  # the text pipeline is another resident
            self.assertFalse(wan_generator.is_resident("not-a-model", True))
        with patch.object(wan_generator, "_pipe", None):
            self.assertFalse(wan_generator.is_resident(MODEL, True))


class TestLadder(unittest.TestCase):
    def run_with(self, fits, fallback):
        pipe = FakePipe(fits)
        torch = MagicMock()
        torch.cuda.OutOfMemoryError = type("OOM", (RuntimeError,), {})
        with patch.object(wan_generator, "_torch", return_value=torch), patch.object(
            wan_generator, "_pipe", pipe
        ), patch.object(wan_generator, "_pipe_name", f"{MODEL}:image"):
            frames = wan_generator.generate_wan_frames(
                MODEL, "a quiet street", 1280, 704, 5, 4, 7, image=photo(), fallback_sizes=fallback
            )
        return frames, pipe

    def test_it_steps_down_until_the_card_holds_the_size_and_says_which(self):
        frames, pipe = self.run_with(900, ((960, 544), (832, 480)))
        self.assertEqual([(1280, 704), (960, 544), (832, 480)], [(c["width"], c["height"]) for c in pipe.calls])
        self.assertEqual((832, 480), frames[0].size)
        for call in pipe.calls:  # the photo is cut to each size, never letterboxed
            self.assertEqual((call["width"], call["height"]), call["image"].size)

    def test_it_keeps_the_first_size_when_it_fits(self):
        frames, pipe = self.run_with(2000, ((960, 544),))
        self.assertEqual(1, len(pipe.calls))
        self.assertEqual((1280, 704), frames[0].size)

    def test_the_last_size_failing_is_an_error_not_a_placeholder(self):
        with self.assertRaises(RuntimeError):
            self.run_with(100, ((960, 544), (832, 480)))


class TestHandOver(unittest.TestCase):
    def test_a_droplet_fetches_the_photo_from_the_job_service(self):
        import app.consumer as consumer  # imported late, as _main()
        from app import asset_sink

        scratch = tempfile.mkdtemp(prefix="orz-i2v-")
        job = {"jobId": "j-1", "userId": "u-1"}
        response = MagicMock()
        response.read.return_value = b"\xff\xd8jpeg"
        response.__enter__.return_value = response
        with patch.dict(os.environ, {"MEDIA_ASSET_SINK_URL": "http://core:8090"}), patch.object(
            asset_sink, "_service_token", return_value="tok"
        ), patch.object(asset_sink.urllib.request, "urlopen", return_value=response) as opened:
            got = consumer._starting_image(job, {"imagePath": "/srv/uploads/u-1/a.jpg"}, scratch)
        request = opened.call_args.args[0]
        self.assertEqual("http://core:8090/internal/v1/jobs/j-1/input/image", request.full_url)
        self.assertEqual("u-1", request.get_header("X-orazaka-user-id"))
        self.assertEqual(b"\xff\xd8jpeg", open(got, "rb").read())
        shutil.rmtree(scratch)

    def test_a_studio_step_names_its_photo_by_asset_id_and_the_droplet_asks_for_that_asset(self):
        import app.consumer as consumer
        from app import asset_sink

        scratch = tempfile.mkdtemp(prefix="orz-i2v-")
        asset = "11111111-1111-1111-1111-111111111111"
        response = MagicMock()
        response.read.return_value = b"photo"
        response.__enter__.return_value = response
        with patch.dict(os.environ, {"MEDIA_ASSET_SINK_URL": "http://core:8090"}), patch.object(
            asset_sink, "_service_token", return_value="tok"
        ), patch.object(asset_sink.urllib.request, "urlopen", return_value=response) as opened:
            consumer._starting_image({"jobId": "j-2", "userId": "u-1"}, {"image": asset}, scratch)
        self.assertEqual(
            f"http://core:8090/internal/v1/jobs/j-2/input/image?asset={asset}", opened.call_args.args[0].full_url
        )
        shutil.rmtree(scratch)

    def test_where_the_disk_is_shared_the_asset_id_is_resolved_for_its_owner_only(self):
        import app.consumer as consumer

        root = tempfile.mkdtemp(prefix="orz-up-")
        asset = "22222222-2222-2222-2222-222222222222"
        os.makedirs(os.path.join(root, "u-1", "temp"))
        mine = os.path.join(root, "u-1", "temp", asset + ".png")
        photo(16, 16).save(mine)
        scratch = tempfile.mkdtemp(prefix="orz-i2v-")
        env = {k: v for k, v in os.environ.items() if k != "MEDIA_ASSET_SINK_URL"}
        with patch.dict(os.environ, env, clear=True), patch.object(consumer, "_upload_root", return_value=root):
            self.assertEqual(
                os.path.realpath(mine),
                os.path.realpath(consumer._starting_image({"userId": "u-1"}, {"image": asset}, scratch)),
            )
            with self.assertRaises(consumer.InvalidJobPayload):
                consumer._starting_image({"userId": "u-2"}, {"image": asset}, scratch)
        shutil.rmtree(scratch)

    def test_a_blank_image_is_none_and_a_named_unreadable_one_is_refused(self):
        import app.consumer as consumer

        scratch = tempfile.mkdtemp(prefix="orz-i2v-")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEDIA_ASSET_SINK_URL", None)
            self.assertIsNone(consumer._starting_image({}, {"image": ""}, scratch))
            with self.assertRaises(consumer.InvalidJobPayload):
                consumer._starting_image({}, {"imagePath": "/nowhere/a.jpg"}, scratch)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    unittest.main()


class TestA14bLocal(unittest.TestCase):
    """Wan 2.2 A14B at 720p on an 80 GB card: two experts, its own guidance, its own profile."""

    def test_it_is_a_wan_model_at_720p_with_the_experts_guidance(self):
        model = video_models.find("wan2.2-t2v-a14b")
        self.assertEqual("wan", model.pipeline)
        self.assertEqual((1280, 720), model.size_for("16:9"))
        self.assertEqual((720, 1280), model.size_for("9:16"))
        self.assertEqual((16, 81), (model.fps, model.max_frames))
        self.assertEqual((4.0, 3.0), (model.guidance, model.guidance_two))
        self.assertEqual(81, model.frames_for(5))
        self.assertIsNone(video_models.find("wan2.2-ti2v-5b").guidance_two)
        self.assertFalse(model.image_to_video)  # text to video: a photo is refused by name

    def test_the_hero_profile_fetches_the_experts_and_no_image_engine(self):
        from app import fetch_models

        plan = fetch_models.plan("hero")
        repos = {repo: patterns for repo, _rev, patterns, _cache in plan["hf"]}
        self.assertIn("transformer/*", repos["Wan-AI/Wan2.2-T2V-A14B-Diffusers"])
        self.assertEqual([], plan["files"])

    def test_it_is_registered_and_gets_the_second_guidance_in_the_call(self):
        main = _main()
        self.assertEqual("Wan", main.MODEL_REGISTRY["wan2.2-t2v-a14b"])
        self.assertGreaterEqual(main.MODEL_MEMORY_GB["wan2.2-t2v-a14b"], 64)
        pipe = FakePipe(5000)
        torch = MagicMock()
        with patch.object(wan_generator, "_torch", return_value=torch), patch.object(wan_generator, "_pipe", pipe), patch.object(
            wan_generator, "_pipe_name", "wan2.2-t2v-a14b"
        ):
            wan_generator.generate_wan_frames("wan2.2-t2v-a14b", "x", 1280, 720, 5, 4, 7)
        self.assertEqual((4.0, 3.0), (pipe.calls[0]["guidance_scale"], pipe.calls[0]["guidance_scale_2"]))


class TestCinemaI2V(unittest.TestCase):
    """The shots of a cinematic short: Wan 2.2 I2V A14B in four distilled steps, from a keyframe (ADR-080)."""

    def test_it_starts_from_a_photo_only_in_four_steps_without_guidance(self):
        model = video_models.find("wan2.2-i2v-a14b")
        self.assertTrue(model.image_to_video and model.requires_image)
        self.assertFalse(model.expand_timesteps)  # the 5B's conditioning, not the A14B's
        self.assertTrue(video_models.find("wan2.2-ti2v-5b").expand_timesteps)
        self.assertEqual((4, 1.0, 1.0), (model.steps, model.guidance, model.guidance_two))
        self.assertEqual((720, 1280), model.size_for("9:16"))
        self.assertEqual((1280, 720), model.size_for("16:9"))

    def test_a_text_only_job_is_refused_before_a_gpu_second(self):
        with self.assertRaises(video_models.InvalidVideoRequest):
            wan_generator.generate_wan_frames("wan2.2-i2v-a14b", "x", 720, 1280, 5, 4, 7)

    def test_its_weights_are_read_from_the_scratch_disk_and_the_lora_pair_goes_to_both_experts(self):
        model = video_models.find("wan2.2-i2v-a14b")
        self.assertEqual("/s/hf/hub", video_models.hub_cache(model, {"MEDIA_SCRATCH_DIR": "/s"}))
        self.assertIsNone(video_models.hub_cache(video_models.find("wan2.2-ti2v-5b"), {}))
        plan = wan_generator.lora_plan(model)
        self.assertEqual([False, True], [second for _f, _a, second in plan])
        self.assertIn("high_noise", plan[0][0])

    def test_the_cinema_profile_fetches_the_experts_the_loras_and_the_image_engine(self):
        from app import fetch_models

        work = fetch_models.plan("cinema")
        by_repo = {repo: (patterns, cache) for repo, _rev, patterns, cache in work["hf"]}
        patterns, cache = by_repo["Wan-AI/Wan2.2-I2V-A14B-Diffusers"]
        self.assertEqual("scratch", cache)
        self.assertIn("transformer_2/*", patterns)
        self.assertNotIn("text_encoder/*", patterns)  # the 5B's bf16 encoder is borrowed
        self.assertEqual("volume", by_repo["Wan-AI/Wan2.2-TI2V-5B-Diffusers"][1])
        self.assertEqual(2, len(by_repo["lightx2v/Wan2.2-Distill-Loras"][0]))
        self.assertEqual(3, len(work["files"]))  # FLUX.2 klein for the keyframes

    def test_the_hero_fetch_includes_the_low_noise_expert(self):
        from app import fetch_models

        repos = {repo: patterns for repo, _rev, patterns, _cache in fetch_models.plan("hero")["hf"]}
        self.assertIn("transformer_2/*", repos["Wan-AI/Wan2.2-T2V-A14B-Diffusers"])

    def test_it_is_registered(self):
        main = _main()
        self.assertEqual("Wan", main.MODEL_REGISTRY["wan2.2-i2v-a14b"])
        self.assertGreaterEqual(main.MODEL_MEMORY_GB["wan2.2-i2v-a14b"], 64)
