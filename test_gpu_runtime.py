"""The worker on a CUDA host: 9:16 shapes, model routing, the broker login, the asset hand-over (ADR-077).

Runs without a GPU and without torch: every pipeline is replaced by a recorder, so what is asserted is
what the worker *asks* of a model — its size, frame count, steps, seed — and where the result goes.
"""

from __future__ import annotations

import functools
import json
import os
import re
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

from app import activity, asset_sink, broker, video_models

ROOT = Path(__file__).resolve().parent


def _workspace_root() -> Path | None:
    for parent in ROOT.parents:
        if (parent / "orazaka.workspace.json").is_file():
            return parent
    return None


class TestAspect(unittest.TestCase):
    """A Studio says "9:16"; each model answers with a size it was trained at."""

    def test_the_vertical_short_is_the_default(self):
        self.assertEqual("9:16", video_models.normalise_aspect(None))
        self.assertEqual("9:16", video_models.normalise_aspect(""))

    def test_people_may_say_it_in_words(self):
        self.assertEqual("9:16", video_models.normalise_aspect("Vertical"))
        self.assertEqual("16:9", video_models.normalise_aspect("landscape"))
        self.assertEqual("1:1", video_models.normalise_aspect("1x1"))

    def test_an_unknown_aspect_is_refused_not_guessed(self):
        with self.assertRaises(video_models.UnsupportedAspect):
            video_models.normalise_aspect("4:3")

    def test_every_model_makes_a_vertical_short_taller_than_wide(self):
        for model in video_models.video_models().values():
            if model.pipeline == "remote":
                continue  # a provider's shapes are measured, not chosen (test_remote_video.py)
            width, height = model.size_for("9:16")
            self.assertGreater(height, width, model.name)
            self.assertEqual(0, width % model.multiple, model.name)
            self.assertEqual(0, height % model.multiple, model.name)
            ratio = height / width
            self.assertAlmostEqual(16 / 9, ratio, delta=0.06, msg=model.name)  # LTX is trained at 704x1216 (1.73)

    def test_wan_sizes_are_the_ones_it_was_trained_at(self):
        self.assertEqual((704, 1280), video_models.find("wan2.2-ti2v-5b").size_for("9:16"))
        self.assertEqual((1280, 704), video_models.find("wan2.2-ti2v-5b").size_for("16:9"))
        self.assertEqual((480, 832), video_models.find("wan2.1-t2v-1.3b").size_for("9:16"))

    def test_an_explicit_size_wins_rounded_to_the_model_grid(self):
        self.assertEqual((704, 1248), video_models.find("wan2.2-ti2v-5b").size_for("16:9", 710, 1250))

    def test_wan_counts_frames_on_its_4k_plus_1_grid(self):
        five_b = video_models.find("wan2.2-ti2v-5b")
        self.assertEqual(121, five_b.frames_for(5))
        self.assertEqual(121, five_b.frames_for(20))  # capped at what the model holds
        self.assertEqual(25, five_b.frames_for(1))
        for seconds in (1, 2, 3, 4, 5, 7):
            self.assertEqual(1, five_b.frames_for(seconds) % 4, seconds)
        self.assertEqual(81, video_models.find("wan2.1-t2v-1.3b").frames_for(5))


class TestModelRouting(unittest.TestCase):
    """Which model runs is the catalogue's choice, then the host's declared default — never a guess."""

    def test_the_host_default_is_declared_by_the_deployment(self):
        self.assertEqual(
            "wan2.1-t2v-1.3b",
            video_models.default_model_name(
                {"MEDIA_VIDEO_DEFAULT_MODEL": "wan2.1-t2v-1.3b", "MEDIA_GPU_PROFILE": "big"}
            ),
        )
        self.assertEqual("wan2.2-ti2v-5b", video_models.default_model_name({"MEDIA_GPU_PROFILE": "big"}))
        self.assertEqual("wan2.1-t2v-1.3b", video_models.default_model_name({"MEDIA_GPU_PROFILE": "small"}))
        self.assertEqual("animatediff-lightning-mps", video_models.default_model_name({}))

    def test_an_unknown_model_is_refused_by_name(self):
        with self.assertRaises(video_models.UnknownVideoModel):
            video_models.find("sora-2")

    def test_every_profile_names_models_the_manifest_describes(self):
        manifest = video_models.load_manifest()
        models = video_models.video_models(manifest)
        for name, profile in manifest["profiles"].items():
            self.assertIn(profile["defaultVideo"], profile["video"], name)
            for model in profile["video"]:
                self.assertIn(model, models, f"profile {name} names {model}")
                donor = models[model].text_encoder_from
                if donor:
                    self.assertIn(donor, models)
            if "image" in profile:  # `hero` runs video only on an 80 GB card
                self.assertIn(profile["image"], manifest["image"], name)

    def test_every_wan_model_is_pinned_and_licensed_for_commercial_use(self):
        manifest = video_models.load_manifest()
        for name, spec in manifest["video"].items():
            # OpenRAIL-M permits commercial use (with use restrictions): AnimateDiff Lightning's card says so.
            # The LTX-Video Open Weights License 0.X grants commercial use with no revenue clause (read in the model's licence file).
            self.assertIn(
                spec["licence"], {"Apache-2.0", "MIT", "CreativeML OpenRAIL-M", "LTX-Video-Open-Weights-0.X"}, name
            )
            if spec["pipeline"] in ("wan", "ltx"):
                self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$", name)

    def test_the_http_pipeline_registers_every_wan_model(self):
        main = _main()
        for name, spec in video_models.load_manifest()["video"].items():
            if spec["pipeline"] == "wan":
                self.assertEqual("Wan", main.MODEL_REGISTRY.get(name), name)

    def test_the_image_recipe_is_the_cli_recipe(self):
        """models.yaml copies the CLI's FLUX.2 [klein] files; a drift would serve another checkpoint."""
        workspace = _workspace_root()
        if workspace is None:
            self.skipTest("the CLI's recipe lives in the Orazaka workspace")
        recipes = workspace / "orazaka-apps" / "ui" / "orazaka-cli" / "src" / "services" / "image-model-recipes.json"
        cli = {model["id"]: model for model in json.loads(recipes.read_text())["models"]}
        for name, spec in video_models.load_manifest()["image"].items():
            if spec.get("workerOnly"):  # a model only this worker serves (flux-1-schnell): its licence is checked below
                continue
            self.assertIn(name, cli)
            ours = {(f["path"], f["sha256"], f["bytes"]) for f in spec["files"]}
            theirs = {(f["path"], f["sha256"], f["bytes"]) for f in cli[name]["files"]}
            self.assertEqual(theirs, ours, name)


class TestTheImageModelsAreLicensedForTheSites(unittest.TestCase):
    """orazaka#508: the renders land on public sites, so a profile's image model must allow commercial use."""

    def test_a_profile_never_serves_a_non_commercial_image_model(self):
        manifest = video_models.load_manifest()
        for name, profile in manifest["profiles"].items():
            if "image" in profile:
                spec = manifest["image"][profile["image"]]
                self.assertIn(spec["licence"], {"Apache-2.0", "MIT"}, f"{name}: {profile['image']}")
                self.assertNotEqual(False, spec.get("commercial", True), name)

    def test_flux_dev_is_not_listed(self):
        """FLUX.1 [dev] is under the FLUX.1 [dev] Non-Commercial License: not for our public sites."""
        self.assertFalse([n for n in video_models.load_manifest()["image"] if "dev" in n])

    def test_every_image_file_is_pinned_by_size_and_checksum(self):
        for name, spec in video_models.load_manifest()["image"].items():
            for f in spec["files"]:
                self.assertRegex(f["sha256"], r"^[0-9a-f]{64}$", name)
                self.assertGreater(f["bytes"], 0, name)
                self.assertTrue(f["url"].startswith("https://huggingface.co/"), name)


class TestTheStudioOffersWhatTheCatalogueAndThisWorkerKnow(unittest.TestCase):
    """A model a Studio's form offers is a catalogue row AND a model this worker runs — or the
    choice is a promise nobody keeps. The form lists names (a pack is data); this checks them."""

    def test_every_offered_video_model_exists_on_both_sides(self):
        workspace = _workspace_root()
        if workspace is None:
            self.skipTest("the blueprint and the catalogue live in the Orazaka workspace")
        blueprint = json.loads(
            (
                workspace / "orazaka-packs" / "orazaka-media" / "studios" / "video-generation" / "blueprint.json"
            ).read_text()
        )
        offered = blueprint["inputSchema"]["properties"].get("model", {}).get("enum")
        if not offered:
            self.skipTest("this blueprint offers no model list yet")
        seed = (
            workspace / "orazaka-apps" / "services" / "orazaka-job-service" / "infra" / "initdb" / "30-jobs-config.sql"
        ).read_text()
        registry = _main().MODEL_REGISTRY
        for name in offered:
            self.assertRegex(
                seed, rf"\('{re.escape(name)}', '[^']+', 'video'", f"{name} is not a video row of the catalogue"
            )
            self.assertIn(name, registry, f"{name} is not a model this worker runs")
        aspects = blueprint["inputSchema"]["properties"].get("aspect", {}).get("enum", [])
        for aspect in aspects:
            self.assertEqual(aspect, video_models.normalise_aspect(aspect))


class TestSdServerLine(unittest.TestCase):
    """The droplet's sd-server is started like the Mac's: the recipe's files and defaults."""

    def test_the_flux_klein_line(self):
        from app import sd_server

        line = sd_server.args("flux-2-klein-4b", Path("/models"), 8086, {})
        self.assertEqual(["--listen-ip", "0.0.0.0", "--listen-port", "8086"], line[:4])  # noqa: S104
        self.assertIn("--diffusion-model", line)
        self.assertEqual("/models/sd/flux2-klein-4b/Qwen3-4B-Q8_0.gguf", line[line.index("--llm") + 1])
        self.assertEqual("4", line[line.index("--steps") + 1])
        self.assertIn("--diffusion-fa", line)
        self.assertNotIn("--offload-to-cpu", line)
        self.assertIn(
            "--offload-to-cpu", sd_server.args("flux-2-klein-4b", Path("/m"), 8086, {"MEDIA_IMAGE_OFFLOAD": "true"})
        )

    def test_the_flux_schnell_line(self):
        from app import sd_server

        line = sd_server.args("flux-1-schnell", Path("/models"), 8086, {})
        for flag in ("--diffusion-model", "--clip_l", "--t5xxl", "--vae"):
            self.assertIn(flag, line)
        self.assertEqual("/models/sd/flux1-schnell/t5xxl_fp16.safetensors", line[line.index("--t5xxl") + 1])
        self.assertEqual("4", line[line.index("--steps") + 1])
        self.assertEqual("1.0", line[line.index("--cfg-scale") + 1])
        self.assertEqual("euler", line[line.index("--sampling-method") + 1])

    def test_the_render_profile_fetches_the_schnell_files(self):
        from app import fetch_models

        paths = {f["path"] for f in fetch_models.plan("render")["files"]}
        self.assertIn("flux1-schnell/flux1-schnell-Q8_0.gguf", paths)

    def test_the_fetch_plan_borrows_the_text_encoder_instead_of_downloading_it_twice(self):
        from app import fetch_models

        small = fetch_models.plan("small")
        repos = {repo: patterns for repo, _rev, patterns, _cache in small["hf"]}
        # The small profile serves the 5B too (image to video on 20 GB, with offload): its whole repo is fetched,
        # and the 1.3B still borrows the 5B's text encoder rather than downloading its own float32 copy.
        self.assertIn("text_encoder/*", repos["Wan-AI/Wan2.2-TI2V-5B-Diffusers"])
        self.assertIn("transformer/*", repos["Wan-AI/Wan2.2-TI2V-5B-Diffusers"])
        self.assertNotIn("text_encoder/*", repos["Wan-AI/Wan2.1-T2V-1.3B-Diffusers"])
        self.assertEqual(3, len(small["files"]))
        big = {repo: patterns for repo, _rev, patterns, _cache in fetch_models.plan("big")["hf"]}
        self.assertIn("transformer/*", big["Wan-AI/Wan2.2-TI2V-5B-Diffusers"])

    def test_a_file_that_does_not_match_its_checksum_is_never_used(self):
        from app import fetch_models

        root = Path(tempfile.mkdtemp(prefix="orz-fetch-"))
        source = root / "src.bin"
        source.write_bytes(b"not the model")
        spec = {"path": "m/model.gguf", "url": source.as_uri(), "bytes": 13, "sha256": "0" * 64}
        with self.assertRaises(RuntimeError):
            fetch_models.fetch_file(spec, root / "sd")
        self.assertFalse((root / "sd" / "m" / "model.gguf").exists())
        import hashlib

        spec["sha256"] = hashlib.sha256(b"not the model").hexdigest()
        target = fetch_models.fetch_file(spec, root / "sd")
        self.assertEqual(b"not the model", target.read_bytes())
        with patch.object(fetch_models.urllib.request, "urlretrieve") as again:
            fetch_models.fetch_file(spec, root / "sd")
        again.assert_not_called()


class TestOffload(unittest.TestCase):
    def test_auto_offloads_when_the_card_cannot_hold_the_model_beside_the_image_engine(self):
        from app.wan_generator import should_offload

        five_b = video_models.find("wan2.2-ti2v-5b")
        self.assertTrue(should_offload(five_b, 20, {}))
        self.assertFalse(should_offload(five_b, 80, {}))
        self.assertTrue(should_offload(five_b, 80, {"MEDIA_VIDEO_OFFLOAD": "true"}))
        self.assertFalse(should_offload(five_b, 20, {"MEDIA_VIDEO_OFFLOAD": "false"}))


def _main():
    if "app.main" in sys.modules:
        return sys.modules["app.main"]
    import app.main as main  # noqa: PLC0415

    return main


class TestWanThroughTheHttpPipeline(unittest.TestCase):
    """What the pipeline asks of Wan for a Studio's request."""

    def setUp(self):
        self.main = _main()
        self.server = HTTPServer(("127.0.0.1", 0), self.main.VideoInferenceHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.calls = []
        frame = MagicMock()
        frame.size = (704, 1280)

        def fake_wan(**kwargs):
            self.calls.append(kwargs)
            return [frame] * kwargs["num_frames"]

        self.patches = [
            patch.object(self.main, "generate_wan_frames", side_effect=fake_wan),
            patch.object(self.main, "export_frames_to_video"),
            patch.object(self.main, "preflight_memory_guard"),
            patch.object(self.main, "send_progress_update"),
        ]
        for active in self.patches:
            active.start()

    def tearDown(self):
        for active in self.patches:
            active.stop()
        self.server.shutdown()
        self.server.server_close()

    def _post(self, body):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.server.server_port}/v1/videos/generations",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 — loopback test server
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def test_a_9_16_short_is_704_by_1280_at_24_fps_for_five_seconds(self):
        out = os.path.join(tempfile.gettempdir(), "orz-test", "video.mp4")
        status, body = self._post(
            {
                "prompt": "a barista pours latte art, close-up",
                "model": "wan2.2-ti2v-5b",
                "aspect": "9:16",
                "durationSeconds": 5,
                "seed": 42,
                "output_path": out,
            }
        )
        self.assertEqual(200, status, body)
        call = self.calls[0]
        self.assertEqual((704, 1280), (call["width"], call["height"]))
        self.assertEqual(121, call["num_frames"])
        self.assertEqual(40, call["steps"])
        self.assertEqual(42, call["seed"])
        self.assertEqual("wan2.2-ti2v-5b", body["metrics"]["model"])
        self.assertEqual("42", body["metrics"]["seed"])
        self.assertEqual("9:16", body["metrics"]["aspect"])
        self.assertEqual(24, body["metrics"]["fps"])
        self.assertAlmostEqual(121 / 24, body["metrics"]["durationSeconds"], places=1)

    def test_without_a_model_the_host_profile_decides(self):
        out = os.path.join(tempfile.gettempdir(), "orz-test", "video.mp4")
        with patch.dict(os.environ, {"MEDIA_GPU_PROFILE": "small"}):
            status, body = self._post({"prompt": "a fox in snow", "output_path": out})
        self.assertEqual(200, status, body)
        self.assertEqual((480, 832), (self.calls[0]["width"], self.calls[0]["height"]))
        self.assertEqual("wan2.1-t2v-1.3b", body["metrics"]["model"])

    def test_the_mac_default_runs_animatediff_not_svd(self):
        """orazaka-worker-media#3: `animatediff-lightning-mps` took SVD's image-to-video branch."""
        out = os.path.join(tempfile.gettempdir(), "orz-test", "video.mp4")
        frame = MagicMock()
        frame.size = (432, 768)
        with (
            patch.object(self.main, "generate_video_frames_mlx", return_value=[frame] * 16) as animatediff,
            patch.object(self.main, "generate_video_frames") as svd,
        ):
            status, body = self._post(
                {"prompt": "x", "model": "animatediff-lightning-mps", "aspect": "9:16", "output_path": out}
            )
        self.assertEqual(200, status, body)
        svd.assert_not_called()
        self.assertEqual((432, 768), (animatediff.call_args.kwargs["width"], animatediff.call_args.kwargs["height"]))

    def test_an_aspect_the_model_cannot_make_is_a_400(self):
        status, _ = self._post({"prompt": "x", "model": "wan2.2-ti2v-5b", "aspect": "4:3"})
        self.assertEqual(400, status)
        self.assertEqual([], self.calls)

    def test_a_starting_image_for_a_text_to_video_model_is_a_400(self):
        status, body = self._post({"prompt": "x", "model": "wan2.1-t2v-1.3b", "image": "asset-123"})
        self.assertEqual(400, status)
        self.assertIn("text only", body)
        self.assertEqual([], self.calls)

    def test_an_unreadable_starting_image_for_the_5b_is_a_400_not_a_grey_clip(self):
        status, body = self._post({"prompt": "x", "model": "wan2.2-ti2v-5b", "image": "asset-123"})
        self.assertEqual(400, status)
        self.assertIn("cannot be read", body)
        self.assertEqual([], self.calls)


class TestBrokerLogin(unittest.TestCase):
    def test_credentials_come_from_the_deployment(self):
        params = broker.connection_parameters(
            {
                "SPRING_RABBITMQ_HOST": "127.0.0.1",
                "SPRING_RABBITMQ_PORT": "5673",
                "SPRING_RABBITMQ_USERNAME": "orazaka",
                "SPRING_RABBITMQ_PASSWORD": "s3cret-value",
            }
        )
        self.assertEqual(("127.0.0.1", 5673), (params.host, params.port))
        self.assertEqual("orazaka", params.credentials.username)
        self.assertIsNone(params.ssl_options)
        self.assertNotIn("s3cret", broker.describe(params))

    def test_tls_verifies_the_broker(self):
        import ssl

        params = broker.connection_parameters(
            {"SPRING_RABBITMQ_HOST": "mq.example.org", "SPRING_RABBITMQ_SSL_ENABLED": "true"}
        )
        context = params.ssl_options.context
        self.assertEqual(ssl.CERT_REQUIRED, context.verify_mode)
        self.assertTrue(context.check_hostname)
        self.assertTrue(broker.describe(params).startswith("amqps://"))

    def test_a_private_ca_and_the_certificate_name_are_declared(self):
        """The GPU host dials a tunnel end (127.0.0.1) and checks the broker's own name, against the dev CA."""
        import ssl
        import subprocess

        directory = Path(tempfile.mkdtemp(prefix="orz-ca-"))
        ca = directory / "ca.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "ec",
                "-pkeyopt",
                "ec_paramgen_curve:P-256",
                "-nodes",
                "-days",
                "1",
                "-subj",
                "/CN=test CA",
                "-keyout",
                str(directory / "ca.key"),
                "-out",
                str(ca),
            ],
            check=True,
            capture_output=True,
        )
        params = broker.connection_parameters(
            {
                "SPRING_RABBITMQ_HOST": "127.0.0.1",
                "SPRING_RABBITMQ_PORT": "5671",
                "SPRING_RABBITMQ_SSL_ENABLED": "true",
                "SPRING_RABBITMQ_SSL_CA_FILE": str(ca),
                "SPRING_RABBITMQ_SSL_SERVER_NAME": "broker-dev.orazaka.com",
            }
        )
        self.assertEqual("broker-dev.orazaka.com", params.ssl_options.server_hostname)
        self.assertEqual(ssl.CERT_REQUIRED, params.ssl_options.context.verify_mode)
        subjects = [dict(x[0] for x in cert["subject"]) for cert in params.ssl_options.context.get_ca_certs()]
        self.assertIn({"commonName": "test CA"}, subjects)

    def test_the_local_default_is_unchanged(self):
        params = broker.connection_parameters({})
        self.assertEqual(("localhost", 5672), (params.host, params.port))
        self.assertEqual("guest", params.credentials.username)


class TestAssetHandOver(unittest.TestCase):
    """A droplet keeps nothing: the output goes to the job service, sealed there."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="orz-sink-")
        self.path = os.path.join(self.dir, "video.mp4")
        Path(self.path).write_bytes(b"\x00\x00\x00\x18ftypmp42")
        self.job = {
            "jobId": "job-9",
            "userId": "user-1",
            "featureKey": "orazaka.core.media.video",
            "dataClass": "STANDARD",
        }

    def test_upload_puts_the_bytes_with_the_run_identity(self):
        sent = {}

        class Response:
            status = 201

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"url": "/api/v1/assets/job-9/video.mp4"}'

        def fake_urlopen(request, timeout):
            sent["url"] = request.full_url
            sent["method"] = request.get_method()
            sent["headers"] = dict(request.header_items())
            sent["data"] = request.data
            return Response()

        env = {"MEDIA_ASSET_SINK_URL": "http://127.0.0.1:8090/"}
        with (
            patch.object(asset_sink, "_service_token", return_value="tok"),
            patch.object(asset_sink.urllib.request, "urlopen", fake_urlopen),
        ):
            url = asset_sink.upload(self.path, self.job, "video.mp4", "video/mp4", env)
        self.assertEqual("/api/v1/assets/job-9/video.mp4", url)
        self.assertEqual("http://127.0.0.1:8090/internal/v1/jobs/job-9/output/video.mp4", sent["url"])
        self.assertEqual("PUT", sent["method"])
        self.assertEqual("Bearer tok", sent["headers"]["Authorization"])
        self.assertEqual("user-1", sent["headers"]["X-orazaka-user-id"])
        self.assertEqual(Path(self.path).read_bytes(), sent["data"])

    def test_a_path_is_never_an_asset_name(self):
        with self.assertRaises(asset_sink.AssetSinkError):
            asset_sink.upload(self.path, self.job, "../video.mp4", "video/mp4", {"MEDIA_ASSET_SINK_URL": "http://x"})

    def test_no_token_no_upload(self):
        with (
            patch.object(asset_sink, "_service_token", return_value=None),
            self.assertRaises(asset_sink.AssetSinkError),
        ):
            asset_sink.upload(self.path, self.job, "video.mp4", "video/mp4", {"MEDIA_ASSET_SINK_URL": "http://x"})

    def test_the_consumer_hands_over_and_deletes_when_a_sink_is_declared(self):
        import app.consumer as consumer

        with (
            patch.dict(os.environ, {"MEDIA_ASSET_SINK_URL": "http://127.0.0.1:8090"}),
            patch.object(consumer.asset_sink, "upload", return_value="/api/v1/assets/job-9/video.mp4") as upload,
            patch.object(consumer, "_seal") as seal,
        ):
            url = consumer._store(self.path, self.job, "video.mp4", "video/mp4", "/uploads/x")
        self.assertEqual("/api/v1/assets/job-9/video.mp4", url)
        upload.assert_called_once()
        seal.assert_not_called()
        self.assertFalse(os.path.exists(self.path), "the droplet must not keep the clip")

    def test_the_consumer_seals_in_place_on_a_shared_disk(self):
        import app.consumer as consumer

        env = {k: v for k, v in os.environ.items() if k != "MEDIA_ASSET_SINK_URL"}
        with patch.dict(os.environ, env, clear=True), patch.object(consumer, "_seal") as seal:
            url = consumer._store(self.path, self.job, "video.mp4", "video/mp4", "/uploads/x")
        self.assertEqual("/uploads/x", url)
        seal.assert_called_once_with(self.path)


class TestConsumerOnAGpuHost(unittest.TestCase):
    def test_an_unknown_aspect_is_the_callers_and_spends_no_gpu(self):
        import app.consumer as consumer

        job = {"jobId": "j", "userId": "u", "payload": {"prompt": "x", "aspect": "4:3"}}
        with patch.object(consumer.urllib.request, "urlopen") as urlopen, self.assertRaises(consumer.InvalidJobPayload):
            consumer._generate(job)
        urlopen.assert_not_called()

    def test_the_job_runs_off_the_connection_thread_and_acks_through_it(self):
        """A five-minute render must not starve the heartbeats; the ack goes back via the I/O loop."""
        import app.consumer as consumer

        channel = MagicMock()
        method = MagicMock(delivery_tag=7, routing_key="job.video.generate")
        done = threading.Event()
        names = []

        def fake_generate(job):
            names.append(threading.current_thread().name)
            return {"url": "/api/v1/assets/j/video.mp4", "format": "mp4"}, {}

        channel.connection.add_callback_threadsafe.side_effect = lambda fn: (fn(), done.set())
        body = json.dumps({"jobId": "j", "userId": "u", "payload": {"prompt": "x"}}).encode()
        with patch.object(consumer, "_generate", side_effect=fake_generate), patch.object(consumer, "send_job_done"):
            consumer._dispatch(channel, method, None, body)
            self.assertTrue(done.wait(5))
        self.assertEqual(["media-job"], names)
        channel.basic_ack.assert_called_once_with(delivery_tag=7)
        callback = channel.connection.add_callback_threadsafe.call_args[0][0]
        self.assertIsInstance(callback, functools.partial)


class TestActivity(unittest.TestCase):
    def test_the_watchdog_signal_is_a_file_touched_by_each_job(self):
        target = Path(tempfile.mkdtemp(prefix="orz-act-")) / "state" / "last-activity"
        activity.touch({"MEDIA_ACTIVITY_FILE": str(target)})
        self.assertTrue(target.exists())

    def test_without_a_declared_file_nothing_is_written(self):
        self.assertIsNone(activity.activity_file({}))
        activity.touch({})


class TestHeartbeat(unittest.TestCase):
    """The watchdog counts a pod busy while MEDIA_HEARTBEAT_FILE is fresh (krizaka-internal#2101): a model load after
    `ready`, the GPU still idle, must never reach the idle limit."""

    def _target(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="orz-hb-")) / "state" / "heartbeat"

    def test_the_beat_starts_at_once_and_repeats_while_the_work_runs(self):
        target = self._target()
        env = {"MEDIA_HEARTBEAT_FILE": str(target)}
        with activity.beating(env, every=0.05):
            self.assertTrue(target.exists())
            os.utime(target, (0, 0))
            deadline = time.time() + 2
            while target.stat().st_mtime == 0 and time.time() < deadline:
                time.sleep(0.02)
            self.assertGreater(target.stat().st_mtime, 0, "the file is touched again while the work runs")

    def test_the_beat_stops_when_the_work_ends_even_on_failure(self):
        target = self._target()
        env = {"MEDIA_HEARTBEAT_FILE": str(target)}
        with self.assertRaises(RuntimeError):
            with activity.beating(env, every=0.05):
                raise RuntimeError("the job failed")
        os.utime(target, (0, 0))
        time.sleep(0.2)
        self.assertEqual(0, target.stat().st_mtime, "no beat after the work: an idle pod must look idle")
        self.assertFalse([t for t in threading.enumerate() if t.name == "media-heartbeat"])

    def test_without_a_declared_file_nothing_beats(self):
        self.assertIsNone(activity.heartbeat_file({}))
        with activity.beating({}, every=0.05):
            self.assertFalse([t for t in threading.enumerate() if t.name == "media-heartbeat"])

    def test_a_job_beats_from_its_first_moment_to_its_acknowledgement(self):
        from app import consumer  # noqa: PLC0415

        target = self._target()
        seen = []

        def fake_generate(job):
            seen.append(target.exists())
            return {"url": "u"}, {}

        channel, method = MagicMock(), MagicMock(routing_key="media.video", delivery_tag=1)
        body = json.dumps({"jobId": "hb-job", "userId": "u", "payload": {"prompt": "x"}}).encode()
        with (
            patch.dict(os.environ, {"MEDIA_HEARTBEAT_FILE": str(target)}),
            patch.object(consumer, "_generate", side_effect=fake_generate),
            patch.object(consumer, "send_job_done"),
            patch.object(consumer, "send_progress_update"),
        ):
            consumer._on_message(channel, method, None, body)
        self.assertEqual([True], seen, "the heartbeat is fresh while the job runs")
        self.assertFalse([t for t in threading.enumerate() if t.name == "media-heartbeat"])

    def test_a_model_load_beats(self):
        from app import wan_generator  # noqa: PLC0415

        target = self._target()
        seen = []
        model = MagicMock(pipeline="ltx")
        with (
            patch.dict(os.environ, {"MEDIA_HEARTBEAT_FILE": str(target)}),
            patch.object(wan_generator, "_load_ltx", side_effect=lambda m: seen.append(target.exists()) or "pipe"),
        ):
            self.assertEqual("pipe", wan_generator._load(model))
        self.assertEqual([True], seen, "the heartbeat is fresh while the weights load")


if __name__ == "__main__":
    unittest.main()
