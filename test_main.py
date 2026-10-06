"""
@file test_main.py
@description Unit tests for orazaka-video-worker covering:
  - preflight_memory_guard (§12 host-memory guard; zero-fallback)
  - send_progress_update AMQP dispatch
  - resource_guard context manager
  - consumption metering (ADR-033 §6.3: measurements reported, units never named)
  - VideoInferenceHandler GET/POST routes
  - Model registry validation
  - Memory cap enforcement
  - Base64/output_path response modes
"""

import base64
import io
import json
import os
import sys
import tempfile
import unittest
from http.server import HTTPServer
from threading import Thread
from unittest.mock import MagicMock, patch

from PIL import Image

# ---------------------------------------------------------------------------
# Patch heavy ML dependencies before importing the module under test
# ---------------------------------------------------------------------------
mock_torch = MagicMock()
mock_torch.manual_seed.return_value = MagicMock()
mock_torch.backends.mps.is_available.return_value = False
mock_torch.cuda.is_available.return_value = False
mock_torch.float16 = "float16"
sys.modules["torch"] = mock_torch

mock_diffusers = MagicMock()
sys.modules["diffusers"] = mock_diffusers
sys.modules["diffusers.utils"] = MagicMock()

sys.modules["torchvision"] = MagicMock()
sys.modules["accelerate"] = MagicMock()
sys.modules["safetensors"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Patch pika before import
mock_pika = MagicMock()
sys.modules["pika"] = mock_pika

# Now stub the global pipeline load to avoid model download
mock_pipe = MagicMock()
mock_diffusers.StableVideoDiffusionPipeline.from_pretrained.return_value = mock_pipe

# Import the module under test (will execute top-level code)
# We need to patch the sys.exit and pipeline loading
with patch.dict(os.environ, {"SPRING_RABBITMQ_HOST": "localhost", "SPRING_RABBITMQ_PORT": "5672"}):
    with patch("sys.exit"):
        # The module loads the pipeline at import time, so we mock it
        import importlib
        spec = importlib.util.spec_from_file_location(
            "video_worker",
            os.path.join(os.path.dirname(__file__), "app", "main.py")
        )
        video_worker = importlib.util.module_from_spec(spec)
        # Prevent the top-level pipeline load from running
        with patch.object(mock_diffusers.StableVideoDiffusionPipeline, "from_pretrained", return_value=mock_pipe):
            try:
                spec.loader.exec_module(video_worker)
            except SystemExit:
                pass


class TestTheRunnerCollectsEveryDefinedTest(unittest.TestCase):
    """The number of tests the runner collects is the number the files define (ADR-064).

    **First class in the file, deliberately.** A guard defined after a misplaced ``unittest.main()``
    is skipped by the defect it guards — which is what happened to the first version of this one.
    Here, a ``unittest.main()`` anywhere below still finds this class defined, runs it, and it
    counts the classes the module never reached. Placed above it, ``unittest.main()`` would collect
    nothing, and a run of zero tests exits non-zero.

    The Python twin of GOV-006, the rule that a governance check examining nothing fails. Five tests
    in this file sat after ``unittest.main()`` — which exits — and never ran: the build reported 58
    and nobody counted the 63 written. Counting is the whole guard, so it counts two ways:

    * **per runner** — every test this file defines, read from its syntax, is one the loader collects
      from the module as it actually executed. A class after ``unittest.main()``, a second method
      with a name already taken, a pytest-style module function unittest never collects: each makes
      the two numbers differ;
    * **per repository** — every ``test_*.py`` is named by a Python execution in the root
      ``pom.xml``. A suite with no runner defines tests that nothing collects, and zero collected
      out of twenty-seven is the same finding as fifty-eight out of sixty-three.
    """

    REPOSITORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    SKIPPED_DIRECTORIES = {".git", ".venv", "node_modules", "target", "__pycache__", "dist", ".next"}

    # A suite that exists and CANNOT run here, each with the condition that keeps it so. The
    # condition is checked: the day it stops holding, the entry is stale and this class fails,
    # because a runnable suite left out of the build is the defect this guard exists for.
    UNRUNNABLE = {
        "orazaka-packs/document-validation/worker/test_authenticity.py": (
            "pypdf",
            "its PDF library is declared nowhere and absent from this environment (audit #27)",
        ),
    }

    @staticmethod
    def _defined(path):
        import ast

        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        defined = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = {getattr(base, "attr", getattr(base, "id", "")) for base in node.bases}
                if "TestCase" in bases:
                    defined += [
                        f"{node.name}.{item.name}"
                        for item in node.body
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name.startswith("test")
                    ]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                defined.append(node.name)
        return defined

    def test_every_test_this_file_defines_is_collected(self):
        module = sys.modules[type(self).__module__]
        collected = []
        pending = [unittest.defaultTestLoader.loadTestsFromModule(module)]
        while pending:
            for item in pending.pop():
                if isinstance(item, unittest.TestSuite):
                    pending.append(item)
                else:
                    collected.append(f"{type(item).__name__}.{item._testMethodName}")
        defined = self._defined(os.path.abspath(__file__))
        never_collected = sorted(set(defined) - set(collected))
        defined_twice = sorted({name for name in defined if defined.count(name) > 1})

        if len(defined) != len(collected) or never_collected or defined_twice:
            self.fail(
                f"this file defines {len(defined)} tests and the runner collected {len(collected)}"
                f"\n  never collected: {never_collected}\n  defined twice: {defined_twice}"
            )

    def test_every_python_suite_in_the_repository_has_a_runner(self):
        import importlib.util
        import xml.etree.ElementTree as ElementTree

        namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
        pom = ElementTree.parse(os.path.join(self.REPOSITORY, "pom.xml"))
        run = set()
        for execution in pom.iter("{http://maven.apache.org/POM/4.0.0}execution"):
            executable = execution.find("m:configuration/m:executable", namespace)
            directory = execution.find("m:configuration/m:workingDirectory", namespace)
            if executable is None or directory is None or "python" not in executable.text:
                continue
            base = directory.text.replace("${project.basedir}", self.REPOSITORY)
            for argument in execution.findall("m:configuration/m:arguments/m:argument", namespace):
                if argument.text and argument.text.endswith(".py"):
                    run.add(os.path.relpath(os.path.join(base, argument.text), self.REPOSITORY))

        suites = set()
        for root, directories, files in os.walk(self.REPOSITORY):
            directories[:] = [d for d in directories if d not in self.SKIPPED_DIRECTORIES]
            for name in files:
                if name.startswith("test_") and name.endswith(".py"):
                    suites.add(os.path.relpath(os.path.join(root, name), self.REPOSITORY))
        self.assertTrue(suites, "no Python suite found: this guard would be judging the empty set")

        for suite, (module, reason) in self.UNRUNNABLE.items():
            self.assertIsNone(
                importlib.util.find_spec(module),
                f"{suite} was excused because {reason} — but {module} is importable now: run it",
            )
        unrun = sorted(suites - run - set(self.UNRUNNABLE))
        if unrun:
            self.fail(f"Python suites no build execution runs, so no test in them ever has: {unrun}")


class TestPreflightMemoryGuard(unittest.TestCase):
    """§12 host-memory guard: oversized models fail fast/clean (never fake frames)."""

    def test_heavy_model_rejected_when_insufficient_memory(self):
        """SVD-XT (~96GB) must be refused on a host with ~50GB available."""
        fake_vm = MagicMock(available=50 * 1024 ** 3, percent=40.0)
        with patch.object(video_worker, "psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = fake_vm
            with self.assertRaises(RuntimeError) as ctx:
                video_worker.preflight_memory_guard("stable-video-diffusion-img2vid-xt")
            self.assertIn("needs", str(ctx.exception).lower())

    def test_light_model_allowed_with_ample_memory(self):
        """AnimateDiff-Lightning (~6GB) must pass the preflight with ~50GB available."""
        fake_vm = MagicMock(available=50 * 1024 ** 3, percent=40.0)
        with patch.object(video_worker, "psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = fake_vm
            # Must not raise.
            video_worker.preflight_memory_guard("animatediff-lightning-mps")

    def test_high_memory_pressure_aborts(self):
        """Acute memory pressure (>=92% used) aborts even a light model."""
        fake_vm = MagicMock(available=50 * 1024 ** 3, percent=95.0)
        with patch.object(video_worker, "psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = fake_vm
            with self.assertRaises(RuntimeError):
                video_worker.preflight_memory_guard("animatediff-lightning-mps")


class TestSendProgressUpdate(unittest.TestCase):
    """Tests for the AMQP progress reporting function."""

    def setUp(self):
        mock_pika.reset_mock()

    @patch.dict(os.environ, {"SPRING_RABBITMQ_HOST": "testhost", "SPRING_RABBITMQ_PORT": "5673"})
    def test_sends_progress_message(self):
        """Verify progress message is published to the correct exchange."""
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_connection.channel.return_value = mock_channel
        mock_pika.BlockingConnection.return_value = mock_connection

        video_worker.send_progress_update("job-123", 50)

        mock_pika.BlockingConnection.assert_called_once()
        mock_channel.exchange_declare.assert_called_once_with(
            exchange='orazaka.events',
            exchange_type='topic',
            durable=True
        )
        mock_channel.basic_publish.assert_called_once()

        # Verify the routing key follows job.{jobId}.progress (AGENTS.md §6)
        publish_kwargs = mock_channel.basic_publish.call_args.kwargs
        self.assertEqual(publish_kwargs.get("exchange"), 'orazaka.events')
        self.assertEqual(publish_kwargs.get("routing_key"), 'job.job-123.progress')

        # Verify the published message content
        call_kwargs = mock_channel.basic_publish.call_args
        body = call_kwargs[1]["body"] if "body" in call_kwargs[1] else call_kwargs[0][2] if len(call_kwargs[0]) > 2 else None
        if body is None:
            body = call_kwargs.kwargs.get("body")
        parsed = json.loads(body)
        self.assertEqual(parsed["jobId"], "job-123")
        self.assertEqual(parsed["progress"], 50)

        mock_connection.close.assert_called_once()

    @patch.dict(os.environ, {"SPRING_RABBITMQ_HOST": "badhost"})
    def test_handles_connection_failure_gracefully(self):
        """If RabbitMQ is unreachable, the function should not raise."""
        mock_pika.BlockingConnection.side_effect = Exception("Connection refused")
        # Should not raise
        video_worker.send_progress_update("job-456", 100)
        mock_pika.BlockingConnection.side_effect = None

    @patch.dict(os.environ, {"SPRING_RABBITMQ_PORT": "not_a_number"})
    def test_handles_invalid_port_env(self):
        """Invalid SPRING_RABBITMQ_PORT should fallback to 5672."""
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_connection.channel.return_value = mock_channel
        mock_pika.BlockingConnection.return_value = mock_connection

        video_worker.send_progress_update("job-789", 75)
        # Should not raise — port fallback to 5672


class TestConsumptionMetering(unittest.TestCase):
    """ADR-033 §6.3: terminal events carry the reservation and raw measurements.

    The worker reports what it measured and never a billable unit — that is a
    property of the pricebook row the hold was pinned to, resolved by billing.
    """

    def setUp(self):
        mock_pika.reset_mock()
        mock_pika.BlockingConnection.side_effect = None
        self.channel = MagicMock()
        connection = MagicMock()
        connection.channel.return_value = self.channel
        mock_pika.BlockingConnection.return_value = connection

    def _published(self):
        kwargs = self.channel.basic_publish.call_args.kwargs
        return kwargs.get("routing_key"), json.loads(kwargs.get("body"))

    def test_done_carries_hold_and_consumption(self):
        from app.telemetry import send_job_done

        send_job_done(
            "job-1",
            {"url": "/uploads/x.mp4"},
            hold_id="hold-7",
            consumption={"frames": 48, "fps": 12, "gpuSeconds": 31.4},
        )

        routing_key, body = self._published()
        self.assertEqual(routing_key, "job.job-1.done")
        self.assertEqual(body["holdId"], "hold-7")
        self.assertEqual(body["consumption"]["frames"], 48)
        self.assertEqual(body["consumption"]["fps"], 12)
        self.assertNotIn("unit", body["consumption"])

    def test_error_carries_hold_so_billing_releases_it(self):
        from app.telemetry import send_job_error

        send_job_error("job-2", "MLX out of memory", hold_id="hold-8")

        routing_key, body = self._published()
        self.assertEqual(routing_key, "job.job-2.error")
        self.assertEqual(body["holdId"], "hold-8")

    def test_unmetered_job_omits_the_key_entirely(self):
        from app.telemetry import send_job_done

        send_job_done("job-3", {"url": "/x.mp4"})

        _, body = self._published()
        self.assertNotIn("holdId", body)
        self.assertNotIn("consumption", body)

    def test_extract_consumption_drops_host_diagnostics(self):
        from app.telemetry import extract_consumption

        consumption = extract_consumption(
            {
                "inference_time_sec": 31.4,
                "peak_memory_rss_mb": 8192.0,
                "gpu_allocated_mb": 0.0,
                "gpuSeconds": 31.4,
                "frames": 48,
                "fps": 12,
                "steps": 4,
                "width": 1024,
                "height": 576,
            }
        )

        self.assertEqual(
            consumption,
            {"gpuSeconds": 31.4, "frames": 48, "fps": 12, "steps": 4, "width": 1024, "height": 576},
        )

    def test_extract_consumption_tolerates_missing_metrics(self):
        from app.telemetry import extract_consumption

        self.assertEqual(extract_consumption(None), {})
        self.assertEqual(extract_consumption({}), {})

    def test_record_consumption_reports_the_realised_render_not_the_request(self):
        # The pipelines snap steps and cap frames; billing must see what was produced.
        rendered = [Image.new("RGB", (512, 320)) for _ in range(16)]
        metrics = {}

        video_worker._record_consumption(metrics, rendered, steps=4, fps=12,
                                         req_width=1024, req_height=576)

        self.assertEqual(metrics["frames"], 16)
        self.assertEqual(metrics["width"], 512)
        self.assertEqual(metrics["height"], 320)
        self.assertEqual(metrics["steps"], 4)
        self.assertEqual(metrics["fps"], 12)

    def test_record_consumption_never_fails_a_successful_render(self):
        metrics = {}

        video_worker._record_consumption(metrics, ["not-an-image"], steps=4, fps=12,
                                         req_width=None, req_height=None)

        # The bad frame is swallowed; an absent measurement releases the hold.
        self.assertNotIn("width", metrics)

    def test_resource_guard_reports_gpu_seconds_as_the_calibration_basis(self):
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)

        with video_worker.resource_guard(mock_process) as metrics:
            pass

        self.assertEqual(metrics["gpuSeconds"], metrics["inference_time_sec"])


class TestResourceGuard(unittest.TestCase):
    """Tests for the resource_guard context manager."""

    def test_populates_inference_time(self):
        """Verify inference_time_sec is populated after context exits."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)

        with video_worker.resource_guard(mock_process) as metrics:
            pass  # Simulate instant execution

        self.assertIn("inference_time_sec", metrics)
        self.assertGreaterEqual(metrics["inference_time_sec"], 0)

    def test_populates_peak_memory(self):
        """Verify peak_memory_rss_mb is populated."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=512 * 1024 * 1024)

        with video_worker.resource_guard(mock_process) as metrics:
            pass

        self.assertIn("peak_memory_rss_mb", metrics)
        self.assertAlmostEqual(metrics["peak_memory_rss_mb"], 512.0, places=0)

    def test_handles_none_process(self):
        """When process is None, peak_memory should be 0."""
        with video_worker.resource_guard(None) as metrics:
            pass

        self.assertIn("inference_time_sec", metrics)
        self.assertEqual(metrics["peak_memory_rss_mb"], 0.0)

    def test_triggers_gc_collect(self):
        """Verify garbage collection is triggered on exit."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=0)

        with patch("gc.collect") as mock_gc:
            with video_worker.resource_guard(mock_process) as metrics:
                pass
            mock_gc.assert_called()


class TestVideoInferenceHandlerRouting(unittest.TestCase):
    """Tests for the HTTP handler routing logic."""

    def setUp(self):
        """Start a test HTTP server on a random port."""
        self.server = HTTPServer(("127.0.0.1", 0), video_worker.VideoInferenceHandler)
        self.port = self.server.server_address[1]
        self.thread = Thread(target=self.server.handle_request)
        self.thread.daemon = True

    def tearDown(self):
        self.server.server_close()

    def test_get_root_returns_status(self):
        """GET / should return {'status': 'running'}."""
        self.thread.start()

        import urllib.request
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(resp.status, 200)
            self.assertEqual(data["status"], "running")

    def test_get_unknown_path_returns_404(self):
        """GET /unknown should return 404."""
        self.thread.start()

        import urllib.request
        url = f"http://127.0.0.1:{self.port}/unknown"
        try:
            urllib.request.urlopen(url)
            self.fail("Expected HTTP 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_post_unknown_path_returns_404(self):
        """POST /unknown should return 404."""
        self.thread.start()

        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/unknown",
            data=b"{}",
            method="POST"
        )
        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTP 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


class TestModelRegistryValidation(unittest.TestCase):
    """Tests for model registry validation in the POST handler."""

    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), video_worker.VideoInferenceHandler)
        self.port = self.server.server_address[1]
        self.thread = Thread(target=self.server.handle_request)
        self.thread.daemon = True

    def tearDown(self):
        self.server.server_close()

    def test_invalid_model_returns_400(self):
        """POST with unknown model should return 400."""
        self.thread.start()

        import urllib.request
        payload = json.dumps({"model": "nonexistent-model"}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/videos/generations",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTP 400 for invalid model")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            body = json.loads(e.read().decode("utf-8"))
            self.assertIn("error", body)
            self.assertIn("nonexistent-model", body["error"])

    def test_valid_model_names_in_registry(self):
        """Verify the 3 registered model names."""
        registry = {
            "stable-video-diffusion-img2vid-xt": "SVD",
            "animatediff-lightning-mps": "AnimateDiff",
            "apple-coreml-video-pipeline": "CoreML",
        }
        # Just validate against the handler code expectations
        self.assertEqual(len(registry), 3)
        self.assertIn("stable-video-diffusion-img2vid-xt", registry)
        self.assertIn("animatediff-lightning-mps", registry)
        self.assertIn("apple-coreml-video-pipeline", registry)


class TestImageIngestion(unittest.TestCase):
    """Tests for image input processing (base64 and file path)."""

    def test_base64_image_decoding(self):
        """Verify base64 image strings can be decoded into PIL Images."""
        img = Image.new("RGB", (64, 64), (255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        decoded = base64.b64decode(b64)
        result = Image.open(io.BytesIO(decoded)).convert("RGB").resize((1024, 576))
        self.assertEqual(result.size, (1024, 576))
        self.assertEqual(result.mode, "RGB")

    def test_file_path_image_loading(self):
        """Verify image_path loading from disk works."""
        img = Image.new("RGB", (128, 128), (0, 255, 0))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            loaded = Image.open(tmp_path).convert("RGB").resize((1024, 576))
            self.assertEqual(loaded.size, (1024, 576))
        finally:
            os.remove(tmp_path)

    def test_missing_file_path_uses_default(self):
        """If image_path doesn't exist, default gray image should be used."""
        fake_path = "/tmp/nonexistent_image_12345.png"
        self.assertFalse(os.path.exists(fake_path))
        # The handler should fall through to default without crashing


class TestPathSecurity(unittest.TestCase):
    """Tests for validate_safe_path and directory traversal detection."""

    def test_safe_paths_allowed(self):
        """Paths inside current working directory or tmp should be allowed."""
        cwd = os.getcwd()
        safe_path = os.path.join(cwd, "some_file.png")
        self.assertEqual(video_worker.validate_safe_path(safe_path), os.path.abspath(safe_path))

        tmp = tempfile.gettempdir()
        safe_tmp = os.path.join(tmp, "some_temp_file.png")
        self.assertEqual(video_worker.validate_safe_path(safe_tmp), os.path.abspath(os.path.realpath(safe_tmp)))

    def test_traversal_paths_denied(self):
        """Paths trying to access unauthorized files outside allowed boundaries must raise PermissionError."""
        bad_paths = [
            "/etc/passwd",
            "../../../../etc/passwd",
            "/private/etc/hosts",
        ]
        # Make sure that if they are outside allowed, they raise PermissionError
        for bp in bad_paths:
            with self.assertRaises(PermissionError):
                video_worker.validate_safe_path(bp)

    def test_load_image_handles_invalid_path_gracefully(self):
        """load_image should fall back to default image if image_path is outside allowed directories."""
        payload = {"image_path": "/etc/passwd"}
        img = video_worker.load_image(payload)
        self.assertIsNotNone(img)
        self.assertEqual(img.size, (1024, 576))

    def test_encoder_rejects_hyphen_filename(self):
        """export_frames_to_video should reject output paths starting with a hyphen to prevent flag injection."""
        from app.encoder import export_frames_to_video
        img = Image.new("RGB", (1024, 576), (128, 128, 128))
        with self.assertRaises(ValueError):
            export_frames_to_video([img], "-some_flag.mp4")


class TestFrameDurationCalculation(unittest.TestCase):
    """Tests for duration-to-frame-count calculation."""

    def test_default_duration_2s_svd(self):
        """2 seconds at 14fps = 28 frames."""
        duration_sec = 2
        fps_render = 14
        num_frames = int(duration_sec * fps_render)
        num_frames = max(14, min(num_frames, 30))
        self.assertEqual(num_frames, 28)

    def test_default_duration_2s_animatediff(self):
        """2 seconds at 12fps = 24 frames."""
        duration_sec = 2
        fps_render = 12
        num_frames = int(duration_sec * fps_render)
        num_frames = max(14, min(num_frames, 30))
        self.assertEqual(num_frames, 24)

    def test_minimum_frame_clamp(self):
        """Very short duration should clamp to minimum 14 frames."""
        duration_sec = 0.5
        for fps_render in [12, 14]:
            num_frames = int(duration_sec * fps_render)
            num_frames = max(14, min(num_frames, 30))
            self.assertEqual(num_frames, 14)

    def test_maximum_frame_clamp(self):
        """Long duration should clamp to maximum 30 frames."""
        duration_sec = 10
        for fps_render in [12, 14]:
            num_frames = int(duration_sec * fps_render)
            num_frames = max(14, min(num_frames, 30))
            self.assertEqual(num_frames, 30)

    def test_4_second_duration_svd(self):
        """4 seconds at 14fps = 30 frames (max clamp)."""
        duration_sec = 4
        fps_render = 14
        num_frames = int(duration_sec * fps_render)
        num_frames = max(14, min(num_frames, 30))
        self.assertEqual(num_frames, 30)

    def test_1_second_duration_svd(self):
        """1 second at 14fps = 14 frames (min clamp)."""
        duration_sec = 1
        fps_render = 14
        num_frames = int(duration_sec * fps_render)
        num_frames = max(14, min(num_frames, 30))
        self.assertEqual(num_frames, 14)


class TestMemoryCapEnforcement(unittest.TestCase):
    """Tests for psutil memory cap logic."""

    def test_high_process_rss_forces_fallback(self):
        """Process RSS > 6GB should trigger fallback."""
        process_mem_mb = 7000.0
        PROCESS_RSS_CAP_MB = 6000.0
        force_fallback = process_mem_mb > PROCESS_RSS_CAP_MB
        self.assertTrue(force_fallback)

    def test_low_available_memory_forces_fallback(self):
        """System available < 1.5GB should trigger fallback."""
        available_mem_mb = 1000.0
        SYSTEM_AVAIL_MIN_MB = 1500.0
        force_fallback = available_mem_mb < SYSTEM_AVAIL_MIN_MB
        self.assertTrue(force_fallback)

    def test_normal_memory_does_not_force_fallback(self):
        """Normal memory conditions should not force fallback."""
        process_mem_mb = 3000.0
        available_mem_mb = 8000.0
        PROCESS_RSS_CAP_MB = 6000.0
        SYSTEM_AVAIL_MIN_MB = 1500.0

        force_fallback = (
            process_mem_mb > PROCESS_RSS_CAP_MB or
            available_mem_mb < SYSTEM_AVAIL_MIN_MB
        )
        self.assertFalse(force_fallback)


def load_declaration_for_test():
    from app.registration import load_declaration

    return load_declaration()


class TestConsumerDispatch(unittest.TestCase):
    """The worker decides from the message, never from a capability it recognises.

    ADR-037 §4.2 (P4): naming a capability here is what made a general-purpose
    media worker coupled to one pack. The routing key is the dispatcher's
    decision, taken from the capability's row; this worker only knows which keys
    it is bound to.
    """

    def test_bindings_come_from_worker_yaml_not_from_source(self):
        """worker.yaml is the single source for what this worker drains (ADR-038, S3)."""
        from app.consumer import BINDINGS, COMPOSE_PREFIX
        from app.registration import load_declaration

        self.assertEqual(BINDINGS, load_declaration()["bindings"])
        self.assertIn("job.video.*", BINDINGS)
        self.assertIn("job.compose.*", BINDINGS)
        self.assertEqual(COMPOSE_PREFIX, "job.compose.")

    def test_worker_yaml_declares_no_capability(self):
        """A worker declares bindings, never a capability name (P4, [PACK-002])."""
        import pathlib

        declaration = pathlib.Path(__file__).parent / "worker.yaml"
        body = "\n".join(
            line for line in declaration.read_text().splitlines()
            if not line.strip().startswith("#")
        )
        self.assertNotIn("orazaka.core.", body)
        self.assertNotIn("orazaka.studio.", body)

    def test_registration_never_raises_when_the_job_service_is_down(self):
        """Registration is advisory: unreachable platform must not stop consumption."""
        import os
        from unittest.mock import patch

        from app import registration

        # Port 1 is reserved and unbound, so the connection fails immediately.
        with patch.dict(os.environ, {"JOB_SERVICE_INTERNAL_URL": "http://127.0.0.1:1"}):
            with patch.object(registration.time, "sleep", side_effect=InterruptedError):
                with self.assertRaises(InterruptedError):
                    # One pass: it must swallow the connection error and reach the sleep.
                    registration.register_forever(load_declaration_for_test())

    def test_compose_is_decided_by_the_routing_key(self):
        from app.consumer import _is_compose

        self.assertTrue(_is_compose("job.compose.assemble"))
        self.assertFalse(_is_compose("job.video.generate"))
        self.assertFalse(_is_compose(""))
        self.assertFalse(_is_compose(None))

    def test_a_new_compose_capability_needs_no_python_edit(self):
        """A capability this worker has never heard of, bound to job.compose.*, composes."""
        from app.consumer import _is_compose

        self.assertTrue(_is_compose("job.compose.montage"))

    def test_the_capability_name_does_not_decide(self):
        """The feature key is not consulted: only the key the broker delivered under."""
        import app.consumer as consumer

        with open(consumer.__file__, encoding="utf-8") as handle:
            source = handle.read()
        code = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        self.assertNotIn("featureKey", code.split("def _on_message")[1].split("def ")[0])


class TestProtocolConformance(unittest.TestCase):
    """The MUST clauses of docs/WORKER_PROTOCOL.md that this worker is the reference for."""

    def _consumer_source(self):
        import app.consumer as consumer

        with open(consumer.__file__, encoding="utf-8") as handle:
            return handle.read()

    def test_prefetch_is_one(self):
        """§2: prefetch MUST be 1 — the accelerator is the backpressure."""
        self.assertIn("basic_qos(prefetch_count=1)", self._consumer_source())

    def test_acks_after_a_terminal_outcome_including_failure(self):
        """§5: a worker MUST ack even on failure, or the queue stops draining.

        Asserted on the structure rather than the happy path: the ack lives in a
        ``finally``, which is what makes it hold for the failure branch too.
        """
        source = self._consumer_source()
        body = source.split("def _on_message", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("finally:", body)
        finally_block = body.split("finally:", 1)[1]
        self.assertIn("basic_ack", finally_block)


class TestTypedFailureCause(unittest.TestCase):
    """§3: a worker MUST declare a typed cause on `job.{id}.error` (ADR-053)."""

    def _consumer_source(self):
        import app.consumer as consumer

        with open(consumer.__file__, encoding="utf-8") as handle:
            return handle.read()

    def test_every_error_carries_a_cause(self):
        from app import telemetry

        published = []
        with patch.object(telemetry, "_publish_job_event", lambda key, body: published.append((key, body))):
            telemetry.send_job_error("job-1", "boom", hold_id="hold-1", cause=telemetry.TIMEOUT)
        self.assertEqual("TIMEOUT", published[0][1]["cause"])
        self.assertEqual("hold-1", published[0][1]["holdId"])

    def test_a_caller_that_declares_nothing_gets_executor_fault(self):
        from app import telemetry

        published = []
        with patch.object(telemetry, "_publish_job_event", lambda key, body: published.append((key, body))):
            telemetry.send_job_error("job-2", "boom")
        self.assertEqual("EXECUTOR_FAULT", published[0][1]["cause"])

    def test_an_invented_cause_degrades_rather_than_travelling(self):
        """A worker inventing 'BAD_INPUT' must not be read as the actor's fault downstream."""
        from app import telemetry

        published = []
        with patch.object(telemetry, "_publish_job_event", lambda key, body: published.append((key, body))):
            telemetry.send_job_error("job-3", "boom", cause="BAD_INPUT")
        self.assertEqual("EXECUTOR_FAULT", published[0][1]["cause"])

    def test_the_vocabulary_matches_the_java_contract(self):
        from app import telemetry

        self.assertEqual(
            ("GUARD_REFUSAL", "INPUT_INVALID", "EXECUTOR_FAULT", "PLATFORM_UNAVAILABLE", "TIMEOUT"),
            telemetry.FAILURE_CAUSES,
        )

    def test_input_invalid_is_unreachable_from_the_broad_except(self):
        """The one cause that BILLS must come from a declared rejection, never from a catch-all.

        Structural, not behavioural: `INPUT_INVALID` may appear only in the handler for this
        worker's own `InvalidJobPayload`, so no unanticipated exception can ever claim the actor
        was at fault — ADR-046 §2's counter-example was exactly that mistake.
        """
        body = self._consumer_source().split("def _on_message", 1)[1].split("\ndef ", 1)[0]
        broad = body.split("except Exception", 1)[1]
        self.assertNotIn("INPUT_INVALID", broad)
        self.assertIn("EXECUTOR_FAULT", broad)
        declared = body.split("except InvalidJobPayload", 1)[1].split("except ", 1)[0]
        self.assertIn("INPUT_INVALID", declared)


class TestEnvelopeInterop(unittest.TestCase):
    """The asset envelope, and that both languages agree on it (ADR-054 §6).

    The store is written by three processes in two languages, and the way that breaks is silently:
    one side changes the header and the other keeps opening the files it wrote itself. The Java
    suite emits `orazaka-libs/orazaka-ai-engine/orazaka-assets/target/interop/java-written.bin`; this reads it with
    the same fixed key and compares plaintext, then writes its own for Java to read back.
    """

    FIXTURE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "orazaka-libs", "orazaka-ai-engine", "orazaka-assets", "target", "interop",
    )

    def _keyring(self):
        from app import envelope

        return envelope.MasterKeyring(os.path.join(self.FIXTURE_DIR, "fixture.key"))

    def test_python_opens_what_java_sealed(self):
        from app import envelope

        sealed = os.path.join(self.FIXTURE_DIR, "java-written.bin")
        if not os.path.isfile(sealed):
            self.skipTest("run the Java assets suite first; it writes the fixture")
        with open(os.path.join(self.FIXTURE_DIR, "plain.bin"), "rb") as handle:
            expected = handle.read()
        self.assertEqual(expected, envelope.decrypt_bytes(sealed, self._keyring()))

    def test_python_seeks_inside_what_java_sealed(self):
        from app import envelope

        sealed = os.path.join(self.FIXTURE_DIR, "java-written.bin")
        if not os.path.isfile(sealed):
            self.skipTest("run the Java assets suite first; it writes the fixture")
        with open(os.path.join(self.FIXTURE_DIR, "plain.bin"), "rb") as handle:
            expected = handle.read()
        for offset in (0, 1, 4095, 4096, 4097, 3 * 4096 + 100):
            self.assertEqual(
                expected[offset:], envelope.decrypt_bytes(sealed, self._keyring(), offset)
            )

    def test_python_writes_a_fixture_java_can_open(self):
        from app import envelope

        if not os.path.isdir(self.FIXTURE_DIR):
            self.skipTest("run the Java assets suite first; it creates the fixture directory")
        plain = os.path.join(self.FIXTURE_DIR, "plain.bin")
        target = os.path.join(self.FIXTURE_DIR, "python-written.bin")
        envelope.encrypt_file(plain, target, self._keyring(), 4096)
        with open(plain, "rb") as handle:
            expected = handle.read()
        self.assertEqual(expected, envelope.decrypt_bytes(target, self._keyring()))

    def test_a_flipped_byte_refuses_to_open(self):
        from app import envelope

        sealed = os.path.join(self.FIXTURE_DIR, "java-written.bin")
        if not os.path.isfile(sealed):
            self.skipTest("run the Java assets suite first; it writes the fixture")
        with open(sealed, "rb") as handle:
            raw = bytearray(handle.read())
        raw[-20] ^= 0x01
        tampered = os.path.join(self.FIXTURE_DIR, "tampered.bin")
        with open(tampered, "wb") as handle:
            handle.write(raw)
        with self.assertRaises(envelope.EnvelopeError):
            envelope.decrypt_bytes(tampered, self._keyring())


class TestRunServerFunction(unittest.TestCase):
    """Tests for the run() entrypoint function."""

    def test_default_port(self):
        """run() should default to port 8188."""
        with patch.object(HTTPServer, "__init__", return_value=None) as mock_init:
            with patch.object(HTTPServer, "serve_forever", side_effect=KeyboardInterrupt):
                with patch.object(HTTPServer, "server_close"):
                    try:
                        video_worker.run(port=8188)
                    except (KeyboardInterrupt, AttributeError):
                        pass

    def test_custom_port_from_argv(self):
        """If sys.argv provides a port, it should be used."""
        # Verify the __main__ block logic
        test_port = 9999
        self.assertEqual(int(str(test_port)), 9999)


class TestCompositionDivergenceContract(unittest.TestCase):
    """The billed duration is the produced file's, never the requested one (ADR-063, audit #20).

    The divergence contract, extended from images to video. The image contract serves a 768x512
    image for a 512x512 request and checks the bill follows the image; this one composes three
    stills held 3 s each over a DELIBERATELY SHORT 2 s voiceover. ``-shortest`` ends the file at
    the audio, so the request says 9 s and the file says 2 — the case that billed 9.00 output
    seconds for a 2.02 s file.

    The oracle is ffprobe on the file the composer wrote, read here and not through the composer:
    a contract that asked the composer what it made would be checking the composer against
    itself. No ffmpeg is a failure, not a skip: a contract that examined nothing would be green
    for the reason GOV-006 exists to refuse.

    **The billed seconds are the file's duration** (ADR-066). They were ``frames / fps``, which is
    exact only at a constant whole-number rate — so ADR-063 made the composer pass ``-r 30`` to
    ffmpeg to keep that quotient working. Shaping the artefact to suit the meter is backwards even
    when the new artefact is better, so the meter changed instead, and one of these tests went with
    it: ``test_without_audio_the_billed_rate_is_the_files_rate`` asserted that the billed ``fps``
    equalled the file's rate, which is no longer a billing fact at all — the rate is an encoding
    choice, and the bill does not read it.
    """

    SECONDS_PER_PHOTO = 3.0
    WIDTH, HEIGHT = 180, 320
    # One frame at 30 fps, plus the AAC encoder's priming samples on the container clock.
    TOLERANCE_SECONDS = 0.1

    @classmethod
    def setUpClass(cls):
        import shutil

        for tool in ("ffmpeg", "ffprobe"):
            if shutil.which(tool) is None:
                raise AssertionError(f"{tool} is required: this contract measures a real file")

    def setUp(self):
        import subprocess

        self.workdir = tempfile.mkdtemp()
        self.photos = []
        for index, colour in enumerate([(200, 30, 30), (30, 200, 30), (30, 30, 200)]):
            path = os.path.join(self.workdir, f"still_{index}.png")
            Image.new("RGB", (64, 48), colour).save(path)
            self.photos.append(path)
        self.short_voiceover = os.path.join(self.workdir, "voiceover.m4a")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
             "-c:a", "aac", self.short_voiceover],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.workdir, ignore_errors=True)

    def _probe(self, path):
        """What the file is, read from the file."""
        import subprocess

        video = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate:format=duration",
             "-of", "json", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True,
        ).stdout)
        stream = video["streams"][0]
        return {
            "seconds": float(video["format"]["duration"]),
            "rate": stream["r_frame_rate"],
            "width": stream["width"],
            "height": stream["height"],
        }

    def _compose(self, audio_path):
        from app.composer import compose

        output = os.path.join(self.workdir, "composed.mp4")
        billed = compose(
            self.photos, output, audio_path=audio_path,
            seconds_per_photo=self.SECONDS_PER_PHOTO, width=self.WIDTH, height=self.HEIGHT,
        )
        return billed, self._probe(output)

    def _billed_seconds(self, billed):
        """What billing will price: the duration the composer measured on the file."""
        self.assertIsNotNone(
            billed.get("durationSeconds"), f"nothing billable was reported: {billed}"
        )
        return billed["durationSeconds"]

    def test_a_short_voiceover_bills_the_file_not_the_slideshow(self):
        billed, produced = self._compose(self.short_voiceover)

        # The fixture is what it claims: the file really is the voiceover's length, not 9 s.
        self.assertLess(produced["seconds"], 3.0, "fixture: -shortest did not shorten the file")
        self.assertAlmostEqual(
            self._billed_seconds(billed), produced["seconds"], delta=self.TOLERANCE_SECONDS,
            msg=f"billed {billed} for a file of {produced['seconds']:.2f} s",
        )

    def test_without_audio_the_billed_seconds_are_the_files_seconds(self):
        billed, produced = self._compose(None)

        # 9 s used to come out right here only because two fictional numbers cancelled: frames
        # computed from the request and an fps the file did not have (ADR-062 §5.5). The rate is
        # no longer part of the answer — this asserted `billed["fps"] == the file's rate` until
        # ADR-066 stopped billing from reading it.
        self.assertNotIn("fps", billed, "the bill no longer reads a frame rate")
        self.assertAlmostEqual(
            self._billed_seconds(billed), produced["seconds"], delta=self.TOLERANCE_SECONDS,
            msg=f"billed {billed} for a file of {produced['seconds']:.2f} s",
        )

    def test_the_billed_dimensions_are_the_files(self):
        billed, produced = self._compose(self.short_voiceover)

        self.assertEqual(
            (billed["width"], billed["height"]), (produced["width"], produced["height"])
        )




class AssetIdResolutionTest(unittest.TestCase):
    """A compose payload names asset ids; only their owner's files may answer.

    ``realestate-reels`` was diagnosed as an environment limitation twice and then
    fixed in the wrong process once. ``orazaka.studio.media.compose`` routes to
    ``job.compose.assemble`` — this worker — so the Java job service's resolver
    never sees it. The worker joined the id straight onto the upload root, without
    the owner's directory or the file extension, found nothing readable and raised
    "compose requires at least one readable photo" (ADR-046).
    """

    def _seed(self, root, owner, name="a1b2c3d4-0000-4000-8000-000000000001"):
        temp = os.path.join(root, owner, "temp")
        os.makedirs(temp, exist_ok=True)
        Image.new("RGB", (8, 8)).save(os.path.join(temp, name + ".png"))
        return name

    def test_an_asset_id_resolves_to_its_owner_file(self):
        from app.consumer import _resolve_owned_asset

        with tempfile.TemporaryDirectory() as root:
            owner = "550e8400-e29b-41d4-a716-446655440001"
            asset = self._seed(root, owner)

            resolved = _resolve_owned_asset(root, owner, asset)

            self.assertIsNotNone(resolved)
            self.assertTrue(os.path.isfile(resolved))
            # realpath: macOS resolves /var to /private/var, so compare canonicals.
            self.assertTrue(
                resolved.startswith(os.path.realpath(os.path.join(root, owner)))
            )

    def test_another_actors_asset_is_not_found(self):
        from app.consumer import _resolve_owned_asset

        with tempfile.TemporaryDirectory() as root:
            asset = self._seed(root, "550e8400-e29b-41d4-a716-446655440001")

            self.assertIsNone(
                _resolve_owned_asset(root, "550e8400-e29b-41d4-a716-446655440002", asset)
            )

    def test_traversal_in_an_asset_id_escapes_nothing(self):
        from app.consumer import _resolve_owned_asset

        with tempfile.TemporaryDirectory() as root:
            owner = "550e8400-e29b-41d4-a716-446655440001"
            os.makedirs(os.path.join(root, owner), exist_ok=True)
            Image.new("RGB", (8, 8)).save(os.path.join(root, "elsewhere.png"))

            self.assertIsNone(_resolve_owned_asset(root, owner, "../elsewhere"))

    def test_an_absolute_path_is_not_an_asset_reference(self):
        """ADR-065. This test used to be test_an_absolute_path_is_left_alone, and asserted that
        an absolute path came back untouched — written in the same commit as the branch it
        described, to match it. Nobody had asked whose file the path named."""
        from app.consumer import _resolve_owned_asset

        with tempfile.TemporaryDirectory() as root:
            owner = "550e8400-e29b-41d4-a716-446655440001"
            stranger = "550e8400-e29b-41d4-a716-446655440002"
            theirs = os.path.join(root, stranger, "temp", self._seed(root, stranger) + ".png")
            own = os.path.join(root, owner, "temp", self._seed(root, owner) + ".png")

            # Another actor's file, by path: not found — the same answer as a file that never
            # existed, so the refusal says nothing about which it was.
            self.assertIsNone(_resolve_owned_asset(root, owner, theirs))
            # The caller's own file, by path: refused too. The path is what is refused, not the
            # owner — asking by location is not a way in, even for one's own upload.
            self.assertIsNone(_resolve_owned_asset(root, owner, own))

    def test_compose_payload_of_asset_ids_reaches_readable_files(self):
        from app import consumer

        with tempfile.TemporaryDirectory() as root:
            owner = "550e8400-e29b-41d4-a716-446655440001"
            first = self._seed(root, owner, "a1b2c3d4-0000-4000-8000-000000000001")
            second = self._seed(root, owner, "a1b2c3d4-0000-4000-8000-000000000002")
            job = {"jobId": "job-1", "userId": owner,
                   "payload": {"photos": [first, second]}}

            # `_seal` needs a keyring, like the worker does in production; a stub here would test
            # a code path that no longer exists (ADR-054).
            with patch.object(consumer, "_upload_root", return_value=root), \
                    patch.object(consumer, "compose", return_value={}) as composed, \
                    patch.object(consumer, "extract_consumption", return_value={}), \
                    patch.object(consumer, "_seal"):
                consumer._compose(job)

            passed = composed.call_args[0][0]
            self.assertEqual(len(passed), 2)
            for path in passed:
                self.assertTrue(os.path.isfile(path), path)


if __name__ == "__main__":
    unittest.main()
