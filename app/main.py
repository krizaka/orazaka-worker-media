#!/usr/bin/env python3
import base64
import io
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from PIL import Image

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

from app.encoder import export_frames_to_video
from app.generator import generate_video_frames
from app.telemetry import resource_guard, send_progress_update

try:
    from app.mlx_generator import generate_video_frames_mlx
except ImportError:
    generate_video_frames_mlx = None

MODEL_REGISTRY: Dict[str, str] = {
    "stable-video-diffusion-img2vid-xt": "SVD",
    "animatediff-lightning-mps": "AnimateDiff",
    "apple-coreml-video-pipeline": "CoreML",
    "mlx-animatediff-lightning": "MLX-AnimateDiff",
    "mlx-stable-diffusion-video": "MLX-SVD",
    "stable-video-diffusion-img2vid-xt-mps-fp32": "SVD-MPS-FP32",
    # MLX Community HuggingFace models
    "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": "MLX-FLUX-Schnell",
    "ByteDance/AnimateDiff-Lightning": "MLX-AnimateDiff-Lightning",
}

# ── Host-memory guard (per user mandate) ──────────────────────────────────────
# Approximate peak unified-memory footprint per model (GB). The preflight uses
# this to FAIL oversized models fast and cleanly instead of letting them swap or
# freeze the host. Data-driven default selection is done in the router from the
# catalog; this is the worker's conservative, self-contained safety net.
MODEL_MEMORY_GB: Dict[str, int] = {
    "stable-video-diffusion-img2vid-xt": 96,
    "stable-video-diffusion-img2vid-xt-mps-fp32": 96,
    "animatediff-lightning-mps": 6,
    "mlx-animatediff-lightning": 6,
    "ByteDance/AnimateDiff-Lightning": 6,
    "mlx-stable-diffusion-video": 8,
    "apple-coreml-video-pipeline": 4,
    "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": 6,
}
_DEFAULT_MODEL_GB = 12  # conservative estimate for unregistered models

# Serialize heavy media work: at most one model resident / one inference at a time.
_media_semaphore = threading.Semaphore(1)


def preflight_memory_guard(model_name: str) -> None:
    """Fail fast/clean if the model cannot fit available unified memory.

    Compares psutil.virtual_memory().available against the model's estimated
    footprint plus a configurable headroom (ORAZAKA_MEDIA_MEM_HEADROOM_GB,
    default 14). Raises RuntimeError -> the job is reported FAILED with a clear
    reason. §12 Zero-Fallback: never fake output, never swap/freeze the machine.
    """
    headroom_gb = float(os.environ.get("ORAZAKA_MEDIA_MEM_HEADROOM_GB", "14"))
    est_gb = MODEL_MEMORY_GB.get(model_name, _DEFAULT_MODEL_GB)
    if psutil is None:
        print("[mem-guard] psutil unavailable — skipping preflight.", flush=True)
        return
    vm = psutil.virtual_memory()
    avail_gb = vm.available / (1024 ** 3)
    # Optional: also bail under acute memory pressure (>92% used).
    if vm.percent >= 92.0:
        raise RuntimeError(
            f"host under high memory pressure ({vm.percent:.0f}% used); refusing to "
            f"load model '{model_name}'. Free memory or pick a lighter model."
        )
    if est_gb > (avail_gb - headroom_gb):
        raise RuntimeError(
            f"model '{model_name}' needs ~{est_gb} GB but only {avail_gb:.1f} GB is "
            f"available (headroom {headroom_gb:.0f} GB); pick a lighter model such as "
            f"'animatediff-lightning-mps'"
        )
    print(
        f"[mem-guard] model='{model_name}' est={est_gb}GB avail={avail_gb:.1f}GB "
        f"headroom={headroom_gb:.0f}GB used={vm.percent:.0f}% -> OK",
        flush=True,
    )


def _upload_roots() -> list:
    """Resolve the monorepo upload directory (``UPLOAD_DIR``) to absolute root(s).

    The router writes generated media under ``<workspace>/var/orazaka-uploads``
    (see ``VideoGenerationStrategy`` / ``JobMediaHelper``) and hands the worker an
    absolute ``output_path`` there. The worker is spawned with a cwd that is NOT
    guaranteed to be the monorepo root, so anchoring the allow-list on cwd alone
    rejects legitimate writes ("outside allowed directories"). We anchor instead
    on this file's location (``app/main.py`` -> 4 levels up = monorepo root) and
    honor an absolute ``UPLOAD_DIR`` override. Only the uploads dir is allowed —
    not the whole workspace — so directory traversal stays blocked.
    """
    upload_dir = os.environ.get("UPLOAD_DIR", "var/orazaka-uploads")
    if os.path.isabs(upload_dir):
        candidates = [upload_dir]
    else:
        worker_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        candidates = [
            os.path.join(worker_root, upload_dir),
            os.path.join(os.getcwd(), upload_dir),
        ]
    return [os.path.abspath(os.path.realpath(c)) for c in candidates]


def validate_safe_path(path: str) -> str:
    """Validates that a path is safe to access, preventing directory traversal."""
    if not path:
        raise ValueError("Path is empty")

    # Resolve to absolute normalized path
    resolved_path = os.path.abspath(os.path.realpath(path))

    # Check if the resolved path is within allowed directories
    # Allowed: current working directory, the configured upload dir, and temp dirs
    allowed_roots = [
        os.path.abspath(os.path.realpath(os.getcwd())),
        os.path.abspath(os.path.realpath(tempfile.gettempdir())),
        *_upload_roots(),
    ]

    # Also allow standard system temp directories if different.
    # noqa S108: these paths are an *allowlist* used to validate a caller-supplied path stays
    # inside a known root — the inverse of the risk the rule describes, which is writing to a
    # predictable temp location.
    for t_dir in ["/tmp", "/private/tmp", "/var/tmp"]:  # noqa: S108
        if os.path.exists(t_dir):
            allowed_roots.append(os.path.abspath(os.path.realpath(t_dir)))

    is_safe = False
    for root in allowed_roots:
        try:
            if os.path.commonpath([root, resolved_path]) == root:
                is_safe = True
                break
        except ValueError:
            continue
            
    if not is_safe:
        raise PermissionError(f"Access to path '{path}' is denied (outside allowed directories).")
        
    return resolved_path


def load_image(payload: Dict[str, Any]) -> Image.Image:
    """Ingests starting image from base64 data or a local path, falling back to a default."""
    img = Image.new("RGB", (1024, 576), (128, 128, 128))
    if payload.get("image_path"):
        try:
            path = payload["image_path"]
            safe_path = validate_safe_path(path)
            if os.path.exists(safe_path):
                return Image.open(safe_path).convert("RGB").resize((1024, 576))
        except Exception as e:
            print(f"Could not open image_path: {e}", flush=True)
    elif payload.get("image"):
        try:
            img_data = base64.b64decode(payload["image"])
            return Image.open(io.BytesIO(img_data)).convert("RGB").resize((1024, 576))
        except Exception as e:
            print(f"Could not parse base64 image: {e}", flush=True)
    return img


def _generate_flux_image(
    prompt: str,
    model_name: str,
    steps: int,
    num_frames: int,
    seed: int,
) -> list:
    """Generate image frames using FLUX.1 via mflux (MLX native).

    Falls back to a prompt-colored gradient image if mflux is not installed.
    Returns a list of PIL Images suitable for video encoding.
    """
    try:
        from mflux import Config, Flux1

        print(f"Initializing FLUX via mflux: model={model_name}, steps={steps}, seed={seed}", flush=True)
        flux = Flux1(
            model_alias="schnell",
            quantize=4,
        )
        image = flux.generate_image(
            seed=seed,
            prompt=prompt,
            config=Config(
                num_inference_steps=steps,
                height=576,
                width=1024,
            ),
        )
        pil_image = image.image
        print(f"FLUX inference completed: {pil_image.size}", flush=True)
        # Return duplicated frames for video encoding compatibility
        return [pil_image.copy() for _ in range(num_frames)]

    except ImportError as exc:
        raise RuntimeError(
            "mflux is not installed. Cannot generate FLUX image without the dependency."
        ) from exc
    except Exception as e:
        raise RuntimeError(f"FLUX inference failed: {e}") from e


def _record_consumption(metrics, frames, steps, fps, req_width, req_height) -> None:
    """Record what was actually produced, for the settlement of this job's hold.

    Measurements only — no billable unit. Which unit a render bills in belongs
    to the pricebook row its hold was pinned to and is resolved by the billing
    service (design §7); a worker that named a unit would be a second copy of a
    pricing decision it does not own.

    Every value is the *realised* one, not the requested one: the pipelines snap
    steps to the variants Lightning supports and cap frames to what MPS memory
    allows, and billing a user for the render they asked for rather than the one
    they got is how a credit system loses trust.
    """
    try:
        if frames:
            metrics["frames"] = len(frames)
            metrics["images"] = len(frames)
            width, height = frames[0].size
            metrics["width"] = int(width)
            metrics["height"] = int(height)
        elif req_width and req_height:
            metrics["width"] = int(req_width)
            metrics["height"] = int(req_height)
        if steps:
            metrics["steps"] = int(steps)
        if fps:
            metrics["fps"] = int(fps)
    except Exception as e:
        # Metering must never fail a render that already succeeded; an absent
        # measurement releases the hold, which errs toward the user.
        print(f"Failed to record consumption metrics: {e}", flush=True)


class VideoInferenceHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "running"}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path != '/v1/videos/generations':
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get('Content-Length', 0))
        try:
            payload: Dict[str, Any] = json.loads(self.rfile.read(content_length).decode('utf-8'))
        except Exception:
            payload = {}

        model_name = payload.get("model", "animatediff-lightning-mps")
        if model_name not in MODEL_REGISTRY:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Model '{model_name}' is not registered."}).encode('utf-8'))
            return

        process = psutil.Process(os.getpid()) if psutil else None
        input_image = load_image(payload)
        try:
            with _media_semaphore, resource_guard(process) as metrics:
                seed_str = os.environ.get("ORAZAKA_VIDEO_GEN_SEED", "-1")
                try:
                    seed_val = int(seed_str)
                except ValueError:
                    seed_val = -1

                if seed_val == -1:
                    import random
                    seed_val = random.randint(0, 2147483647)  # noqa: S311 — a generation seed, not a secret

                print(f"Using video generation seed: {seed_val}", flush=True)
                generator = torch.manual_seed(seed_val) if torch else None
                job_id = payload.get("job_id")
                
                model_name = payload.get("model", "animatediff-lightning-mps")

                # Host-memory preflight: refuse oversized models BEFORE loading any
                # weights, so SVD-XT fails fast/clean on a 64GB Mac instead of
                # swapping/freezing the host (§12: honest failure, never fake output).
                preflight_memory_guard(model_name)

                steps = payload.get("videoSteps") or payload.get("video_steps")
                fps_render = payload.get("videoFps") or payload.get("video_fps")

                if not steps:
                    model_lower = model_name.lower()
                    if "flux" in model_lower:
                        steps = 4  # FLUX Schnell distilled
                    elif "animatediff" in model_lower:
                        steps = 16  # Deep inference for 64GB Unified Memory
                    else:
                        steps = 25

                if not fps_render:
                    model_lower = model_name.lower()
                    if "animatediff" in model_lower:
                        fps_render = 12
                    elif "flux" in model_lower:
                        fps_render = 1  # FLUX generates still images
                    else:
                        fps_render = 14

                if job_id:
                    send_progress_update(job_id, 0)
                
                def progress_cb(pipe_self: Any, step: int, timestep: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
                    if job_id:
                        progress = min(int((step + 1) / steps * 100), 100)
                        send_progress_update(job_id, progress)
                    return kwargs

                duration = payload.get("durationSeconds") or payload.get("video_length") or 2
                # Resolution from payload (defaults handled by mlx_generator)
                req_width = payload.get("width")
                req_height = payload.get("height")

                # MLX models: minimum 16 frames for cinematic output (4.0s at 4fps)
                is_mlx_model = model_name.startswith("mlx-") or model_name.startswith("mlx-community/") or model_name == "ByteDance/AnimateDiff-Lightning"
                min_frames = 16 if is_mlx_model else 14
                frames_n = max(min_frames, min(int(duration * fps_render), 30))

                # Allow explicit num_frames override from payload (backward-compatible control pathway)
                explicit_frames = payload.get("num_frames")
                if explicit_frames and int(explicit_frames) > 0:
                    frames_n = int(explicit_frames)

                if model_name.startswith("argmaxinc/"):
                    # FLUX image generation via mflux (MLX native)
                    frames = _generate_flux_image(
                        prompt=payload.get("prompt", ""),
                        model_name=model_name,
                        steps=steps,
                        num_frames=frames_n,
                        seed=seed_val,
                    )
                elif is_mlx_model:
                    if generate_video_frames_mlx is None:
                        raise ImportError("AnimateDiff-Lightning pipeline is not available on this host.")
                    mlx_kwargs = {
                        "input_image": input_image,
                        "num_frames": frames_n,
                        "num_inference_steps": steps,
                        "progress_callback": progress_cb,
                        "prompt": payload.get("prompt", ""),
                    }
                    if req_width:
                        mlx_kwargs["width"] = int(req_width)
                    if req_height:
                        mlx_kwargs["height"] = int(req_height)
                    frames = generate_video_frames_mlx(**mlx_kwargs)
                else:
                    frames = generate_video_frames(
                        input_image=input_image,
                        num_frames=frames_n,
                        num_inference_steps=steps,
                        generator=generator,
                        progress_callback=progress_cb
                    )

                _record_consumption(metrics, frames, steps, fps_render, req_width, req_height)

                output_path = payload.get("output_path")
                if output_path:
                    safe_output_path = validate_safe_path(output_path)
                    os.makedirs(os.path.dirname(safe_output_path), exist_ok=True)
                    export_frames_to_video(frames, safe_output_path, fps=fps_render)
                    if job_id:
                        send_progress_update(job_id, 100)
                    response_data = {"status": "success", "output_path": safe_output_path, "metrics": metrics}
                else:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        export_frames_to_video(frames, tmp_path, fps=fps_render)
                        with open(tmp_path, "rb") as f:
                            video_bytes = f.read()
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    
                    base64_video = base64.b64encode(video_bytes).decode('utf-8')
                    if job_id:
                        send_progress_update(job_id, 100)
                    response_data = {"data": [{"b64_json": base64_video}], "metrics": metrics}

            body = json.dumps(response_data)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode('utf-8'))
        except Exception as e:
            print(f"Error during video generation: {e}", flush=True)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

def run(port: int = 8188) -> None:
    server_address = ('', port)
    httpd = HTTPServer(server_address, VideoInferenceHandler)
    print(f"Video Worker listening on port {port}...", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == '__main__':
    port_arg = 8188
    if len(sys.argv) > 1:
        port_arg = int(sys.argv[1])
    run(port_arg)
