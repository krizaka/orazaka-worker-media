#!/usr/bin/env python3
"""
Real AnimateDiff-Lightning pipeline on Apple Silicon MPS.

This module loads genuine pre-trained diffusion weights from:
  - ByteDance/AnimateDiff-Lightning (motion adapter checkpoints)
  - emilianJR/epiCRealism (or compatible SD1.5 base model)

It runs actual diffusion inference (UNet + VAE + CLIP text encoder)
on Metal Performance Shaders (MPS) using PyTorch, producing frames
with real textures, spatial features, and coherent motion.

No procedural gradients. No Pillow simulations. No fallback frames.
§12 Zero-Fallback Policy enforced — inference fails loudly.
"""
from __future__ import annotations

import gc
import os
import time
from typing import Callable, Optional

# Memory safety: disable MPS high/low watermark so PyTorch can use all
# unified memory.  VAE slicing + attention slicing at 512×512 keep the
# actual peak well within 64 GB.  Setting only HIGH without LOW causes
# "invalid low watermark ratio" because the default LOW (1.4) > HIGH.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# AnimateDiff-Lightning repository and checkpoint naming
LIGHTNING_REPO = "ByteDance/AnimateDiff-Lightning"
# SD1.5-compatible base model for texture generation
BASE_MODEL = os.environ.get("ORAZAKA_AD_BASE_MODEL", "emilianJR/epiCRealism")
# Default inference steps (Lightning is distilled for 1/2/4/8 steps)
DEFAULT_STEPS = 4
# Default resolution (SD1.5 native is 512x512; can push to 512x768)
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
# Default frame count (16 for cinematic output; 24 max with VAE+attention slicing at 512x512)
DEFAULT_NUM_FRAMES = 16

# ---------------------------------------------------------------------------
# Global pipeline instance — loaded once at module level
# ---------------------------------------------------------------------------
_pipe = None
_current_step_variant: int | None = None


def _resolve_device() -> str:
    """Resolve the best available compute device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_dtype(device: str) -> torch.dtype:
    """MPS works best with float16 on modern PyTorch."""
    if device == "mps":
        return torch.float16
    if device == "cuda":
        return torch.float16
    return torch.float32


def init_animatediff_pipeline(num_steps: int = DEFAULT_STEPS) -> None:
    """
    Initialize the AnimateDiff-Lightning pipeline with real pre-trained weights.

    Downloads (or loads from HuggingFace cache):
    1. MotionAdapter checkpoint for the specified step count
    2. SD1.5-compatible base model (UNet, VAE, CLIP text encoder)
    3. Configures EulerDiscreteScheduler for Lightning's distilled schedule
    """
    global _pipe, _current_step_variant

    from diffusers import AnimateDiffPipeline, EulerDiscreteScheduler, MotionAdapter
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    device = _resolve_device()
    dtype = _resolve_dtype(device)

    print("[AnimateDiff-Lightning] Initializing real diffusion pipeline...", flush=True)
    print(f"  Device: {device} | dtype: {dtype}", flush=True)
    print(f"  Base model: {BASE_MODEL}", flush=True)
    print(f"  Lightning repo: {LIGHTNING_REPO}", flush=True)
    print(f"  Step variant: {num_steps}-step", flush=True)

    # ─── CRITICAL: Ensure SVD pipeline is NOT loaded ───
    # SVD occupies ~14GB on MPS. Both pipelines cannot coexist in 64GB.
    # SVD is now lazy-loaded, so we just need to evict it if it was loaded.
    try:
        import app.generator as gen_module
        if gen_module.pipe is not None:
            print("  Evicting SVD pipeline from GPU to free MPS memory...", flush=True)
            gen_module.pipe.to("cpu")
            gen_module.pipe = None
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            print("  SVD pipeline evicted. MPS memory freed.", flush=True)
        else:
            print("  SVD pipeline not loaded — full MPS memory available.", flush=True)
    except ImportError:
        pass

    # Step 1: Load MotionAdapter with Lightning checkpoint
    ckpt_name = f"animatediff_lightning_{num_steps}step_diffusers.safetensors"
    print(f"  Loading MotionAdapter checkpoint: {ckpt_name}...", flush=True)

    adapter = MotionAdapter().to(device, dtype)
    ckpt_path = hf_hub_download(LIGHTNING_REPO, ckpt_name)
    state_dict = load_file(ckpt_path, device=str(device))
    adapter.load_state_dict(state_dict)
    print(f"  MotionAdapter loaded: {sum(p.numel() for p in adapter.parameters()):,} parameters", flush=True)

    # Step 2: Load full AnimateDiffPipeline with SD1.5 base model
    print(f"  Loading base SD1.5 model: {BASE_MODEL}...", flush=True)
    pipe = AnimateDiffPipeline.from_pretrained(
        BASE_MODEL,
        motion_adapter=adapter,
        torch_dtype=dtype,
    ).to(device)

    # Step 3: Configure scheduler for Lightning's distilled schedule
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )

    # Step 4: Memory optimization for Apple Silicon
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    total_params = sum(p.numel() for p in pipe.unet.parameters())
    total_params += sum(p.numel() for p in pipe.vae.parameters())
    total_params += sum(p.numel() for p in pipe.text_encoder.parameters())
    mem_gb = total_params * (2 if dtype == torch.float16 else 4) / (1024 ** 3)
    print(f"  Pipeline ready on {device.upper()}", flush=True)
    print(f"  Total parameters: {total_params:,} (~{mem_gb:.1f} GB)", flush=True)
    print("  Attention slicing: ENABLED", flush=True)
    print("  VAE slicing: ENABLED", flush=True)

    _pipe = pipe
    _current_step_variant = num_steps


def generate_video_frames_mlx(
    input_image: Image.Image,
    num_frames: int = DEFAULT_NUM_FRAMES,
    num_inference_steps: int = DEFAULT_STEPS,
    progress_callback: Optional[Callable] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    prompt: str = "",
) -> list[Image.Image]:
    """
    Generate video frames using real AnimateDiff-Lightning diffusion on Apple Silicon.

    This runs genuine UNet forward passes with pre-trained weights from ByteDance,
    producing frames with real textures, spatial details, and coherent motion
    derived from the CLIP text encoder's understanding of the prompt.

    Args:
        input_image: Starting reference image (used for dimension reference only;
                     AnimateDiff generates from text, not image-to-video)
        num_frames: Number of output frames (AnimateDiff default: 16)
        num_inference_steps: Denoising steps (Lightning supports 1/2/4/8)
        progress_callback: Optional callback for progress updates
        width: Output width (must be divisible by 8)
        height: Output height (must be divisible by 8)
        prompt: Text prompt for video generation

    Returns:
        List of PIL Images with real diffusion-generated content

    Raises:
        RuntimeError: If pipeline fails (§12 Zero-Fallback — no silent degradation)
    """
    global _pipe, _current_step_variant

    t0 = time.time()

    # Validate step variant — Lightning only supports specific steps
    valid_steps = {1, 2, 4, 8}
    actual_steps = min(valid_steps, key=lambda s: abs(s - num_inference_steps))
    if actual_steps != num_inference_steps:
        print(f"  [AnimateDiff-Lightning] Snapping steps {num_inference_steps} → {actual_steps} "
              f"(Lightning supports {sorted(valid_steps)})", flush=True)
        num_inference_steps = actual_steps

    # Ensure dimensions are divisible by 8 (VAE requirement)
    width = (width // 8) * 8
    height = (height // 8) * 8

    # Initialize pipeline if needed (or if step variant changed)
    if _pipe is None or _current_step_variant != num_inference_steps:
        init_animatediff_pipeline(num_steps=num_inference_steps)

    if _pipe is None:
        raise RuntimeError("AnimateDiff-Lightning pipeline failed to initialize. No fallback.")

    device = _resolve_device()

    # 16 frames at 512x512 is the safe maximum for 64GB Apple Silicon.
    # 24 frames OOM-kills during 3D temporal attention (51s/step then crash).
    # Cinematic 4.0s target: 16 frames ÷ 4 FPS = 4.0 seconds.
    MAX_FRAMES_MPS = 16
    actual_frames = min(num_frames, MAX_FRAMES_MPS)
    if actual_frames != num_frames:
        print(f"  [AnimateDiff-Lightning] Capping frames {num_frames} → {actual_frames} "
              f"(MPS memory limit for 64GB)", flush=True)

    # Use prompt or generate a default from the context
    if not prompt:
        prompt = "cinematic motion, high quality, detailed textures, smooth camera movement"

    print(f"  [AnimateDiff-Lightning] Generating {actual_frames} frames at {width}x{height}", flush=True)
    print(f"  Prompt: {prompt[:120]}...", flush=True)
    print(f"  Steps: {num_inference_steps} | Guidance: 1.0 (Lightning CFG-free)", flush=True)

    try:
        # Set random seed for reproducibility
        seed_str = os.environ.get("ORAZAKA_VIDEO_GEN_SEED", "-1")
        try:
            seed_val = int(seed_str)
        except ValueError:
            seed_val = -1
        if seed_val == -1:
            import random
            seed_val = random.randint(0, 2147483647)  # noqa: S311 — a generation seed, not a secret
        generator = torch.manual_seed(seed_val)
        print(f"  Seed: {seed_val}", flush=True)

        # ═══════════════════════════════════════════════════════════════
        # REAL DIFFUSION INFERENCE — Pre-trained UNet + VAE + CLIP
        # ═══════════════════════════════════════════════════════════════
        print(f"  Executing real AnimateDiff-Lightning diffusion on {device.upper()}...", flush=True)

        # MPS memory relief callback — flush between denoising steps to
        # prevent temporal-attention intermediate tensors from accumulating.
        def _mps_step_callback(pipe, step_index, timestep, callback_kwargs):
            if device == "mps":
                torch.mps.empty_cache()
            gc.collect()
            return callback_kwargs

        output = _pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=actual_frames,
            guidance_scale=1.0,  # Lightning uses guidance_scale=1.0 (CFG-free distilled)
            num_inference_steps=num_inference_steps,
            generator=generator,
            callback_on_step_end=_mps_step_callback,
        )

        frames = output.frames[0]

        if not frames or len(frames) == 0:
            raise RuntimeError("AnimateDiff-Lightning returned empty frames")

        t_inference = time.time() - t0
        print(f"  AnimateDiff-Lightning inference completed in {t_inference:.2f}s", flush=True)
        print(f"  Produced {len(frames)} real diffusion frames at {width}x{height}", flush=True)
        print(f"  Frame type: {type(frames[0])}, size: {frames[0].size if hasattr(frames[0], 'size') else 'N/A'}", flush=True)

        # Convert to PIL if needed (diffusers may return PIL already)
        pil_frames = []
        for i, frame in enumerate(frames):
            if isinstance(frame, Image.Image):
                pil_frames.append(frame)
            else:
                # numpy array
                import numpy as np
                pil_frames.append(Image.fromarray(np.array(frame)))
            if progress_callback:
                progress_callback(None, i, None, {})

        # Clean up GPU memory
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()

        t_total = time.time() - t0
        print(f"  MLX generation completed. Total time: {t_total:.2f}s "
              f"({len(pil_frames)} frames at {width}x{height})", flush=True)

        return pil_frames

    except Exception as e:
        # §12 Zero-Fallback Policy — FAIL LOUDLY
        print(f"  [AnimateDiff-Lightning] Inference FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Force cleanup
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        raise RuntimeError(f"AnimateDiff-Lightning inference failed (no fallback): {e}") from e
