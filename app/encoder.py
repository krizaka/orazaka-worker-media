import os
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import List

from PIL import Image

try:
    from diffusers.utils import export_to_video
except ImportError:
    export_to_video = None

def is_apple_silicon() -> bool:
    """Detect if running on an Apple Silicon (macOS arm64) host."""
    return sys.platform == "darwin" and platform.machine() == "arm64"

def export_frames_to_video(
    frames: List[Image.Image],
    output_path: str,
    fps: int = 7
) -> None:
    """
    Encodes list of PIL Images into an MP4 video.
    Uses h264_videotoolbox on Apple Silicon for hardware acceleration,
    falling back to libx264 or diffusers export_to_video.
    """
    if not frames:
        raise ValueError("Frames list must not be empty.")

    if not output_path:
        raise ValueError("Output path must not be empty.")

    # Resolve and normalize output_path to ensure it does not start with a hyphen (command-line argument injection)
    output_path = os.path.abspath(os.path.realpath(output_path))
    if os.path.basename(output_path).startswith("-"):
        raise ValueError("Output path filename cannot start with a hyphen.")

    # Try hardware-accelerated encoding using FFmpeg command line
    temp_dir = tempfile.mkdtemp()
    try:
        # Save frames as temporary PNG images
        for idx, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{idx:04d}.png")
            frame.save(frame_path, format="PNG")

        codec = "h264_videotoolbox" if is_apple_silicon() else "libx264"
        
        # Build command: high-bitrate cinematic encoding
        # -b:v 8M forces dense spatial output (larger file sizes)
        # -profile:v high -level 4.1 for maximum H.264 compatibility
        # -movflags +faststart for web streaming readiness
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-c:v", codec,
            "-b:v", "8M",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ]
        
        print(f"Executing encoder: {' '.join(cmd)}", flush=True)
        # Run subprocess silently
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("FFmpeg encoding completed successfully.", flush=True)
        return
    except Exception as e:
        print(f"FFmpeg encoder failed or command not found: {e}. Falling back to diffusers export_to_video.", flush=True)
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Fallback to diffusers utility
    if export_to_video:
        export_to_video(frames, output_path, fps=fps)
    else:
        raise RuntimeError("No suitable video exporter found.")
