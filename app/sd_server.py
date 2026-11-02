"""Starts sd-server for one image model of ``models.yaml`` (``python -m app.sd_server flux-2-klein-4b``).

The same command line the Mac's CLI builds from the same recipe (``sdServerArgs`` in orazaka-cli):
each file under its role's flag, then the recipe's own sampling defaults. Requests still send their
size, steps, seed and negative prompt (ADR-075). On a card that must share its memory with a video
model, ``MEDIA_IMAGE_OFFLOAD=true`` keeps the weights in RAM until a request needs them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from app.video_models import load_manifest

_FLAG = {
    "model": "-m",
    "diffusion-model": "--diffusion-model",
    "llm": "--llm",
    "vae": "--vae",
    "clip_l": "--clip_l",
    "t5xxl": "--t5xxl",
}


def args(name: str, models_dir: Path, port: int, env: dict | None = None, manifest: dict | None = None) -> list[str]:
    """The sd-server arguments for an image model of the manifest.

    Raises:
        KeyError: for a model ``models.yaml`` does not describe.
    """
    source = os.environ if env is None else env
    recipe = (manifest or load_manifest())["image"][name]
    launch = recipe["launch"]
    line = ["--listen-ip", source.get("MEDIA_IMAGE_LISTEN_IP", "0.0.0.0"), "--listen-port", str(port)]  # noqa: S104 — inside the host's private container network
    for spec in recipe["files"]:
        line += [_FLAG[spec["role"]], str(models_dir / "sd" / spec["path"])]
    line += [
        "--steps",
        str(launch["steps"]),
        "--cfg-scale",
        str(launch["cfgScale"]),
        "-W",
        str(launch["width"]),
        "-H",
        str(launch["height"]),
        "--seed",
        "-1",
        *launch.get("flags", []),
    ]
    if (source.get("MEDIA_IMAGE_OFFLOAD") or "").lower() == "true":
        line.append("--offload-to-cpu")
    return line


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m app.sd_server <image model>", file=sys.stderr)
        return 2
    models_dir = Path(os.environ.get("MEDIA_MODELS_DIR", "/models"))
    port = int(os.environ.get("MEDIA_IMAGE_PORT", "8086"))
    line = args(argv[1], models_dir, port)
    binary = os.environ.get("MEDIA_SD_SERVER", "/usr/local/bin/sd-server")
    print(f"[sd-server] {argv[1]} on :{port}", flush=True)
    os.execv(binary, [binary, *line])  # noqa: S606 — a fixed binary with arguments from models.yaml
    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main(sys.argv))
