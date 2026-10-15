"""Foley for a video with MMAudio — run by ``sound_generators.generate_foley`` in its own interpreter.

Standalone on purpose: MMAudio (github.com/hkchengrex/MMAudio) pins its own torch stack, so it lives in
``~/.orazaka/sound/venv`` and this file imports nothing from the worker. The code is MIT; the WEIGHTS it
downloads are CC-BY-NC 4.0 — non-commercial (``models.yaml``, ``sound:``). Adapted from MMAudio's demo.py.
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import torchaudio
from mmaudio.eval_utils import all_model_cfg, generate, load_video
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

logging.basicConfig(level=logging.INFO, stream=sys.stderr)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative", default="")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", default="large_44k_v2")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = all_model_cfg[args.variant]
    config.download_if_needed()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16

    net = get_my_mmaudio(config.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(config.model_path, map_location=device, weights_only=True))
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    flow = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=args.steps)
    features = FeaturesUtils(tod_vae_ckpt=config.vae_path, synchformer_ckpt=config.synchformer_ckpt,
                             enable_conditions=True, mode=config.mode,
                             bigvgan_vocoder_ckpt=config.bigvgan_16k_path, need_vae_encoder=False)
    features = features.to(device, dtype).eval()

    info = load_video(args.video.expanduser(), args.duration)
    config.seq_cfg.duration = info.duration_sec
    net.update_seq_lengths(config.seq_cfg.latent_seq_len, config.seq_cfg.clip_seq_len,
                           config.seq_cfg.sync_seq_len)
    audios = generate(info.clip_frames.unsqueeze(0), info.sync_frames.unsqueeze(0), [args.prompt],
                      negative_text=[args.negative], feature_utils=features, net=net, fm=flow, rng=rng,
                      cfg_strength=args.cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.out, audios.float().cpu()[0], config.seq_cfg.sampling_rate)


if __name__ == "__main__":
    main()
