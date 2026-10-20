"""Says a directed line with Zonos v0.1 (Zyphra, Apache-2.0), in its own interpreter (``SPEECH_PYTHON``): its torch and
espeak stack never share the worker's environment, as MMAudio's does not (``mmaudio_runner.py``).

    <speech python> -m app.zonos_runner --plan plan.json --voice reference.wav --language en-us --out line.wav [--seed 7]

``plan.json`` is ``voice_direction.plan``'s phrases (text, emotion, rate, pitch_std, pause_after). The voice is a
reference recording of a **synthetic** voice (a Kokoro voice, Apache-2.0) — never a real person's. Writes a 44.1 kHz wav.
"""
from __future__ import annotations

import argparse
import json

ZONOS_REPO = "Zyphra/Zonos-v0.1-transformer"


def main() -> None:
    import numpy as np
    import soundfile
    import torch
    from zonos.conditioning import make_cond_dict
    from zonos.model import Zonos

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    phrases = json.load(open(args.plan, encoding="utf-8"))
    model = Zonos.from_pretrained(ZONOS_REPO, device="cpu")
    data, rate = soundfile.read(args.voice, dtype="float32")
    speaker = model.make_speaker_embedding(torch.from_numpy(data if data.ndim == 1 else data.mean(1)).unsqueeze(0), rate)
    out_rate = model.autoencoder.sampling_rate
    pieces = []
    for k, p in enumerate(phrases):
        torch.manual_seed(args.seed + k)
        cond = make_cond_dict(text=p["text"], speaker=speaker, language=args.language, emotion=list(p["emotion"]),
                              speaking_rate=p["rate"], pitch_std=p["pitch_std"], device="cpu")
        audio = model.autoencoder.decode(model.generate(model.prepare_conditioning(cond))).cpu()[0].squeeze(0).numpy()
        # Zonos leaves a breath of silence at both ends: trimmed to 60 ms, so the pauses are the direction's own.
        loud = np.flatnonzero(np.abs(audio) > 0.01)
        if loud.size:
            audio = audio[max(0, loud[0] - int(0.06 * out_rate)): loud[-1] + int(0.06 * out_rate)]
        pieces += [audio, np.zeros(int(p["pause_after"] * out_rate), dtype=audio.dtype)]
    soundfile.write(args.out, np.concatenate(pieces), out_rate)


if __name__ == "__main__":
    main()
