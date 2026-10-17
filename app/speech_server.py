"""Speech on the GPU host (ADR-080): Kokoro-82M (Apache-2.0) behind the image engine's port.

The platform reaches a GPU host through one relay port for its OpenAI-compatible engines (``gpu-relay:8086``). This
server takes that port: ``POST /v1/audio/speech`` is answered here by Kokoro (a voice-over for a cinematic short), every
other path is passed through to sd-server, moved behind it (``MEDIA_IMAGE_UPSTREAM``, default ``127.0.0.1:8089``).

The request is OpenAI's: ``{"model", "input", "voice", "response_format"}``. ``model`` must be one this file serves
(``kokoro-82m``): another name is refused by name, never answered with a different voice engine.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "kokoro-82m"
REPO = "hexgrad/Kokoro-82M"
REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
SAMPLE_RATE = 24000
#: Kokoro's own voices; the default is its best-rated English one.
DEFAULT_VOICE = "af_heart"
VOICES = {"af_heart", "af_bella", "af_nicole", "af_sarah", "am_michael", "am_fenrir", "bf_emma", "bm_george"}
MAX_CHARS = 5000


class SpeechRequestError(ValueError):
    """A request this server positively checked and cannot honour."""


def parse_request(raw: bytes) -> dict:
    """The speech request, checked (pure): model, text, voice, format."""
    try:
        body = json.loads(raw or b"{}")
    except json.JSONDecodeError as bad:
        raise SpeechRequestError(f"not JSON ({bad.msg})") from None
    model = str(body.get("model") or MODEL)
    if model != MODEL:
        raise SpeechRequestError(f"model '{model}' is not served here ({MODEL})")
    text = " ".join(str(body.get("input") or "").split())
    if not text:
        raise SpeechRequestError("input is empty")
    if len(text) > MAX_CHARS:
        raise SpeechRequestError(f"input is longer than {MAX_CHARS} characters")
    voice = str(body.get("voice") or DEFAULT_VOICE)
    if voice not in VOICES:
        voice = DEFAULT_VOICE  # a catalogue voice name of another engine ("alloy") falls back to the default, said in the header
    fmt = str(body.get("response_format") or "mp3").lower()
    if fmt not in ("mp3", "wav"):
        raise SpeechRequestError("response_format must be mp3 or wav")
    speed = float(body.get("speed") or 1.0)
    return {"text": text, "voice": voice, "format": fmt, "speed": max(0.7, min(1.3, speed))}


def routes_to_speech(method: str, path: str) -> bool:
    """Whether a request is this server's (pure); everything else belongs to the image engine."""
    return method == "POST" and path.split("?")[0].rstrip("/") == "/v1/audio/speech"


_PIPELINES: dict = {}


def synthesize(text: str, voice: str, speed: float = 1.0) -> bytes:
    """Kokoro's waveform for ``text`` as 16-bit wav bytes (24 kHz mono)."""
    import numpy as np  # noqa: PLC0415
    import soundfile  # noqa: PLC0415
    from kokoro import KPipeline  # noqa: PLC0415

    lang = voice[0]
    if lang not in _PIPELINES:
        _PIPELINES[lang] = KPipeline(lang_code=lang, repo_id=REPO)
    chunks = [audio for _, _, audio in _PIPELINES[lang](text, voice=voice, speed=speed)]
    wave = np.concatenate([np.asarray(c) for c in chunks]) if chunks else np.zeros(1, dtype="float32")
    buffer = io.BytesIO()
    soundfile.write(buffer, wave, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def to_mp3(wav: bytes) -> bytes:
    done = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "mp3", "-b:a", "192k", "pipe:1"],
                          input=wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return done.stdout


#: The content types a passed-through answer may carry; anything else is sent as JSON's (never a header copied as is).
PASSED_TYPES = ("application/json", "image/png", "image/jpeg", "image/webp", "text/plain")


def safe_type(value: str | None) -> str:
    """One of PASSED_TYPES, chosen by prefix — never the upstream's string itself (pure)."""
    for known in PASSED_TYPES:
        if (value or "").lower().startswith(known):
            return known
    return "application/json"


class Handler(BaseHTTPRequestHandler):
    upstream = os.environ.get("MEDIA_IMAGE_UPSTREAM", "127.0.0.1:8089")

    def _send(self, code: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
        request = urllib.request.Request(f"http://{self.upstream}{self.path}", data=body, method=self.command, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=900) as response:  # noqa: S310 — the local image engine only
                self._send(response.status, response.read(), safe_type(response.headers.get("Content-Type")))
        except urllib.error.HTTPError as refused:
            self._send(refused.code, refused.read(), safe_type(refused.headers.get("Content-Type")))
        except OSError as down:
            self._send(502, json.dumps({"error": f"image engine unreachable: {down}"}).encode(), "application/json")

    def _speech(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            request = parse_request(self.rfile.read(length))
        except SpeechRequestError as bad:
            self._send(400, json.dumps({"error": str(bad)}).encode(), "application/json")
            return
        try:
            wav = synthesize(request["text"], request["voice"], request["speed"])
            body, kind = (to_mp3(wav), "audio/mpeg") if request["format"] == "mp3" else (wav, "audio/wav")
        except Exception as failure:  # noqa: BLE001 — reported, never a silent placeholder
            self._send(500, json.dumps({"error": f"speech failed: {failure}"}).encode(), "application/json")
            return
        # The voice is not echoed in a header (a header is never built from a request's value); the model is a constant.
        self._send(200, body, kind, {"X-Orazaka-Speech-Model": MODEL})

    def do_POST(self) -> None:  # noqa: N802 — http.server's name
        if routes_to_speech(self.command, self.path):
            self._speech()
        else:
            self._proxy()

    def do_GET(self) -> None:  # noqa: N802
        self._proxy()


def main() -> int:
    port = int(os.environ.get("MEDIA_SPEECH_PORT", "8086"))
    print(f"[speech] {MODEL} on :{port}, other paths → {Handler.upstream}", flush=True)
    ThreadingHTTPServer((os.environ.get("MEDIA_SPEECH_LISTEN_IP", "127.0.0.1"), port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
