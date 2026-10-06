"""Worker self-declaration and registration (ADR-038, docs/WORKER_PROTOCOL.md).

The worker reads ``worker.yaml`` at boot, binds its queues from it, and tells the
platform what it drains. The declaration is the single source: the bindings used
to be literals in ``consumer.py``, which meant adding one was a Python edit.

**Registration is never a startup dependency.** A worker that cannot reach the job
service must still consume — the registry is an availability signal, not an
authorisation. Every call here fails soft and retries in the background; getting
this backwards turns an observability feature into an outage.
"""
import base64
import hashlib
import hmac
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

#: How long between heartbeats. The platform's staleness horizon is deliberately
#: several multiples of this, so one missed beat never marks a healthy worker stale.
HEARTBEAT_SECONDS = 30

#: Backoff between failed registration attempts. Bounded, because a worker that
#: cannot register is still doing its real job and must not spin.
RETRY_SECONDS = 30


def load_declaration(path: Path | None = None) -> dict:
    """Reads ``worker.yaml`` from beside the package.

    Args:
        path: override, for tests.

    Returns:
        The declaration: name, family, bindings, version, concurrency.

    Raises:
        ValueError: if the file omits a field the platform requires, or declares a
            binding outside the ``job.`` grammar. This one IS fatal: a worker that
            cannot say what it drains cannot bind its queues either.
    """
    declaration_path = path or Path(__file__).resolve().parent.parent / "worker.yaml"
    with open(declaration_path, encoding="utf-8") as handle:
        declaration = yaml.safe_load(handle)

    for field in ("name", "family", "bindings", "version"):
        if not declaration.get(field):
            raise ValueError(f"worker.yaml is missing required field '{field}'")
    if not isinstance(declaration["bindings"], list) or not declaration["bindings"]:
        raise ValueError("worker.yaml must declare a non-empty bindings list")
    for binding in declaration["bindings"]:
        if not str(binding).startswith("job."):
            raise ValueError(f"binding must start with 'job.', was: {binding}")
    declaration.setdefault("concurrency", 1)
    return declaration


def _job_service_url() -> str:
    return os.environ.get("JOB_SERVICE_INTERNAL_URL", "http://localhost:8090")


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _service_token() -> str | None:
    """Mints the SERVICE token ``/internal/v1`` requires (ADR-035).

    Signed HS256 with the shared identity secret, with ``roles: ["SERVICE"]`` and an
    empty authority prefix — byte-for-byte the claim every Java ``ServiceTokenProvider``
    produces. The boundary is the secret, not the language: a worker holding it is as
    entitled to the internal surface as any service, which is exactly what makes a
    worker in any language a first-class participant (ADR-037 §3.4).

    A pre-minted ``ORAZAKA_SERVICE_TOKEN`` wins if the deployment supplies one. With
    neither, the worker simply never registers — and still consumes.
    """
    supplied = os.environ.get("ORAZAKA_SERVICE_TOKEN")
    if supplied:
        return supplied
    secret = os.environ.get("IDENTITY_JWT_SECRET")
    if not secret or len(secret) < 32:
        return None
    now = int(time.time())
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64(
        json.dumps(
            {"sub": "orazaka-worker-media", "roles": ["SERVICE"], "iat": now, "exp": now + 300},
            separators=(",", ":"),
        ).encode()
    )
    signing_input = f"{header}.{payload}"
    signature = _b64(hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest())
    return f"{signing_input}.{signature}"


def _post(path: str, body: dict | None) -> int:
    data = json.dumps(body).encode("utf-8") if body is not None else b""
    request = urllib.request.Request(  # noqa: S310 — deployment config, never caller-supplied
        f"{_job_service_url()}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    token = _service_token()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 — JOB_SERVICE_INTERNAL_URL is deployment config, never caller-supplied
        return response.status


def register_forever(declaration: dict) -> None:
    """Registers, then heartbeats, until the process exits.

    Never raises: every failure is logged and retried. The worker's real job is
    draining its queues, and that must not depend on this thread succeeding.
    """
    registered = False
    while True:
        try:
            if not registered:
                _post("/internal/v1/workers", declaration)
                registered = True
                print(f"[registry] registered {declaration['name']}", flush=True)
            else:
                status = _post(f"/internal/v1/workers/{declaration['name']}/heartbeat", None)
                if status == 404:
                    registered = False
        except urllib.error.HTTPError as e:
            if e.code == 404:
                registered = False
            else:
                print(f"[registry] heartbeat/registration refused ({e.code}) — still consuming", flush=True)
        except Exception as e:  # noqa: BLE001 — advisory by design; never fatal
            registered = False
            print(f"[registry] job service unreachable ({e}) — still consuming", flush=True)
        time.sleep(HEARTBEAT_SECONDS if registered else RETRY_SECONDS)


def start_background_registration(declaration: dict) -> threading.Thread:
    """Starts registration on a daemon thread so it can never block consumption.

    Args:
        declaration: the parsed ``worker.yaml``.

    Returns:
        The started thread.
    """
    thread = threading.Thread(
        target=register_forever, args=(declaration,), daemon=True, name="worker-registry"
    )
    thread.start()
    return thread
