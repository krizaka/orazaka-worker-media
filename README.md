<!-- krizaka-header -->
<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/krizaka/.github/main/profile/assets/orazaka-dark.svg">
  <img src="https://raw.githubusercontent.com/krizaka/.github/main/profile/assets/orazaka-light.svg" alt="Orazaka" width="96">
</picture>

# Orazaka Media Worker

**The AI that never leaves home.**

Native (Metal) Python worker for image/video generation and media composition, speaking the Orazaka AMQP worker protocol.

[![CI](https://github.com/krizaka/orazaka-worker-media/actions/workflows/ci.yml/badge.svg)](https://github.com/krizaka/orazaka-worker-media/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/krizaka/orazaka-worker-media/badge)](https://scorecard.dev/viewer/?uri=github.com/krizaka/orazaka-worker-media)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Orazaka](https://img.shields.io/badge/part%20of-Orazaka-f67e23)](https://github.com/krizaka/orazaka#repositories)
[![Docs](https://img.shields.io/badge/docs-krizaka.com-f67e23)](https://www.krizaka.com/en/products/orazaka)

[Documentation](https://www.krizaka.com/en/products/orazaka) · [Website](https://www.krizaka.com) · [Krizaka on GitHub](https://github.com/krizaka)

</div>
<!-- /krizaka-header -->

**Layer:** Native worker · **Version:** `1.0.0-SNAPSHOT` · **License:** Apache-2.0 ·
part of the [Orazaka platform](https://github.com/krizaka/orazaka) by [Krizaka](https://krizaka.com)

## What it provides

A native (Metal / MLX) Python worker that drains `job.media.*` / `job.video.*` / `job.compose.*` from
RabbitMQ and reports progress on `orazaka.events`. Its bindings are declared in `worker.yaml`; the
wire protocol is documented in [WORKER_PROTOCOL.md](https://github.com/krizaka/orazaka/blob/main/docs/WORKER_PROTOCOL.md).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q test_capability_contract.py && python test_main.py
```

On the Mac, inference never runs in Docker (AGENTS.md §1): this worker runs on the host, on Metal.

### On an NVIDIA GPU (CUDA image)

`Dockerfile.cuda` builds `ghcr.io/krizaka/orazaka-worker-media:cuda-<sha>` (CI, `image.yml`): this worker plus
`sd-server` (stable-diffusion.cpp built for CUDA). It is what the hourly GPU droplet of the dev environment runs
(`scripts/gpu-dev.sh start big|small` in [krizaka/orazaka](https://github.com/krizaka/orazaka)).

| Model (catalogue name) | What | Licence | 9:16 size | Clip | Host |
|---|---|---|---|---|---|
| `wan2.2-ti2v-5b` | Wan 2.2 TI2V 5B, text- **and image-to-video** (a photo is cut to the aspect around its subject, never letterboxed; on a small card the size steps down 1280×704 → 960×544 → 832×480 and the run says which) | Apache-2.0 | 704×1280 · 1280×704 | 5 s @ 24 fps | `big` (48 GB), `small` (20 GB, CPU offload) |
| `wan2.1-t2v-1.3b` | Wan 2.1 T2V 1.3B, text-to-video | Apache-2.0 | 480×832 | 5 s @ 16 fps | `small` (20 GB), `big` |
| `wan2-2-t2v-a14b` | Wan 2.2 T2V A14B, text-to-video, **served by DigitalOcean serverless inference and routed by Orazaka** (`app/remote_video.py`; nothing runs on this host) | Apache-2.0 | 832×480 (16:9 only) | 5 s @ 16 fps, $0.60/clip | `AI_INFERENCE_KEY` |
| `flux-2-klein-4b` | FLUX.2 [klein] 4B via sd-server | Apache-2.0 | any size | image | both |

`models.yaml` says how each model is fetched (pinned revisions, checksums) and shaped; the platform's catalogue
(`orazaka_models`) says which one a job uses. A Studio passes `aspect` (`9:16` by default, `16:9`, `1:1`) and the
worker picks the size the model was trained at; `durationSeconds` becomes Wan's `4k+1` frames.

What the deployment declares (never assumed):

| Variable | Meaning |
|---|---|
| `SPRING_RABBITMQ_HOST` / `_PORT` / `_USERNAME` / `_PASSWORD` / `_SSL_ENABLED` | the broker and its login |
| `SPRING_RABBITMQ_SSL_CA_FILE` / `_SSL_SERVER_NAME` | a private CA to trust, and the name the broker's certificate is checked against when the address dialled is a tunnel end |
| `AI_INFERENCE_URL` / `AI_INFERENCE_KEY` | the serverless-inference provider (default `https://inference.do-ai.run`) and a model access key that lists `wan2-2-t2v-a14b`; only read for that model |
| `MEDIA_GPU_PROFILE` | `big` or `small`: the host's default video model (`models.yaml` profiles) |
| `MEDIA_VIDEO_DEFAULT_MODEL` | overrides that default |
| `MEDIA_ASSET_SINK_URL` | the job service; outputs are handed to it (sealed there) instead of kept on this disk |
| `ORAZAKA_SERVICE_TOKEN` | a short-lived `SERVICE` token for the hand-over and the worker registry |
| `MEDIA_ACTIVITY_FILE` | touched at each job's start and end, for the host's idle watchdog |
| `MEDIA_VIDEO_OFFLOAD` | `auto` (default), `true`, `false`: diffusers model offload |

```bash
python -m app.fetch_models big     # weights of a profile onto MEDIA_MODELS_DIR (/models), verified, once
```

## Position in the platform

| | |
|:---|:---|
| Depends on | _none — this repository is a root of the dependency graph._ |
| Used by | _no other Orazaka repository._ |
| Workspace path | `orazaka-apps/workers/orazaka-worker-media` |



## Governance

This repository follows the Orazaka governance contract — [AGENTS.md](https://github.com/krizaka/orazaka/blob/main/AGENTS.md)
in the workspace is normative; the local [AGENTS.md](AGENTS.md) only scopes it to this repository.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

<sub>A Krizaka product · [krizaka.com](https://www.krizaka.com) · Apache-2.0</sub>

## Hosted-only mode (a cheap CPU consumer)

`MEDIA_WORKER_HOSTED_ONLY=true` runs only the models a provider serves (`pipeline: remote` in `models.yaml`, e.g. the
hosted Wan 2.2): the process calls the provider's API, so it needs no GPU, no torch and no weights
(`requirements-hosted.txt`). It registers as `orazaka-worker-media-hosted`, drains `orazaka.jobs.video.hosted`
(`job.hostedvideo.*`, never matched by the GPU worker's `job.video.*`) and refuses a model that runs on a GPU by name.
Dev runs it always on inside the core container; the GPU worker stays off until a film needs it.

## Progress stages (orazaka-worker-media#64)

`job.<id>.progress` on the `orazaka.events` exchange carries, besides `jobId`, `progress` and `phase`, optional keys
(ADR-048; a consumer that ignores them is unaffected): `stage` (`COLD_START_WAKEUP` -> `STREAMING_WEIGHTS_NVME` ->
`INFERENCE_STEP` -> `SANDBOX_VERIFICATION` -> `COMPLETED`), `weightsPercent` (bytes of weights really read), `step` /
`steps`, `etaSeconds` (median of the last loads per model and hardware; a sliding mean of the latest steps) and `at`
(monotonic epoch ms). No history, no ETA; an unknown total, no percentage. `MEDIA_ETA_FILE` keeps the history across
restarts. Code: `app/stages.py`; the streamed weight load (#63) feeds `StageReporter.weights`.

## Model-cache contract (orazaka-worker-media#63)

A model is one directory with `.manifest.json` beside its files (size + SHA-256 each), written by `python -m app.fetch_models <profile>`
and checked by `python -m app.fetch_models --verify <profile> [--deep]` (sizes by default, digests with `--deep`; a partial or
corrupt file is reported, a verified model is never fetched again). `MEDIA_MODELS_STAGE_DIR` (local NVMe) stages a verified
snapshot there with parallel readers before the load and loads from it (the volume stays the source of truth; unusable stage dir,
plain load); `MEDIA_MODELS_PREFETCH=1` warms the page cache with parallel readers instead. Every Wan load logs its seconds.
Both feed `StageReporter.weights` (the `STREAMING_WEIGHTS_NVME` percentage). Code: `app/model_cache.py`.
