# Orazaka Media Worker

> Native (Metal) Python worker for image/video generation and media composition, speaking the Orazaka AMQP worker protocol.

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

Inference never runs in Docker (AGENTS.md §1): this worker runs on the host.

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
