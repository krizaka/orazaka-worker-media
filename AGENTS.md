# orazaka-worker-media — Governance scope (agent-neutral)

> This repository is one component of the **Orazaka platform**. The normative contract is
> [`AGENTS.md`](https://github.com/krizaka/orazaka/blob/main/AGENTS.md) at the root of the Orazaka workspace
> ([`krizaka/orazaka`](https://github.com/krizaka/orazaka)), together with its `.agent/rules/*`. When this repository
> is cloned inside the workspace (`orazaka-apps/workers/orazaka-worker-media`), that contract is loaded first and applies
> without exception. **No rule lives here** — this file only scopes it.

## Scope of this repository

- **Role:** Native (Metal) Python worker for image/video generation and media composition, speaking the Orazaka AMQP worker protocol.
- **Layer:** Native worker
- **Depends on:** nothing — never on another repository's Tier-3 implementation (AGENTS.md §2, [SEAM-002]).
- **Workspace path:** `orazaka-apps/workers/orazaka-worker-media`

## Definition of done

1. `python -m pytest -q test_capability_contract.py` and `python test_main.py` are green.
2. Bindings declared in `worker.yaml` match the capability registry (EXEC-002, checked in the workspace).
