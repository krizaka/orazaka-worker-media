"""The worker asserts it handles every input its capabilities declare (ADR-069 §6, audit #43).

`compose` declared seven inputs and implemented two. `captions` and `brandKit` were promised by a
label nobody had checked against the code, `clip` and `bRoll` by a template, and `realestate-reels`
believed all of it: it generated a b-roll clip, passed it as `bRoll`, and the worker read `photos`
and `audio` and nothing else — 3 600 credits of a run's reserve for an artefact no reel ever
contained (audit #35).

**Why this is a contract test and not a rule.** The declaration is a row in Postgres; the executor
is Python in another process. No static rule in the Java build can see both, and one that tried
would either read the Python by regex from a module that does not import it, or flag the wrong
scope. What CAN see both is the worker itself — it knows which routing keys it drains, and it can
read the same seed the platform does. That is the same judgement the metering-quantity rule and the
billing divergence contracts took: where no honest static anchor exists, a contract through the real
thing beats a rule over the wrong one.

**The direction that is asserted.** Every input a capability DECLARES must be read by the handler
that serves it — an undeclared extra in the code is a private detail, but a declared input nobody
reads is either work bought and discarded or a promise a blueprint author will believe.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

def _workspace_root() -> Path | None:
    """The Orazaka workspace (orazaka.workspace.json) this worker is cloned in, if any."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "orazaka.workspace.json").is_file():
            return parent
    return None


#: The capability seed is owned by the job service's repository; the workspace holds both. A
#: standalone clone of this worker cannot see it, so the contract is skipped there — never passed.
WORKSPACE_ROOT = _workspace_root()
if WORKSPACE_ROOT is None:
    pytest.skip(
        "the capability seed lives in orazaka-job-service: run inside the Orazaka workspace",
        allow_module_level=True,
    )
SEED = (
    WORKSPACE_ROOT
    / "orazaka-apps"
    / "services"
    / "orazaka-job-service"
    / "infra"
    / "initdb"
    / "30-jobs-config.sql"
)
WORKER_YAML = Path(__file__).resolve().parent / "worker.yaml"

#: Which module serves which routing-key prefix. The consumer dispatches on the prefix, so this is
#: the same discriminant it uses — not a second mapping keyed on a capability name.
HANDLERS = {
    "job.compose.": ["app/consumer.py", "app/composer.py"],
    # main.py is where a video job's payload is actually read — the first version of this contract
    # left it out and reported `durationSeconds` and `image` unread, which was the test being wrong
    # rather than the worker. A contract that names the wrong module is the transcription problem
    # again, one level up.
    "job.video.": ["app/consumer.py", "app/main.py", "app/generator.py", "app/mlx_generator.py"],
}

#: A seeded capability row: key, handler, routing key … then its two contract schemas.
CAPABILITY_ROW = re.compile(
    r"\('(?P<key>orazaka\.[a-z0-9.]+)',\s*'(?P<handler>[a-z]+\.[a-z]+)',"
    r"\s*'(?P<routing>job\.[a-z0-9.]+)'.*?"
    r"\$\$(?P<input>\{.*?})\$\$::jsonb",
    re.DOTALL,
)


def _bindings() -> list[str]:
    """The routing keys this worker drains, from its own declaration."""
    declared = yaml.safe_load(WORKER_YAML.read_text())
    return [binding.replace("*", "") for binding in declared["bindings"]]


def _declared_capabilities() -> list[tuple[str, str, dict]]:
    """Every seeded capability this worker drains: (key, routing key, input schema)."""
    sql = SEED.read_text()
    start = sql.index("INSERT INTO orazaka_capabilities")
    found = []
    for match in CAPABILITY_ROW.finditer(sql[start:]):
        routing = match.group("routing")
        if any(routing.startswith(prefix) for prefix in _bindings()):
            found.append((match.group("key"), routing, json.loads(match.group("input"))))
    return found


def _source_for(routing_key: str) -> str:
    """The handler source that serves this routing key."""
    for prefix, modules in HANDLERS.items():
        if routing_key.startswith(prefix):
            return "\n".join(
                (Path(__file__).resolve().parent / module).read_text() for module in modules
            )
    raise AssertionError(f"this worker drains {routing_key} and no module is named for it")


def _reads(source: str, name: str) -> bool:
    """Whether the handler actually READS this input, rather than merely mentioning it.

    Anchored on payload access — `payload.get("x")`, `payload["x"]`, `.get("x", …)` — because the
    first version of this contract matched the name anywhere in the file and a docstring saying
    "captions are not implemented" counted as an implementation. It passed a planted declaration,
    which is exactly the failure mode a plant exists to find.
    """
    access = (
        rf'\.get\(\s*["\']{re.escape(name)}["\']',
        rf'\[\s*["\']{re.escape(name)}["\']\s*\]',
    )
    return any(re.search(pattern, source) for pattern in access)


def test_the_worker_drains_capabilities_that_declare_a_contract():
    """A population check: a contract test over nothing reports green (GOV-006's reasoning)."""
    capabilities = _declared_capabilities()
    assert capabilities, "no seeded capability routes to this worker — the contract asserts nothing"
    for key, _routing, schema in capabilities:
        assert schema.get("properties"), f"{key} declares no inputs at all"


@pytest.mark.parametrize("key,routing,schema", _declared_capabilities())
def test_every_declared_input_is_read_by_the_handler(key, routing, schema):
    """Every input the capability promises is one the executor actually reads."""
    source = _source_for(routing)
    unread = [
        name
        for name in schema.get("properties", {})
        # `model` is resolved by the platform before dispatch and reaches the engine through the
        # envelope rather than the payload, so it is read by the generator's model resolution and
        # not by a payload lookup.
        if name != "model" and not _reads(source, name)
    ]
    assert not unread, (
        f"{key} declares {unread}, and the code that serves {routing} never reads them. "
        "An input nobody implements is a promise a blueprint author will believe — fix the "
        "declaration or fix the executor, and say which (ADR-069 §6)."
    )
