"""Every base image of the Dockerfile is pinned by digest (OpenSSF Scorecard, Pinned-Dependencies)."""
import re
from pathlib import Path


def test_every_from_is_pinned_by_digest():
    lines = Path(__file__).with_name("Dockerfile.cuda").read_text().splitlines()
    stages, froms = set(), []
    for line in lines:
        m = re.match(r"\s*FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$", line, re.I)
        if not m:
            continue
        if m.group(2):
            stages.add(m.group(2).lower())
        if m.group(1).lower() not in stages and m.group(1) != "scratch":
            froms.append(m.group(1))
    assert froms, "the Dockerfile has base images"
    for image in froms:
        assert re.search(r"@sha256:[0-9a-f]{64}$", image), f"{image} is not pinned by digest"
