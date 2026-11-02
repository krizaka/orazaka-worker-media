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


def test_every_pip_install_requires_hashes_and_the_lock_hashes_every_pin():
    """Sonar S8544: pip installs only from the hash-locked file, and each pin in it carries a hash."""
    root = Path(__file__).parent
    installs = [l for l in (root / "Dockerfile.cuda").read_text().splitlines() if "pip install" in l]
    assert installs, "the Dockerfile installs Python packages"
    for line in installs:
        assert "--require-hashes" in line, f"pip install without --require-hashes: {line.strip()}"
    lock = (root / "requirements-cuda.lock").read_text()
    entries = re.findall(r"^([A-Za-z0-9_.\-]+==\S+) \\\n((?:\s+--hash=sha256:[0-9a-f]{64}(?: \\)?\n)+)", lock, re.M)
    pins = re.findall(r"^[A-Za-z0-9_.\-]+==\S+ \\$", lock, re.M)
    assert pins and len(entries) == len(pins), "a pin of the lock has no sha256 hash"
    assert "torch==2.11.0+cu128" in lock and "pip==26.1.1" in lock


def test_docopt_is_the_one_package_built_from_source():
    """docopt 0.6.2 has no wheel: `--only-binary :all:` alone fails the CUDA image (krizaka-internal#2121). Only it is exempt."""
    install = next(
        l for l in (Path(__file__).with_name("Dockerfile.cuda")).read_text().splitlines() if "pip install" in l
    )
    assert "--only-binary :all:" in install
    assert re.findall(r"--no-binary (\S+)", install) == ["docopt"]
