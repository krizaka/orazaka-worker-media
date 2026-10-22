"""The model-cache contract (orazaka-worker-media#63): layout, digests, verify, NVMe staging, streamed read."""

import os
import tempfile
import unittest
from pathlib import Path

from app import model_cache


def make_snapshot(root: Path, files: dict) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = make_snapshot(self.tmp / "src", {"a/one.safetensors": b"1" * 1000, "b.json": b"{}"})

    def test_the_manifest_records_size_and_sha256_per_file(self):
        manifest = model_cache.write_manifest(self.src)
        self.assertEqual({"a/one.safetensors", "b.json"}, set(manifest["files"]))
        self.assertEqual(1000, manifest["files"]["a/one.safetensors"]["bytes"])
        self.assertEqual(64, len(manifest["files"]["b.json"]["sha256"]))
        self.assertTrue((self.src / model_cache.MANIFEST).exists())

    def test_a_clean_snapshot_verifies(self):
        model_cache.write_manifest(self.src)
        self.assertEqual([], model_cache.verify(self.src))
        self.assertEqual([], model_cache.verify(self.src, deep=True))

    def test_a_partial_file_is_detected_by_size(self):
        model_cache.write_manifest(self.src)
        (self.src / "a/one.safetensors").write_bytes(b"1" * 10)
        self.assertEqual(["a/one.safetensors: size 10, expected 1000"], model_cache.verify(self.src))

    def test_a_corrupt_file_of_the_right_size_is_detected_only_by_the_deep_check(self):
        model_cache.write_manifest(self.src)
        (self.src / "a/one.safetensors").write_bytes(b"2" * 1000)
        self.assertEqual([], model_cache.verify(self.src))
        self.assertEqual(1, len(model_cache.verify(self.src, deep=True)))

    def test_a_missing_file_is_detected(self):
        model_cache.write_manifest(self.src)
        (self.src / "b.json").unlink()
        self.assertEqual(["b.json: missing"], model_cache.verify(self.src))

    def test_no_manifest_is_not_verified(self):
        self.assertEqual(["no manifest"], model_cache.verify(self.src))


class TestStaging(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = make_snapshot(self.tmp / "src", {"x/big.safetensors": os.urandom(300_000), "y.json": b"{}"})
        model_cache.write_manifest(self.src)

    def test_staging_copies_everything_and_reports_the_bytes_read(self):
        seen = []
        out = model_cache.stage(self.src, self.tmp / "nvme", workers=3, chunk=64_000, progress=lambda r, t: seen.append((r, t)))
        self.assertEqual([], model_cache.verify(out, deep=True))
        self.assertEqual(300_002, seen[-1][0])
        self.assertTrue(all(total == 300_002 for _, total in seen))
        reads = [r for r, _ in seen]
        self.assertEqual(sorted(reads), reads)  # the percentage never goes down

    def test_a_verified_stage_is_not_copied_again(self):
        model_cache.stage(self.src, self.tmp / "nvme")
        copies = []
        out = model_cache.stage(self.src, self.tmp / "nvme", progress=lambda r, t: copies.append(r))
        self.assertEqual([300_002], copies)  # reported as complete at once, nothing read
        self.assertEqual([], model_cache.verify(out))

    def test_a_partial_stage_is_repaired(self):
        out = model_cache.stage(self.src, self.tmp / "nvme")
        (out / "x/big.safetensors").write_bytes(b"cut")
        out = model_cache.stage(self.src, self.tmp / "nvme")
        self.assertEqual([], model_cache.verify(out, deep=True))

    def test_without_a_stage_dir_the_source_is_used(self):
        self.assertEqual(self.src, model_cache.resolve(self.src, None))
        self.assertEqual(self.src, model_cache.resolve(self.src, ""))

    def test_a_stage_that_cannot_be_written_falls_back_to_the_source(self):
        blocked = self.tmp / "file"
        blocked.write_text("not a directory")
        self.assertEqual(self.src, model_cache.resolve(self.src, blocked / "nvme"))

    def test_resolve_returns_the_staged_copy(self):
        staged = model_cache.resolve(self.src, self.tmp / "nvme")
        self.assertNotEqual(self.src, staged)
        self.assertEqual([], model_cache.verify(staged))


class TestPrefetch(unittest.TestCase):
    def test_prefetch_reads_every_byte_and_reports_progress(self):
        tmp = Path(tempfile.mkdtemp())
        src = make_snapshot(tmp / "s", {"a.bin": os.urandom(200_000), "b.bin": os.urandom(100_000)})
        seen = []
        model_cache.prefetch(src, workers=4, chunk=50_000, progress=lambda r, t: seen.append(r))
        self.assertEqual(300_000, seen[-1])


class TestTimedLoad(unittest.TestCase):
    def test_the_load_is_logged_with_bytes_and_rate_and_recorded(self):
        calls = []
        clock = iter([100.0, 112.0])
        result = model_cache.timed_load("wan", 34e9, lambda: "pipe", clock=lambda: next(clock), log=calls.append)
        self.assertEqual(("pipe", 12.0), result)
        self.assertIn("wan", calls[0])
        self.assertIn("2.8 GB/s", calls[0])


class TestLocalSnapshot(unittest.TestCase):
    """The worker fetches only the files a model needs (allow patterns), so Hugging Face's own `snapshot_download(local_files_only=True)`
    calls the snapshot incomplete and raises: found on the first measurement pod (#63), where it also silently turned NVMe staging off.
    The snapshot is found by its layout instead."""

    def test_the_snapshot_directory_is_found_by_layout_even_when_only_some_files_were_fetched(self):
        cache = Path(tempfile.mkdtemp())
        snap = make_snapshot(cache / "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers" / "snapshots" / "abc123", {"transformer/x.safetensors": b"1"})
        self.assertEqual(snap, model_cache.local_snapshot("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "abc123", str(cache)))

    def test_the_cache_defaults_to_HF_HUB_CACHE(self):
        cache = Path(tempfile.mkdtemp())
        snap = make_snapshot(cache / "models--o--n" / "snapshots" / "r1", {"f": b"1"})
        previous = os.environ.get("HF_HUB_CACHE")
        os.environ["HF_HUB_CACHE"] = str(cache)
        try:
            self.assertEqual(snap, model_cache.local_snapshot("o/n", "r1"))
        finally:
            os.environ.pop("HF_HUB_CACHE") if previous is None else os.environ.update(HF_HUB_CACHE=previous)

    def test_a_snapshot_that_is_not_there_says_so(self):
        with self.assertRaises(FileNotFoundError):
            model_cache.local_snapshot("o/n", "missing", str(tempfile.mkdtemp()))


if __name__ == "__main__":
    unittest.main()
