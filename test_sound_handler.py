"""The sound job end to end, with the generators stubbed: ownership, refusals, outputs, the silent variant."""
import subprocess

import pytest

from app import consumer, sound, sound_generators


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    out = root / "u1" / "jobA" / "output"
    out.mkdir(parents=True)
    _ff("-f", "lavfi", "-i", "testsrc=size=320x240:rate=16:duration=3", "-pix_fmt", "yuv420p", "-c:v", "libx264",
        str(out / "video.mp4"))
    other = root / "u2" / "jobB" / "output"
    other.mkdir(parents=True)
    (other / "video.mp4").write_bytes(b"not yours")
    monkeypatch.setattr(consumer, "_upload_root", lambda: str(root))
    monkeypatch.setattr(consumer, "_store", lambda path, job, name, ctype, url: url)
    monkeypatch.setattr(consumer, "_materialise", lambda path, scratch: path)
    calls = []

    def fake_music(model, prompt, seconds, out_wav, seed=None):
        calls.append(("music", model.name, prompt, seed))
        _ff("-f", "lavfi", "-i", "sine=frequency=261:duration=12:sample_rate=48000", out_wav)
        return {"model": model.name, "seed": seed or 1, "generateSeconds": 0.1}

    def fake_foley(model, video, prompt, seconds, out_wav, seed=None):
        calls.append(("foley", model.name, prompt, seed))
        _ff("-f", "lavfi", "-i", "sine=frequency=700:duration=3:sample_rate=44100", out_wav)
        return {"model": model.name, "seed": seed or 1, "generateSeconds": 0.1}

    monkeypatch.setattr(sound_generators, "generate_music", fake_music)
    monkeypatch.setattr(sound_generators, "generate_foley", fake_foley)
    return root, calls


def job(payload, user="u1", job_id="jobS"):
    return {"jobId": job_id, "userId": user, "payload": payload}


def test_music_bed_gives_the_clip_sound_and_keeps_a_silent_variant(world):
    root, calls = world
    result, consumption = consumer._sound(job({"video": "/uploads/u1/jobA/output/video.mp4",
                                              "music": "warm lo-fi bed", "seed": "42"}))
    assert calls == [("music", "acestep-v15-xl-turbo", "warm lo-fi bed", 42)]
    out = root / "u1" / "jobS" / "output"
    assert sound.probe(str(out / "video.mp4"))["audio"] == "aac"
    assert sound.probe(str(out / "video.silent.mp4"))["audio"] is None
    assert result["url"].endswith("/video.mp4") and result["silentUrl"].endswith("/video.silent.mp4")
    assert abs(result["metrics"]["loudnessLufs"] - sound.TARGET_LUFS) <= 1.0
    assert result["metrics"]["models"] == {"music": "acestep-v15-xl-turbo"}
    assert consumption["durationSeconds"] == pytest.approx(3.0, abs=0.2)


def test_a_job_naming_non_commercial_weights_is_refused_in_production(world, monkeypatch):
    _, calls = world
    monkeypatch.delenv("MEDIA_ALLOW_NON_COMMERCIAL_WEIGHTS", raising=False)
    with pytest.raises(consumer.InvalidJobPayload, match="non-commercial"):
        consumer._sound(job({"video": "/uploads/u1/jobA/output/video.mp4", "foley": "tea pouring",
                             "foleyModel": "mmaudio-large-44k-v2"}))
    assert calls == []          # refused before any download or generation


def test_foley_needs_the_non_commercial_model_by_name(world, monkeypatch):
    monkeypatch.setenv("MEDIA_ALLOW_NON_COMMERCIAL_WEIGHTS", "true")
    _, calls = world
    with pytest.raises(consumer.InvalidJobPayload, match="foleyModel"):
        consumer._sound(job({"video": "/uploads/u1/jobA/output/video.mp4", "foley": "tea pouring"}))
    assert calls == []
    consumer._sound(job({"video": "/uploads/u1/jobA/output/video.mp4", "foley": "tea pouring",
                         "foleyModel": "mmaudio-large-44k-v2"}))
    assert calls[0][:2] == ("foley", "mmaudio-large-44k-v2")


def test_another_actors_output_is_not_found(world):
    with pytest.raises(consumer.InvalidJobPayload, match="not found"):
        consumer._sound(job({"video": "/uploads/u2/jobB/output/video.mp4", "music": "x"}))
    with pytest.raises(consumer.InvalidJobPayload, match="not found"):
        consumer._sound(job({"video": "/uploads/u1/../u2/jobB/output/video.mp4", "music": "x"}))


def test_nothing_to_add_is_refused_before_any_model_runs(world):
    _, calls = world
    with pytest.raises(consumer.InvalidJobPayload):
        consumer._sound(job({"video": "/uploads/u1/jobA/output/video.mp4"}))
    assert calls == []


def test_unknown_model_is_refused_by_name(world):
    with pytest.raises(consumer.InvalidJobPayload, match="musicgen"):
        consumer._sound(job({"video": "/uploads/u1/jobA/output/video.mp4", "music": "x", "musicModel": "musicgen"}))


def test_the_sound_binding_is_told_by_the_routing_key():
    assert consumer._is_sound("job.sound.generate") and not consumer._is_sound("job.compose.assemble")
    assert not consumer._is_compose("job.sound.generate")


def test_a_voice_over_alone_is_enough(world):
    root, _ = world
    voice = root / "u1" / "jobA" / "output" / "voice.wav"
    _ff("-f", "lavfi", "-i", "sine=frequency=500:duration=2:sample_rate=22050", str(voice))
    result, _ = consumer._sound(job({"video": "/uploads/u1/jobA/output/video.mp4", "voice": "/uploads/u1/jobA/output/voice.wav"}))
    assert result["metrics"]["tracks"] == ["voice"]


def test_the_asset_url_of_an_earlier_step_resolves_to_the_actors_own_output(world):
    root, _ = world
    result, _ = consumer._sound(job({"video": "/api/v1/assets/jobA/video.mp4",
                                     "music": "x"}))
    assert result["url"].endswith("/video.mp4")


def test_on_a_host_without_the_platform_disk_the_clip_is_read_back_from_the_job_service(world, monkeypatch, tmp_path):
    root, _ = world
    src = root / "u1" / "jobA" / "output" / "video.mp4"
    seen = {}

    def fake_fetch(job_id, filename, user_id, destination, env=None):
        seen["args"] = (job_id, filename, user_id)
        return str(src)

    monkeypatch.setattr(consumer.asset_sink, "sink_url", lambda env=None: "http://core")
    monkeypatch.setattr(consumer.asset_sink, "fetch", fake_fetch)
    consumer._sound(job({"video": "/api/v1/assets/jobA/video.mp4", "music": "x"}))
    assert seen["args"] == ("jobA", "video.mp4", "u1")


def test_another_actors_asset_url_is_not_found(world):
    with pytest.raises(consumer.InvalidJobPayload, match="not found"):
        consumer._sound(job({"video": "/api/v1/assets/jobB/video.mp4", "music": "x"}))


def test_asset_url_parsing_accepts_two_words_and_nothing_else():
    from app import asset_sink
    assert asset_sink.parse_asset_url("/api/v1/assets/abc-1/video.mp4") == ("abc-1", "video.mp4")
    for bad in ("/api/v1/assets/../x/video.mp4", "/api/v1/assets/a/b/c.mp4", "/uploads/u/j/output/v.mp4", ""):
        assert asset_sink.parse_asset_url(bad) is None


def test_the_sound_queue_is_consumed_only_where_the_host_says_so():
    assert consumer.sound_enabled({"MEDIA_SOUND_ENABLED": "true"})
    assert not consumer.sound_enabled({})
    assert not consumer.sound_enabled({"MEDIA_SOUND_ENABLED": "false"})
    assert consumer.SOUND_QUEUE != consumer.VIDEO_QUEUE
    shared, sound = consumer.split_bindings(["job.video.*", "job.compose.*", "job.sound.*"], False)
    assert shared == ["job.video.*", "job.compose.*"] and sound == []   # declared, but neither bound nor registered here
    shared, sound = consumer.split_bindings(["job.video.*", "job.compose.*", "job.sound.*"], True)
    assert shared == ["job.video.*", "job.compose.*"] and sound == ["job.sound.*"]
    assert "job.sound.*" not in consumer.BINDINGS


def test_a_sound_only_process_does_not_drain_the_shared_queue():
    assert consumer.sound_only({"MEDIA_WORKER_ONLY_SOUND": "true"})
    assert not consumer.sound_only({}) and not consumer.sound_only({"MEDIA_WORKER_ONLY_SOUND": "no"})


def test_a_published_clip_is_read_only_from_the_platforms_cdn(monkeypatch, tmp_path):
    seen = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            return b"mp4-bytes"

    monkeypatch.setattr(consumer.urllib.request, "urlopen", lambda url, timeout=0: seen.append(url) or Reply())
    ok = consumer._fetch_published("https://krizaka-assets.b-cdn.net/site/products/orazaka/gallery-video/tea-pour.c41.mp4", str(tmp_path))
    assert ok and open(ok, "rb").read() == b"mp4-bytes"
    for bad in ("http://krizaka-assets.b-cdn.net/x.mp4", "https://evil.example/x.mp4",
                "https://krizaka-assets.b-cdn.net@evil.example/x.mp4", "https://u:p@krizaka-assets.b-cdn.net/x.mp4",
                "file:///etc/passwd"):
        assert consumer._fetch_published(bad, str(tmp_path)) is None
    assert len(seen) == 1
