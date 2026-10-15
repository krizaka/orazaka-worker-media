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


def test_foley_needs_the_non_commercial_model_by_name(world):
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


def test_the_sound_binding_is_told_by_the_routing_key_and_precedes_compose():
    assert consumer._is_sound("job.compose.sound") and not consumer._is_sound("job.compose.assemble")
    assert consumer._is_compose("job.compose.sound")


def test_a_voice_over_alone_is_enough(world):
    root, _ = world
    voice = root / "u1" / "jobA" / "output" / "voice.wav"
    _ff("-f", "lavfi", "-i", "sine=frequency=500:duration=2:sample_rate=22050", str(voice))
    result, _ = consumer._sound(job({"video": "/uploads/u1/jobA/output/video.mp4", "voice": "/uploads/u1/jobA/output/voice.wav"}))
    assert result["metrics"]["tracks"] == ["voice"]
