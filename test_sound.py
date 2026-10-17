"""The sound step: mux, loudness, ducking, the silent variant kept (orazaka-packs#25). ffmpeg only, no model."""
import subprocess

import pytest

from app import sound
from app.sound import SoundError, Tracks


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A 4 s silent h264 clip, the shape of what Orazaka generates."""
    d = tmp_path_factory.mktemp("clip")
    path = str(d / "in.mp4")
    _ff("-f", "lavfi", "-i", "testsrc=size=320x240:rate=16:duration=4", "-pix_fmt", "yuv420p", "-c:v", "libx264", path)
    return path


def _tone(d, name, freq, seconds, volume=0.5):
    path = str(d / f"{name}.wav")
    _ff("-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}:sample_rate=44100", "-af",
        f"volume={volume}", path)
    return path


def test_the_picture_is_copied_not_reencoded_and_aac_is_muxed(clip, tmp_path):
    out = str(tmp_path / "out.mp4")
    result = sound.add_sound(clip, Tracks(music=_tone(tmp_path, "m", 220, 6)), out)
    info = sound.probe(out)
    assert info["video"] == "h264" and info["audio"] == "aac"
    assert (info["width"], info["height"]) == (320, 240)
    assert abs(info["durationSeconds"] - 4.0) < 0.15
    assert result["tracks"] == ["music"]
    # copied bit for bit: the video stream's packets are those of the source
    def pkts(p):
        return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "packet=size", "-of", "csv=p=0", p], capture_output=True, text=True).stdout

    assert pkts(out) == pkts(clip)


@pytest.mark.parametrize("level", [0.05, 0.9])
def test_loudness_lands_on_minus_14_lufs_whatever_the_input_level(clip, tmp_path, level):
    out = str(tmp_path / "out.mp4")
    result = sound.add_sound(clip, Tracks(foley=_tone(tmp_path, "f", 440, 4, level)), out)
    assert abs(result["loudnessLufs"] - sound.TARGET_LUFS) <= 1.0
    assert abs(sound.integrated_loudness(out) - sound.TARGET_LUFS) <= 1.0


def test_the_silent_variant_is_kept_beside_the_clip_with_no_audio(clip, tmp_path):
    out = str(tmp_path / "video.mp4")
    result = sound.add_sound(clip, Tracks(music=_tone(tmp_path, "m", 330, 5)), out)
    silent = result["silentPath"]
    assert silent == str(tmp_path / "video.silent.mp4")
    info = sound.probe(silent)
    assert info["audio"] is None and info["video"] == "h264"
    assert abs(info["durationSeconds"] - 4.0) < 0.15


def test_a_source_that_already_had_sound_is_replaced_and_its_silent_variant_is_silent(clip, tmp_path):
    first = str(tmp_path / "a.mp4")
    sound.add_sound(clip, Tracks(music=_tone(tmp_path, "m", 330, 5)), first)
    second = str(tmp_path / "b.mp4")
    result = sound.add_sound(first, Tracks(foley=_tone(tmp_path, "f", 600, 4)), second)
    assert result["sourceHadAudio"] is True
    assert sound.probe(result["silentPath"])["audio"] is None
    assert sound.probe(second)["audio"] == "aac"


def test_the_voice_ducks_the_bed(clip, tmp_path):
    """Music alone is louder in the first seconds than the same music while a voice speaks over it."""
    music = _tone(tmp_path, "m", 200, 4, 0.5)
    voice = str(tmp_path / "v.wav")
    # a voice that speaks from 2 s to 4 s, silent before
    _ff("-f", "lavfi", "-i", "sine=frequency=1000:duration=2:sample_rate=44100", "-af",
        "volume=0.6,adelay=2000|2000,apad=whole_dur=4", voice)
    graph, order = sound.mix_graph(Tracks(music=music, voice=voice), 4.0)
    assert "sidechaincompress" in graph and order == ["music", "voice"]
    mixed = str(tmp_path / "mix.wav")
    _ff("-i", clip, "-i", music, "-i", voice, "-filter_complex", graph, "-map", "[mix]", mixed)

    def band_db(start, end, freq):
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-ss", str(start), "-to", str(end), "-i", mixed, "-af",
             f"bandpass=f={freq}:width_type=h:w=60,volumedetect", "-f", "null", "-"],
            capture_output=True, text=True).stderr
        return float(out.split("mean_volume:")[1].split("dB")[0])

    before_voice = band_db(0.8, 1.8, 200)   # the bed, alone
    under_voice = band_db(2.6, 3.8, 200)    # the bed, voice speaking
    assert before_voice - under_voice >= 8


def test_at_least_one_track_is_required(clip, tmp_path):
    with pytest.raises(SoundError):
        sound.add_sound(clip, Tracks(), str(tmp_path / "o.mp4"))


def test_an_unreadable_track_or_video_is_refused_by_name(clip, tmp_path):
    with pytest.raises(SoundError, match="music"):
        sound.add_sound(clip, Tracks(music=str(tmp_path / "nope.wav")), str(tmp_path / "o.mp4"))
    with pytest.raises(SoundError, match="video"):
        sound.add_sound(str(tmp_path / "nope.mp4"), Tracks(music=_tone(tmp_path, "m", 220, 2)), str(tmp_path / "o.mp4"))


def test_the_output_never_overwrites_the_source(clip, tmp_path):
    with pytest.raises(SoundError):
        sound.add_sound(clip, Tracks(music=_tone(tmp_path, "m", 220, 2)), clip)


def test_a_track_shorter_than_the_clip_is_padded_not_cutting_the_picture(clip, tmp_path):
    out = str(tmp_path / "o.mp4")
    sound.add_sound(clip, Tracks(foley=_tone(tmp_path, "f", 500, 1)), out)
    assert abs(sound.probe(out)["durationSeconds"] - 4.0) < 0.15


def test_every_track_is_conformed_to_float_planar_so_the_sidechain_never_renegotiates():
    # ffmpeg 4.4 (the GPU pod's) failed the foley + music + voice mix with "Error reinitializing filters" when a voice
    # decoded as s16 met the float bed in sidechaincompress; every track now enters the graph as fltp 48 kHz stereo.
    from app import sound

    graph, order = sound.mix_graph(sound.Tracks(foley="/f.wav", music="/m.wav", voice="/v.mp3"), 10.0)
    assert order == ["foley", "music", "voice"]
    assert graph.count("aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo") == 3
