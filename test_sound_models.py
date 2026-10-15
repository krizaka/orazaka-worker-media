"""The sound models of models.yaml: a commercial default per kind, a non-commercial one only by name."""
import pytest

from app import sound_generators as g


def test_every_kind_default_is_commercial_and_unique():
    by_kind = {}
    for model in g.sound_models().values():
        if model.default:
            assert model.commercial, f"{model.name}: a default must be commercial ({model.licence})"
            assert model.kind not in by_kind, f"two defaults for {model.kind}"
            by_kind[model.kind] = model.name
    assert by_kind["music"] == "acestep-v15-xl-turbo"


def test_a_non_commercial_model_says_so_and_is_never_picked_unasked():
    mm = g.sound_models()["mmaudio-large-44k-v2"]
    assert not mm.commercial and "NC" in mm.licence and not mm.default
    assert g.resolve("foley", None) is None          # no commercial foley default: nothing is substituted
    assert g.resolve("foley", "mmaudio-large-44k-v2") == mm


def test_music_resolves_to_its_pinned_default():
    model = g.resolve("music", "")
    assert model.name == "acestep-v15-xl-turbo" and model.licence == "MIT"
    assert len(model.spec["revision"]) == 40


def test_unknown_or_wrong_kind_is_refused_by_name():
    with pytest.raises(g.UnknownSoundModel):
        g.resolve("music", "musicgen-small")
    with pytest.raises(g.UnknownSoundModel):
        g.resolve("music", "mmaudio-large-44k-v2")
