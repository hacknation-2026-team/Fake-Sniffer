import numpy as np
import pytest
from PIL import Image
from types import SimpleNamespace
from unittest.mock import MagicMock

from feature_extractor.deepfake_extractor import (
    AUDIO_MODEL,
    IMAGE_MODEL,
    DeepfakeDetector,
    find_fake_label,
)
from feature_extractor.model_cache import HUGGINGFACE
from feature_extractor.speech_extractor import SAMPLE_RATE


def model_with(labels):
    return SimpleNamespace(config=SimpleNamespace(id2label=dict(enumerate(labels))))


def detector_with(logits, labels=("Realism", "Deepfake"), **kwargs):
    detector = DeepfakeDetector(**kwargs)
    processor = MagicMock(return_value={"pixel_values": "x"})
    detector._image = (processor, model_with(labels))
    detector._audio = (MagicMock(return_value={"input_values": "x"}), model_with(labels))
    results = iter(logits)
    detector._logits = lambda model, inputs: np.asarray(next(results), dtype=float)
    return detector


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "foto.png"
    Image.new("RGB", (32, 32), "white").save(path)
    return str(path)


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["Realism", "Deepfake"], "Deepfake"),
        (["real", "fake"], "fake"),
        (["bonafide", "spoof"], "spoof"),
        (["human", "AI-generated"], "AI-generated"),
        (["LABEL_0", "LABEL_1"], None),
        (["fake", "deepfake"], None),
    ],
)
def test_fake_label_detection(labels, expected):
    assert find_fake_label(labels) == expected


def test_preferred_fake_label_wins_and_must_exist():
    assert find_fake_label(["LABEL_0", "LABEL_1"], "LABEL_1") == "LABEL_1"
    assert find_fake_label(["LABEL_0", "LABEL_1"], "inna") is None


def test_image_probability_comes_from_softmax(photo):
    detector = detector_with([[0.0, np.log(3)]])
    result = detector.analyze_image(photo)
    assert result["scores"] == {"Realism": 0.25, "Deepfake": 0.75}
    assert result["fake_probability"] == 0.75
    assert result["label"] == "Deepfake"
    assert result["file"] == "foto.png" and result["model"] == IMAGE_MODEL


def test_image_with_unknown_labels_gives_scores_only(photo):
    detector = detector_with([[1.0, 0.0]], labels=("LABEL_0", "LABEL_1"))
    result = detector.analyze_image(photo)
    assert result["fake_probability"] is None
    assert set(result["scores"]) == {"LABEL_0", "LABEL_1"}


def test_missing_image_file_is_reported():
    with pytest.raises(FileNotFoundError):
        DeepfakeDetector().analyze_image("nie-ma.png")


def test_frames_are_aggregated(photo):
    detector = detector_with([[0.0, 0.0], [0.0, np.log(9)], [np.log(9), 0.0]])
    result = detector.analyze_images([photo, photo, photo])
    assert [f["fake_probability"] for f in result["frames"]] == [0.5, 0.9, 0.1]
    assert result["mean_fake_probability"] == 0.5
    assert result["max_fake_probability"] == 0.9


def test_no_frames_gives_empty_aggregate():
    result = DeepfakeDetector().analyze_images([])
    assert result["frames"] == [] and result["mean_fake_probability"] is None


def patch_audio(monkeypatch, seconds):
    samples = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    monkeypatch.setattr(
        "feature_extractor.deepfake_extractor.load_audio", lambda path: (samples, {"has_audio": True})
    )


def test_audio_is_cut_into_windows(monkeypatch):
    patch_audio(monkeypatch, 25)
    detector = detector_with([[0.0, 0.0]] * 3, labels=("real", "fake"))
    result = detector.analyze_audio("nagranie.mp3")
    assert [w["start_sec"] for w in result["windows"]] == [0.0, 10.0, 20.0]
    assert result["mean_fake_probability"] == 0.5
    assert result["labels"] == ["fake", "real"]
    assert result["model"] == AUDIO_MODEL and result["file"] == "nagranie.mp3"


def test_long_audio_is_sampled_evenly(monkeypatch):
    patch_audio(monkeypatch, 1000)
    detector = detector_with([[0.0, 0.0]] * 4, labels=("real", "fake"))
    result = detector.analyze_audio("dlugie.wav", window_sec=10, max_windows=4)
    starts = [w["start_sec"] for w in result["windows"]]
    assert len(starts) == 4 and starts[0] == 0.0 and starts[-1] >= 900


def test_short_audio_gets_one_window(monkeypatch):
    patch_audio(monkeypatch, 0.5)
    detector = detector_with([[0.0, 0.0]], labels=("real", "fake"))
    assert len(detector.analyze_audio("krotkie.wav")["windows"]) == 1


def test_audio_without_sound_gives_none(monkeypatch):
    monkeypatch.setattr("feature_extractor.deepfake_extractor.load_audio", lambda path: (None, {}))
    assert DeepfakeDetector().analyze_audio("film.mp4") is None


def test_audio_fake_label_can_be_set_by_hand(monkeypatch):
    patch_audio(monkeypatch, 5)
    detector = detector_with([[0.0, np.log(3)]], labels=("LABEL_0", "LABEL_1"), audio_fake_label="LABEL_1")
    assert detector.analyze_audio("a.wav")["max_fake_probability"] == 0.75


def test_models_load_with_shared_cache_directory(tmp_path, monkeypatch):
    transformers = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)
    detector = DeepfakeDetector(device="cpu", models_dir=str(tmp_path))

    detector.load_image_model()
    detector.load_image_model()
    cache = str(tmp_path / HUGGINGFACE)
    transformers.AutoImageProcessor.from_pretrained.assert_called_once_with(IMAGE_MODEL, cache_dir=cache)
    transformers.AutoModelForImageClassification.from_pretrained.assert_called_once_with(IMAGE_MODEL, cache_dir=cache)

    detector.load_audio_model()
    transformers.AutoFeatureExtractor.from_pretrained.assert_called_once_with(AUDIO_MODEL, cache_dir=cache)
    transformers.AutoModelForAudioClassification.from_pretrained.assert_called_once_with(AUDIO_MODEL, cache_dir=cache)


def test_nothing_is_loaded_on_construction():
    detector = DeepfakeDetector()
    assert detector._image is None and detector._audio is None
