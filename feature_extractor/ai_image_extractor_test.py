import numpy as np
import pytest
from PIL import Image
from types import SimpleNamespace
from unittest.mock import MagicMock

from feature_extractor.ai_image_extractor import AI_IMAGE_MODEL, AiImageDetector
from feature_extractor.deepfake_extractor import find_fake_label


def detector_with(logits, labels, **kwargs):
    detector = AiImageDetector(**kwargs)
    model = SimpleNamespace(config=SimpleNamespace(id2label=dict(enumerate(labels))))
    detector._image = (MagicMock(return_value={"pixel_values": "x"}), model)
    results = iter(logits)
    detector._logits = lambda model, inputs: np.asarray(next(results), dtype=float)
    return detector


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "foto.png"
    Image.new("RGB", (32, 32), "white").save(path)
    return str(path)


def test_default_model_labels_are_recognised():
    assert find_fake_label(["ai", "hum"]) == "ai"
    assert find_fake_label(["Real", "AI-generated"]) == "AI-generated"


def test_ai_probability_for_ai_hum_labels(photo):
    detector = detector_with([[np.log(3), 0.0]], ["ai", "hum"])
    result = detector.analyze_image(photo)
    assert result["model"] == AI_IMAGE_MODEL
    assert result["fake_probability"] == 0.75
    assert result["label"] == "ai"


def test_numeric_labels_need_explicit_fake_label(photo):
    labels = ["LABEL_0", "LABEL_1"]
    unknown = detector_with([[0.0, np.log(4)]], labels, model="m/x").analyze_image(photo)
    assert unknown["fake_probability"] is None
    known = detector_with([[0.0, np.log(4)]], labels, model="m/x", fake_label="LABEL_1").analyze_image(photo)
    assert known["fake_probability"] == 0.8


def test_frames_are_aggregated(photo):
    detector = detector_with([[0.0, 0.0], [0.0, np.log(9)]], ["hum", "ai"])
    result = detector.analyze_images([photo, photo])
    assert result["max_fake_probability"] == 0.9
    assert result["mean_fake_probability"] == 0.7
    assert len(result["frames"]) == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AiImageDetector().analyze_image(str(tmp_path / "brak.png"))
