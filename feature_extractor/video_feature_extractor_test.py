import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock

from video_feature_extractor import VideoFeatureExtractor, shared_scene_ratio
from visual_feature_extractor import ImageHasher


@pytest.fixture
def frame_files(tmp_path):
    paths = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        path = tmp_path / f"kf_{seed}.png"
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture
def keyframes(frame_files):
    mock = MagicMock()
    mock.extract.return_value = [
        {"keyframe_id": i + 1, "file_path": p, "timestamp_sec": float(i)} for i, p in enumerate(frame_files)
    ]
    return mock


def test_hashes_only_when_optional_extractors_missing(keyframes):
    result = VideoFeatureExtractor(keyframes, ImageHasher()).analyze("ad.mp4", "out")

    keyframes.extract.assert_called_once_with("ad.mp4", "out")
    assert len(result["dhashes"]) == 2
    assert result["ocr_text"] == ""
    assert result["face_embeddings"] == []
    assert result["visual_embedding"] is None


def test_ocr_lines_are_deduplicated_across_frames(keyframes):
    ocr = MagicMock()
    ocr.extract_text.side_effect = [["Zarób 5000 zł", "Kliknij TERAZ"], ["zarób  5000 zł", "Tylko dziś"]]

    result = VideoFeatureExtractor(keyframes, ImageHasher(), ocr=ocr).analyze("ad.mp4", "out")

    assert result["ocr_text"] == "Zarób 5000 zł\nKliknij TERAZ\nTylko dziś"
    assert ocr.extract_text.call_args[1] == {"join_lines": False}


def test_face_embeddings_are_collected_from_all_frames(keyframes):
    faces = MagicMock()
    faces.extract_faces.side_effect = [[{"embedding": [1.0, 0.0]}], [{"embedding": [0.0, 1.0]}, {"embedding": [0.5, 0.5]}]]

    result = VideoFeatureExtractor(keyframes, ImageHasher(), faces=faces).analyze("ad.mp4", "out")

    assert result["face_embeddings"] == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    assert len(result["keyframes"][1]["faces"]) == 2


def test_visual_embedding_is_normalized_mean(keyframes):
    embedder = MagicMock()
    embedder.extract_embeddings.return_value = [[1.0, 0.0], [0.0, 1.0]]

    result = VideoFeatureExtractor(keyframes, ImageHasher(), embedder=embedder).analyze("ad.mp4", "out")

    assert result["visual_embedding"] == pytest.approx([0.7071, 0.7071], abs=1e-3)


def test_no_keyframes_gives_empty_result():
    empty = MagicMock()
    empty.extract.return_value = []
    embedder = MagicMock()

    result = VideoFeatureExtractor(empty, ImageHasher(), embedder=embedder).analyze("ad.mp4", "out")

    assert result["dhashes"] == [] and result["visual_embedding"] is None
    embedder.extract_embeddings.assert_not_called()


# --- porównanie filmów ---

A, B, C = "0" * 16, "f" * 16, "0f" * 8  # trzy dHashe daleko od siebie


def test_shared_scene_ratio_detects_reedit_with_extra_scenes():
    original = [A, B]
    reedit = [C, A, B, C]  # te same dwa ujęcia + wstawki
    assert shared_scene_ratio(original, reedit) == 1.0


def test_shared_scene_ratio_partial_and_none():
    assert shared_scene_ratio([A, B], [A, C]) == 0.5
    assert shared_scene_ratio([A], [B]) == 0.0


def test_shared_scene_ratio_tolerates_small_hash_differences():
    nearly_a = "000000000000000" + "7"  # 3 bity różnicy
    assert shared_scene_ratio([A], [nearly_a], max_distance=8) == 1.0
    assert shared_scene_ratio([A], [nearly_a], max_distance=2) == 0.0


def test_shared_scene_ratio_empty_inputs():
    assert shared_scene_ratio([], [A]) == 0.0
    assert shared_scene_ratio([A], []) == 0.0
