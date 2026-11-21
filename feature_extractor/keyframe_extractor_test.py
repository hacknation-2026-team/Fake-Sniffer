import cv2
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from keyframe_extractor import LightweightKeyframeExtractor


def noise_frame(seed: int, size=(96, 64)) -> np.ndarray:
    # szum ma dużą wariancję Laplasjanu (ostry), a każdy seed daje inny dHash
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)


def write_video(path: Path, scenes, fps=10, seconds_per_scene=2) -> str:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (96, 64))
    for frame in scenes:
        for _ in range(fps * seconds_per_scene):
            writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture
def three_scene_video(tmp_path: Path) -> str:
    return write_video(tmp_path / "ad.avi", [noise_frame(1), noise_frame(2), noise_frame(3)])


def test_one_keyframe_per_distinct_scene(three_scene_video, tmp_path):
    extractor = LightweightKeyframeExtractor(sample_interval_sec=0.5, max_keyframes=8)
    result = extractor.extract(three_scene_video, str(tmp_path / "out"))

    assert len(result) == 3  # 12 próbek, ale tylko 3 różne ujęcia
    assert [r["timestamp_sec"] for r in result] == [0.0, 2.0, 4.0]
    assert all(Path(r["file_path"]).exists() for r in result)
    assert len({r["dhash"] for r in result}) == 3


def test_blurry_frames_are_rejected(tmp_path):
    flat = np.full((64, 96, 3), 128, dtype=np.uint8)  # zerowa ostrość
    video = write_video(tmp_path / "blur.avi", [flat, noise_frame(7)])

    result = LightweightKeyframeExtractor(min_sharpness=60.0).extract(video, str(tmp_path / "out"))

    assert len(result) == 1
    assert result[0]["timestamp_sec"] == 2.0


def test_max_keyframes_prefers_frames_with_faces(three_scene_video, tmp_path):
    extractor = LightweightKeyframeExtractor(max_keyframes=1)
    # twarz "widać" tylko w drugim ujęciu (detektor jest wołany raz na kandydata)
    extractor.face_cascade = MagicMock()
    extractor.face_cascade.detectMultiScale.side_effect = [[], [(5, 5, 40, 40)], []]

    result = extractor.extract(three_scene_video, str(tmp_path / "out"))

    assert len(result) == 1
    assert result[0]["has_face"] is True
    assert result[0]["faces_boxes"] == [{"x": 5, "y": 5, "w": 40, "h": 40}]
    assert result[0]["timestamp_sec"] == 2.0


def test_results_are_chronological_with_sequential_ids(three_scene_video, tmp_path):
    result = LightweightKeyframeExtractor(max_keyframes=2).extract(three_scene_video, str(tmp_path / "out"))
    assert [r["keyframe_id"] for r in result] == [1, 2]
    assert result[0]["timestamp_sec"] < result[1]["timestamp_sec"]


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="brak.mp4"):
        LightweightKeyframeExtractor().extract("brak.mp4", str(tmp_path))


def test_unreadable_video_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"to nie jest film")
    with pytest.raises(RuntimeError):
        LightweightKeyframeExtractor().extract(str(bad), str(tmp_path / "out"))
