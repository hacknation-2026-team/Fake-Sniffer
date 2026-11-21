import cv2
import json
import numpy as np
import pytest
import wave
from pathlib import Path

from feature_extractor.spectrogram_extractor import SAMPLE_RATE, SpectrogramExtractor


def tone(seconds: float, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds: float, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> str:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return str(path)


@pytest.fixture
def tone_wav(tmp_path: Path) -> str:
    return write_wav(tmp_path / "tone.wav", tone(2.0))


@pytest.fixture
def silent_video(tmp_path: Path) -> str:
    path = tmp_path / "bez_dzwieku.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64))
    for _ in range(10):
        writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
    writer.release()
    return str(path)


def test_compute_mel_defaults():
    spectrogram = SpectrogramExtractor().compute(tone(1.0))

    assert spectrogram.power_db.shape == (80, SAMPLE_RATE // 160 + 1)
    assert spectrogram.freqs.shape == (80,)
    assert spectrogram.times[1] - spectrogram.times[0] == pytest.approx(0.01)
    assert spectrogram.power_db.max() == 0.0 and spectrogram.power_db.min() >= -80.0


def test_mel_peak_matches_tone_frequency():
    spectrogram = SpectrogramExtractor().compute(tone(1.0, freq=1000.0))

    peak = spectrogram.freqs[spectrogram.power_db.mean(axis=1).argmax()]
    assert 850 < peak < 1150


def test_linear_spectrogram_respects_band():
    spectrogram = SpectrogramExtractor(kind="linear", fmin=100.0, fmax=4000.0).compute(tone(1.0))

    assert spectrogram.freqs.min() >= 100 and spectrogram.freqs.max() <= 4000
    assert spectrogram.power_db.shape[0] == len(spectrogram.freqs)


@pytest.mark.parametrize("kwargs", [{"kind": "cqt"}, {"n_fft": 256, "win_length": 400}])
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        SpectrogramExtractor(**kwargs)


@pytest.mark.parametrize("fmin, fmax", [(-1.0, 8000.0), (500.0, 400.0), (0.0, 9000.0)])
def test_invalid_mel_band_raises(fmin, fmax):
    with pytest.raises(ValueError, match="zakres"):
        SpectrogramExtractor(fmin=fmin, fmax=fmax).compute(tone(0.5))


def test_features_of_pure_tone():
    features = SpectrogramExtractor().compute(tone(1.0, freq=1000.0)).features

    assert len(features["embedding"]) == 160
    assert np.linalg.norm(features["embedding"]) == pytest.approx(1.0, abs=1e-3)
    assert features["spectral_centroid_hz_mean"] == pytest.approx(1000, rel=0.05)
    assert features["silence_ratio"] == pytest.approx(0.0, abs=0.02)
    assert features["spectral_flatness_mean"] < 0.05


def test_silence_ratio_counts_pause():
    audio = np.concatenate([tone(1.0), np.zeros(SAMPLE_RATE, np.float32)])

    assert SpectrogramExtractor().compute(audio).features["silence_ratio"] == pytest.approx(0.5, abs=0.03)


def test_embedding_ignores_loudness_and_separates_material():
    extractor = SpectrogramExtractor()
    base_audio = np.concatenate([tone(0.5, freq=500.0), noise(0.5), tone(0.5, freq=2000.0)])
    base, quiet, same, other = (
        extractor.compute(audio).features["embedding"]
        for audio in (
            base_audio,
            0.2 * base_audio,
            base_audio + noise(1.5, amp=0.01, seed=1),
            np.concatenate([noise(0.5, seed=2), tone(1.0, freq=300.0)]),
        )
    )

    assert SpectrogramExtractor.cosine_similarity(base, quiet) > 0.999
    assert SpectrogramExtractor.cosine_similarity(base, same) > SpectrogramExtractor.cosine_similarity(base, other)


def test_different_kinds_of_sound_are_clearly_dissimilar():
    extractor = SpectrogramExtractor()
    pure, mixed, noisy = (
        extractor.compute(audio).features["embedding"]
        for audio in (
            tone(1.5, freq=1000.0),
            np.concatenate([tone(0.5, freq=500.0), noise(0.5), tone(0.5, freq=2000.0)]),
            np.concatenate([noise(0.5, seed=2), tone(1.0, freq=300.0)]),
        )
    )

    assert SpectrogramExtractor.cosine_similarity(pure, mixed) < 0.7
    assert SpectrogramExtractor.cosine_similarity(pure, noisy) < 0.7


def test_cosine_similarity():
    assert SpectrogramExtractor.cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert SpectrogramExtractor.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert SpectrogramExtractor.cosine_similarity([0, 0], [1, 1]) == 0.0
    with pytest.raises(ValueError, match="wymiary"):
        SpectrogramExtractor.cosine_similarity([1, 2], [1, 2, 3])


def test_extract_without_output_dir_writes_nothing(tone_wav):
    record = SpectrogramExtractor().extract(tone_wav)

    assert record["status"] == "ok"
    assert record["shape"] == [80, 2 * SAMPLE_RATE // 160 + 1]
    assert record["duration_s"] == pytest.approx(2.0, abs=0.02)
    assert record["params"]["fmax"] == SAMPLE_RATE / 2
    assert record["files"] == {}
    json.dumps(record)


def test_extract_saves_npz_and_png(tone_wav, tmp_path):
    record = SpectrogramExtractor(n_mels=40).extract(tone_wav, str(tmp_path / "out"))

    npz, png = record["files"]["npz"], record["files"]["png"]
    assert Path(npz).name == f"tone_{record['source']['sha256'][:8]}.npz"
    with np.load(npz) as data:
        assert list(data["power_db"].shape) == record["shape"] == [40, 2 * SAMPLE_RATE // 160 + 1]
        assert json.loads(str(data["params"]))["n_mels"] == 40
    assert Path(png).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_extract_can_skip_png(tone_wav, tmp_path):
    record = SpectrogramExtractor().extract(tone_wav, str(tmp_path), save_png=False)

    assert set(record["files"]) == {"npz"}
    assert list(tmp_path.glob("*.png")) == []


def test_extract_statuses_without_sound(tmp_path, silent_video):
    silence = write_wav(tmp_path / "cisza.wav", np.zeros(SAMPLE_RATE, np.float32))
    output_dir = tmp_path / "out"
    extractor = SpectrogramExtractor()

    assert extractor.extract(silence, str(output_dir))["status"] == "silence"
    record = extractor.extract(silent_video, str(output_dir))
    assert (record["status"], record["features"], record["files"]) == ("no_audio", None, {})
    assert not output_dir.exists()


def test_extract_embedding(tone_wav, tmp_path):
    silence = write_wav(tmp_path / "cisza.wav", np.zeros(SAMPLE_RATE, np.float32))
    extractor = SpectrogramExtractor()

    assert len(extractor.extract_embedding(tone_wav)) == 160
    assert extractor.extract_embedding(silence) is None
