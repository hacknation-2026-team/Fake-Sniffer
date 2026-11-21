import av
import cv2
import json
import numpy as np
import pytest
import warnings
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from feature_extractor.model_cache import WHISPER, model_dir
from feature_extractor.speech_extractor import SAMPLE_RATE, SpeechExtractor, load_audio, sha256_file


def tone(seconds: float, freq: float = 440.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return str(path)


def whisper_segment(start, end, text, words=(), avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.2):
    return {
        "start": start,
        "end": end,
        "text": text,
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
        "compression_ratio": compression_ratio,
        "words": list(words),
    }


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


@pytest.fixture
def mock_model():
    with patch("whisper.load_model") as mock_load:
        model = MagicMock()
        model.transcribe.return_value = {
            "language": "pl",
            "segments": [whisper_segment(0.0, 1.5, " Zainwestuj teraz. ")],
        }
        mock_load.return_value = model
        yield model


@pytest.mark.parametrize("device, expected_model", [("cuda", "turbo"), ("cpu", "small")])
def test_loads_default_model_for_device(device, expected_model):
    with patch("whisper.load_model") as mock_load:
        extractor = SpeechExtractor(device=device)

    assert extractor.model_name == expected_model
    mock_load.assert_called_once_with(
        expected_model, device=device, download_root=str(model_dir(WHISPER))
    )


def test_whisper_downloads_into_requested_directory(tmp_path):
    with patch("whisper.load_model") as mock_load:
        SpeechExtractor(device="cpu", models_dir=str(tmp_path / "m"))
    assert mock_load.call_args[1]["download_root"] == str(tmp_path / "m" / "whisper")


def test_api_backend_does_not_create_model_directory(tmp_path):
    with patch("openai.OpenAI"):
        SpeechExtractor(backend="api", models_dir=str(tmp_path / "m"))
    assert not (tmp_path / "m").exists()


def test_api_backend_creates_openai_client():
    with patch("openai.OpenAI") as mock_client_cls:
        extractor = SpeechExtractor(backend="api")

    assert (extractor.model_name, extractor.device) == ("whisper-1", None)
    assert extractor.client is mock_client_cls.return_value


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        SpeechExtractor(backend="azure")


def test_load_audio_resamples_to_16k_mono(tmp_path):
    path = write_wav(tmp_path / "8k.wav", tone(2.0, sample_rate=8000), sample_rate=8000)

    audio, meta = load_audio(path)

    assert audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) == pytest.approx(2 * SAMPLE_RATE, abs=SAMPLE_RATE * 0.01)
    assert (meta["audio_sample_rate"], meta["has_video"]) == (8000, False)


def test_load_audio_returns_none_for_video_without_sound(silent_video):
    audio, meta = load_audio(silent_video)

    assert audio is None
    assert (meta["has_video"], meta["has_audio"]) == (True, False)


def test_missing_and_corrupt_files_raise(mock_model, tmp_path):
    corrupt = tmp_path / "uszkodzony.mp4"
    corrupt.write_bytes(b"to nie jest plik wideo" * 50)
    extractor = SpeechExtractor(device="cpu")

    with pytest.raises(FileNotFoundError, match="brak.mp4"):
        extractor.transcribe("brak.mp4")
    with pytest.raises(av.error.InvalidDataError):
        extractor.transcribe(str(corrupt))


def test_transcribe_returns_text_and_speech_stats(mock_model, tone_wav):
    result = SpeechExtractor(language="pl", device="cpu").transcribe(tone_wav)

    assert (result["status"], result["language"], result["text"]) == ("ok", "pl", "Zainwestuj teraz.")
    assert result["speech_seconds"] == 1.5
    assert result["speech_ratio"] == pytest.approx(0.75, abs=0.01)
    assert result["source"]["sha256"] == sha256_file(tone_wav)
    assert (result["model"], result["device"]) == ("small", "cpu")
    json.dumps(result)


def test_transcribe_passes_options_to_whisper(mock_model, tone_wav):
    SpeechExtractor(language="pl", device="cpu").transcribe(tone_wav, word_timestamps=True)

    audio = mock_model.transcribe.call_args.args[0]
    options = mock_model.transcribe.call_args.kwargs
    assert audio.dtype == np.float32
    assert len(audio) == pytest.approx(2 * SAMPLE_RATE, abs=SAMPLE_RATE * 0.01)
    assert options["language"] == "pl"
    assert options["word_timestamps"] is True
    assert options["fp16"] is False
    assert options["condition_on_previous_text"] is False


def test_unreliable_segments_are_dropped_with_reason(mock_model, tone_wav):
    mock_model.transcribe.return_value = {
        "language": "pl",
        "segments": [
            whisper_segment(0.0, 0.4, "Napisy stworzone przez społeczność Amara.org"),
            whisper_segment(0.4, 0.9, " Pierwsze zdanie."),
            whisper_segment(0.9, 1.0, "  "),
            whisper_segment(1.0, 1.2, "coś", no_speech_prob=0.9, avg_logprob=-1.5),
            whisper_segment(1.2, 1.4, "tak tak tak tak", compression_ratio=3.1),
            whisper_segment(1.4, 1.9, " Drugie zdanie.", no_speech_prob=0.9, avg_logprob=-0.3),
        ],
    }

    result = SpeechExtractor(device="cpu").transcribe(tone_wav)

    assert result["text"] == "Pierwsze zdanie. Drugie zdanie."
    assert result["speech_seconds"] == 1.0
    assert [s["reason"] for s in result["dropped_segments"]] == [
        "known_hallucination", "empty", "no_speech", "repetitive",
    ]


def test_only_hallucinations_give_no_speech(mock_model, tone_wav):
    mock_model.transcribe.return_value = {
        "language": "pl",
        "segments": [whisper_segment(0.0, 2.0, " Dziękuję za uwagę.")],
    }

    result = SpeechExtractor(device="cpu").transcribe(tone_wav)

    assert (result["status"], result["text"], len(result["dropped_segments"])) == ("no_speech", "", 1)


def test_silence_and_video_without_sound_skip_whisper(mock_model, tmp_path, silent_video):
    silence = write_wav(tmp_path / "cisza.wav", np.zeros(3 * SAMPLE_RATE, np.float32))
    extractor = SpeechExtractor(device="cpu")

    assert extractor.transcribe(silence)["status"] == "silence"
    assert extractor.transcribe(silent_video)["status"] == "no_audio"
    mock_model.transcribe.assert_not_called()


def test_word_timestamps_are_cleaned(mock_model, tone_wav):
    mock_model.transcribe.return_value = {
        "language": "pl",
        "segments": [
            whisper_segment(0.0, 1.0, " Kup teraz.", words=[
                {"start": 0.0, "end": 0.4, "word": " Kup", "probability": 0.98123},
                {"start": 0.4, "end": 1.0, "word": " teraz.", "probability": 0.9},
                {"start": 1.0, "end": 1.0, "word": " ", "probability": 0.1},
            ]),
        ],
    }

    segment = SpeechExtractor(device="cpu").transcribe(tone_wav, word_timestamps=True)["segments"][0]

    assert segment["words"] == [
        {"start": 0.0, "end": 0.4, "word": "Kup", "probability": 0.9812},
        {"start": 0.4, "end": 1.0, "word": "teraz.", "probability": 0.9},
    ]


def test_triton_warning_is_silenced(mock_model, tone_wav):
    def transcribe(audio, **kwargs):
        warnings.warn("Failed to launch Triton kernels, likely due to missing CUDA toolkit", UserWarning)
        warnings.warn("inne ostrzeżenie", UserWarning)
        return {"language": "pl", "segments": []}

    mock_model.transcribe.side_effect = transcribe

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SpeechExtractor(device="cpu").transcribe(tone_wav)

    assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == ["inne ostrzeżenie"]


def test_api_backend_splits_audio_and_assigns_words(tmp_path):
    path = write_wav(tmp_path / "dlugie.wav", tone(2.5))
    requests = []

    def create(**kwargs):
        with wave.open(kwargs["file"], "rb") as wav:
            requests.append({**kwargs, "duration": wav.getnframes() / wav.getframerate()})
        n = len(requests)
        return SimpleNamespace(
            language="polish",
            segments=[
                SimpleNamespace(
                    start=0.0, end=0.5, text=f" Część {n}.", avg_logprob=-0.2, no_speech_prob=0.01,
                    compression_ratio=1.2,
                )
            ],
            words=[
                SimpleNamespace(start=0.0, end=0.3, word=" Część"),
                SimpleNamespace(start=0.3, end=0.6, word=f" {n}."),
            ],
        )

    with patch("openai.OpenAI") as mock_client_cls, patch("feature_extractor.speech_extractor.API_CHUNK_SECONDS", 1):
        mock_client_cls.return_value.audio.transcriptions.create.side_effect = create
        result = SpeechExtractor(backend="api", language="pl").transcribe(path, word_timestamps=True)

    assert [r["duration"] for r in requests] == [1.0, 1.0, 0.5]
    assert requests[0]["language"] == "pl"
    assert requests[0]["timestamp_granularities"] == ["segment", "word"]
    assert (result["language"], result["text"]) == ("pl", "Część 1. Część 2. Część 3.")
    assert [s["start"] for s in result["segments"]] == [0.0, 1.0, 2.0]
    assert [w["word"] for w in result["segments"][1]["words"]] == ["Część", "2."]


@pytest.mark.integration
def test_real_whisper_returns_clean_record(tmp_path):
    path = write_wav(tmp_path / "ton.wav", tone(3.0))
    extractor = SpeechExtractor(language="pl")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = extractor.transcribe(path, word_timestamps=True)

    assert result["status"] in ("ok", "no_speech")
    assert result["model"] == ("turbo" if extractor.device == "cuda" else "small")
    assert all(set(w) == {"start", "end", "word", "probability"} for s in result["segments"] for w in s["words"])
    assert not any("Triton" in str(w.message) for w in caught)
    json.dumps(result)
