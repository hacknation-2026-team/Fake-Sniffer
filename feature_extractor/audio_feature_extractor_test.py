import copy
import json
import pytest
from unittest.mock import MagicMock

from feature_extractor.audio_feature_extractor import AudioFeatureExtractor
from feature_extractor.sentence_rhythm_extractor import SentenceRhythmExtractor


def word(start, end, text):
    return {"start": start, "end": end, "word": text, "probability": 0.9}


def segment(start, end, text, words=()):
    return {
        "start": start,
        "end": end,
        "text": text,
        "avg_logprob": -0.2,
        "no_speech_prob": 0.01,
        "compression_ratio": 1.2,
        "words": list(words),
    }


TRANSCRIPT = {
    "status": "ok",
    "text": "Uwaga! Znany inwestor ujawnia sekret. Zarejestruj się dzisiaj",
    "segments": [
        segment(0.5, 3.0, "Uwaga! Znany inwestor ujawnia sekret.", [
            word(0.5, 0.9, "Uwaga!"), word(1.0, 1.4, "Znany"), word(1.4, 2.0, "inwestor"),
            word(2.0, 2.5, "ujawnia"), word(2.5, 3.0, "sekret."),
        ]),
        segment(3.2, 5.0, "Zarejestruj się dzisiaj", [
            word(3.2, 3.9, "Zarejestruj"), word(3.9, 4.2, "się"), word(4.2, 5.0, "dzisiaj"),
        ]),
    ],
}
SENTENCES = {0.5: "Uwaga!", 1.0: "Znany inwestor ujawnia sekret.", 3.2: "Zarejestruj się dzisiaj"}


@pytest.fixture
def speech():
    def transcribe(media_path, word_timestamps=False):
        transcript = copy.deepcopy(TRANSCRIPT)
        if not word_timestamps:
            for s in transcript["segments"]:
                s["words"] = []
        return transcript

    mock = MagicMock()
    mock.transcribe.side_effect = transcribe
    return mock


def test_extract_text_returns_plain_text(speech):
    text = AudioFeatureExtractor(speech).extract_text("ad.mp4")

    assert text == TRANSCRIPT["text"]
    speech.transcribe.assert_called_once_with("ad.mp4", word_timestamps=False)


def test_segment_timestamps_do_not_need_word_times(speech):
    timestamps = AudioFeatureExtractor(speech).extract_text("ad.mp4", timestamps=True)

    assert timestamps == {0.5: "Uwaga! Znany inwestor ujawnia sekret.", 3.2: "Zarejestruj się dzisiaj"}
    speech.transcribe.assert_called_once_with("ad.mp4", word_timestamps=False)


@pytest.mark.parametrize("key, expected", [
    ("start", SENTENCES),
    ("range", {
        (0.5, 0.9): "Uwaga!",
        (1.0, 3.0): "Znany inwestor ujawnia sekret.",
        (3.2, 5.0): "Zarejestruj się dzisiaj",
    }),
    ("hms", {
        "00:00:00.500": "Uwaga!",
        "00:00:01.000": "Znany inwestor ujawnia sekret.",
        "00:00:03.200": "Zarejestruj się dzisiaj",
    }),
])
def test_sentence_timestamps(speech, key, expected):
    timestamps = AudioFeatureExtractor(speech).extract_text(
        "ad.mp4", timestamps=True, granularity="sentence", key=key
    )

    assert timestamps == expected
    speech.transcribe.assert_called_once_with("ad.mp4", word_timestamps=True)


def test_word_timestamps(speech):
    timestamps = AudioFeatureExtractor(speech).extract_text("ad.mp4", timestamps=True, granularity="word")

    assert list(timestamps)[:2] == [0.5, 1.0]
    assert list(timestamps.values()) == [
        "Uwaga!", "Znany", "inwestor", "ujawnia", "sekret.", "Zarejestruj", "się", "dzisiaj",
    ]


@pytest.mark.parametrize("kwargs, match", [
    ({"granularity": "paragraph"}, "granulacja"),
    ({"key": "iso"}, "format klucza"),
])
def test_invalid_options_fail_before_transcription(speech, kwargs, match):
    with pytest.raises(ValueError, match=match):
        AudioFeatureExtractor(speech).extract_text("ad.mp4", timestamps=True, **kwargs)

    speech.transcribe.assert_not_called()


def test_sentence_spans_segments_and_keeps_abbreviations():
    transcript = {
        "segments": [
            segment(0.0, 1.0, "Znany lekarz, np.", [
                word(0.0, 0.5, "Znany"), word(0.5, 0.8, "lekarz,"), word(0.8, 1.0, "np."),
            ]),
            segment(1.0, 3.0, "Jan Kowalski. Kup teraz.", [
                word(1.0, 1.5, "Jan"), word(1.5, 2.0, "Kowalski."), word(2.2, 2.6, "Kup"), word(2.6, 3.0, "teraz."),
            ]),
        ]
    }

    timestamps = AudioFeatureExtractor.timestamps_from_transcript(transcript, "sentence", "range")

    assert timestamps == {(0.0, 2.0): "Znany lekarz, np. Jan Kowalski.", (2.2, 3.0): "Kup teraz."}


def test_segments_without_words_fall_back_to_segment_text():
    transcript = {
        "segments": [
            segment(0.0, 1.0, "Raz dwa", [word(0.0, 0.5, "Raz"), word(0.5, 1.0, "dwa")]),
            segment(1.0, 2.0, "Segment bez słów."),
        ]
    }

    timestamps = AudioFeatureExtractor.timestamps_from_transcript(transcript, "sentence")

    assert timestamps == {0.0: "Raz dwa", 1.0: "Segment bez słów."}


def test_analyze_with_all_extractors(speech):
    spectrogram = MagicMock()
    spectrogram.extract.return_value = {"status": "ok"}

    result = AudioFeatureExtractor(speech, spectrogram, SentenceRhythmExtractor()).analyze("ad.mp4", "out")

    speech.transcribe.assert_called_once_with("ad.mp4", word_timestamps=True)
    spectrogram.extract.assert_called_once_with("ad.mp4", "out")
    assert result["text"] == TRANSCRIPT["text"]
    assert result["timestamps"] == SENTENCES
    assert result["rhythm"]["n_sentences"] == 3
    assert result["spectrogram"] == {"status": "ok"}
    json.dumps(result, ensure_ascii=False)


def test_analyze_without_optional_extractors(speech):
    result = AudioFeatureExtractor(speech).analyze("ad.mp4", granularity="segment")

    speech.transcribe.assert_called_once_with("ad.mp4", word_timestamps=False)
    assert result["rhythm"] is None and result["spectrogram"] is None
    assert list(result["timestamps"]) == [0.5, 3.2]


def test_extract_rhythm_uses_word_times(speech):
    rhythm = AudioFeatureExtractor(speech, rhythm=SentenceRhythmExtractor()).extract_rhythm("ad.mp4")

    sentences = rhythm["sentences"]
    assert [(s["start"], s["end"]) for s in sentences] == [(0.5, 0.9), (1.0, 3.0), (3.2, 5.0)]
    assert [s["pause_before_s"] for s in sentences] == [None, 0.1, 0.2]
    assert [s["type"] for s in sentences] == ["declarative", "declarative", "imperative"]
    assert rhythm["timing"]["words_per_minute"] == pytest.approx(8 / 4.2 * 60, abs=0.1)


def test_rhythm_without_word_times_uses_text():
    extractor = AudioFeatureExtractor(MagicMock(), rhythm=SentenceRhythmExtractor())

    transcript = {"text": "Raz. Dwa trzy.", "segments": [segment(0.0, 2.0, "Raz. Dwa trzy.")]}

    rhythm = extractor.rhythm_from_transcript(transcript)

    assert rhythm["n_sentences"] == 2 and rhythm["timing"] is None


def test_extract_rhythm_requires_rhythm_extractor(speech):
    with pytest.raises(ValueError, match="rytmu"):
        AudioFeatureExtractor(speech).extract_rhythm("ad.mp4")
