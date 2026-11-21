from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, TypeAlias, Union

from sentence_rhythm_extractor import SentenceRhythmExtractor

if TYPE_CHECKING:
    from spectrogram_extractor import SpectrogramExtractor
    from speech_extractor import SegmentDict, SpeechExtractor, TranscriptDict

Granularity: TypeAlias = Literal["segment", "sentence", "word"]
KeyFormat: TypeAlias = Literal["start", "range", "hms"]
TimestampKey: TypeAlias = Union[float, Tuple[float, float], str]

GRANULARITIES = ("segment", "sentence", "word")
KEY_FORMATS = ("start", "range", "hms")


def _format_hms(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


class AudioFeatureExtractor:
    def __init__(
        self,
        speech: SpeechExtractor,
        spectrogram: Optional[SpectrogramExtractor] = None,
        rhythm: Optional[SentenceRhythmExtractor] = None,
    ):
        self.speech = speech
        self.spectrogram = spectrogram
        self.rhythm = rhythm

    def analyze(
        self,
        media_path: str,
        output_dir: Optional[str] = None,
        granularity: Granularity = "sentence",
        key: KeyFormat = "start",
    ) -> Dict[str, Any]:
        self._validate(granularity, key)
        transcript = self.speech.transcribe(
            media_path, word_timestamps=self.rhythm is not None or granularity != "segment"
        )
        return {
            "media": media_path,
            "text": transcript["text"],
            "timestamps": self.timestamps_from_transcript(transcript, granularity, key),
            "rhythm": self.rhythm_from_transcript(transcript) if self.rhythm is not None else None,
            "speech": transcript,
            "spectrogram": (
                self.spectrogram.extract(media_path, output_dir) if self.spectrogram is not None else None
            ),
        }

    def extract_text(
        self,
        media_path: str,
        timestamps: bool = False,
        granularity: Granularity = "segment",
        key: KeyFormat = "start",
    ) -> Union[str, Dict[TimestampKey, str]]:
        if timestamps:
            self._validate(granularity, key)
        transcript = self.speech.transcribe(
            media_path, word_timestamps=timestamps and granularity != "segment"
        )
        if not timestamps:
            return transcript["text"]
        return self.timestamps_from_transcript(transcript, granularity, key)

    def extract_rhythm(self, media_path: str) -> Dict[str, Any]:
        return self.rhythm_from_transcript(self.speech.transcribe(media_path, word_timestamps=True))

    def rhythm_from_transcript(self, transcript: TranscriptDict) -> Dict[str, Any]:
        if self.rhythm is None:
            raise ValueError("Brak ekstraktora rytmu zdań (rhythm=None).")

        segments = transcript["segments"]
        if segments and all(s["words"] for s in segments):
            units = self._units(segments, "sentence")
            return self.rhythm.analyze_sentences(
                [text for _, _, text in units], times=[(start, end) for start, end, _ in units]
            )
        return self.rhythm.analyze_text(transcript["text"])

    @staticmethod
    def timestamps_from_transcript(
        transcript: TranscriptDict,
        granularity: Granularity = "segment",
        key: KeyFormat = "start",
    ) -> Dict[TimestampKey, str]:
        AudioFeatureExtractor._validate(granularity, key)
        timestamps: Dict[TimestampKey, str] = {}
        for start, end, text in AudioFeatureExtractor._units(transcript["segments"], granularity):
            stamp = AudioFeatureExtractor._make_key(start, end, key)
            timestamps[stamp] = f"{timestamps[stamp]} {text}" if stamp in timestamps else text
        return timestamps

    @staticmethod
    def _validate(granularity: str, key: str) -> None:
        if granularity not in GRANULARITIES:
            raise ValueError(
                f"Nieznana granulacja: {granularity!r} (dostępne: {', '.join(GRANULARITIES)})"
            )
        if key not in KEY_FORMATS:
            raise ValueError(f"Nieznany format klucza: {key!r} (dostępne: {', '.join(KEY_FORMATS)})")

    @staticmethod
    def _make_key(start: float, end: float, key: str) -> TimestampKey:
        if key == "start":
            return round(start, 2)
        if key == "range":
            return (round(start, 2), round(end, 2))
        return _format_hms(start)

    @staticmethod
    def _units(segments: Sequence[SegmentDict], granularity: str) -> List[Tuple[float, float, str]]:
        if granularity == "segment":
            return [(s["start"], s["end"], s["text"]) for s in segments if s["text"]]

        units: List[Tuple[float, float, str]] = []
        buffer: List[Dict[str, Any]] = []

        def flush() -> None:
            tokens = [w["word"] for w in buffer]
            for i, j in SentenceRhythmExtractor.segment_sentences(tokens):
                units.append((buffer[i]["start"], buffer[j - 1]["end"], " ".join(tokens[i:j])))
            buffer.clear()

        for segment in segments:
            words = segment["words"]
            if not words:
                flush()
                if segment["text"]:
                    units.append((segment["start"], segment["end"], segment["text"]))
            elif granularity == "word":
                units.extend((w["start"], w["end"], w["word"]) for w in words)
            else:
                buffer.extend(words)
        flush()
        return units
