import av
import hashlib
import io
import numpy as np
import os
import re
import time
import warnings
import wave
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias, TypedDict

from .model_cache import WHISPER, model_dir

Backend: TypeAlias = Literal["local", "api"]

SAMPLE_RATE = 16_000
SILENCE_PEAK = 1e-3
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4
API_CHUNK_SECONDS = 10 * 60

HALLUCINATION_PATTERNS = re.compile(
    r"amara\.org|napisy (stworzone|wykonane|przygotowane)|"
    r"dzi(ę|e)kuj(ę|e) za (uwag(ę|e)|obejrzenie)|dzi(ę|e)ki za obejrzenie|"
    r"subtitles by|thanks? (you )?for watching|transcribed by",
    re.IGNORECASE,
)


class WordDict(TypedDict):
    start: float
    end: float
    word: str
    probability: Optional[float]


class SegmentDict(TypedDict):
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    words: List[WordDict]


class DroppedSegmentDict(SegmentDict):
    reason: str


class TranscriptDict(TypedDict):
    source: Dict[str, Any]
    backend: Backend
    model: str
    device: Optional[str]
    status: str
    language: Optional[str]
    text: str
    segments: List[SegmentDict]
    dropped_segments: List[DroppedSegmentDict]
    speech_seconds: float
    speech_ratio: float
    processing_seconds: float


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audio(
    path: str, sample_rate: int = SAMPLE_RATE
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku pod ścieżką: {path}")

    meta: Dict[str, Any] = {"has_audio": False, "has_video": False}
    with av.open(str(path)) as container:
        meta["container"] = container.format.name
        if container.duration is not None:
            meta["container_duration_s"] = round(container.duration / av.time_base, 3)
        meta["has_video"] = bool(container.streams.video)
        if not container.streams.audio:
            return None, meta

        stream = container.streams.audio[0]
        meta.update(
            has_audio=True,
            audio_codec=stream.codec_context.name,
            audio_sample_rate=stream.codec_context.sample_rate,
            audio_channels=stream.codec_context.channels,
        )
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        chunks: List[np.ndarray] = []
        for frame in container.decode(stream):
            chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(None))

    audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    meta["audio_duration_s"] = round(len(audio) / sample_rate, 3)
    return audio, meta


class SpeechExtractor:
    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: Backend = "local",
        language: Optional[str] = None,
        device: Optional[str] = None,
        models_dir: Optional[str] = None,
    ):
        if backend not in ("local", "api"):
            raise ValueError(f"Nieznany backend: {backend!r} (dostępne: local, api)")

        self.backend = backend
        self.language = language
        self.device = (device or self._default_device()) if backend == "local" else None
        self.model_name = model_name or self._default_model(backend, self.device)
        self.model: Any = None
        self.client: Any = None

        if backend == "local":
            import whisper

            self.model = whisper.load_model(
                self.model_name,
                device=self.device,
                download_root=str(model_dir(WHISPER, models_dir)),
            )
        else:
            from openai import OpenAI

            self.client = OpenAI()

    @staticmethod
    def _default_device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _default_model(backend: Backend, device: Optional[str]) -> str:
        if backend == "api":
            return "whisper-1"
        return "turbo" if device == "cuda" else "small"

    def extract_text(self, media_path: str) -> str:
        return self.transcribe(media_path)["text"]

    def transcribe(self, media_path: str, word_timestamps: bool = False) -> TranscriptDict:
        started = time.perf_counter()
        audio, meta = load_audio(media_path)
        transcript: TranscriptDict = {
            "source": {
                "path": str(media_path),
                "sha256": sha256_file(media_path),
                "size_bytes": os.path.getsize(media_path),
                **meta,
            },
            "backend": self.backend,
            "model": self.model_name,
            "device": self.device,
            "status": "ok",
            "language": None,
            "text": "",
            "segments": [],
            "dropped_segments": [],
            "speech_seconds": 0.0,
            "speech_ratio": 0.0,
            "processing_seconds": 0.0,
        }

        if audio is None:
            transcript["status"] = "no_audio"
        elif len(audio) < SAMPLE_RATE * 0.3 or float(np.abs(audio).max()) < SILENCE_PEAK:
            transcript["status"] = "silence"
        else:
            if self.backend == "local":
                language, segments = self._transcribe_local(audio, word_timestamps)
            else:
                language, segments = self._transcribe_api(audio, word_timestamps)

            for segment in segments:
                reason = self._drop_reason(segment)
                if reason:
                    transcript["dropped_segments"].append({**segment, "reason": reason})
                else:
                    transcript["segments"].append(segment)

            kept = transcript["segments"]
            speech = sum(s["end"] - s["start"] for s in kept)
            duration = len(audio) / SAMPLE_RATE
            transcript["language"] = language
            transcript["text"] = " ".join(s["text"] for s in kept).strip()
            transcript["speech_seconds"] = round(speech, 2)
            transcript["speech_ratio"] = round(min(speech / duration, 1.0), 3) if duration else 0.0
            if not kept:
                transcript["status"] = "no_speech"

        transcript["processing_seconds"] = round(time.perf_counter() - started, 2)
        return transcript

    def _transcribe_local(
        self, audio: np.ndarray, word_timestamps: bool
    ) -> Tuple[Optional[str], List[SegmentDict]]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Failed to launch Triton kernels", category=UserWarning
            )
            result = self.model.transcribe(
                audio,
                language=self.language,
                task="transcribe",
                fp16=self.device == "cuda",
                condition_on_previous_text=False,
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                logprob_threshold=LOGPROB_THRESHOLD,
                compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
                word_timestamps=word_timestamps,
                verbose=None,
            )

        segments: List[SegmentDict] = []
        for segment in result["segments"]:
            segments.append(
                {
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": segment["text"].strip(),
                    "avg_logprob": float(segment["avg_logprob"]),
                    "no_speech_prob": float(segment["no_speech_prob"]),
                    "compression_ratio": float(segment["compression_ratio"]),
                    "words": [
                        self._word(w["start"], w["end"], w["word"], w.get("probability"))
                        for w in segment.get("words", [])
                        if w["word"].strip()
                    ],
                }
            )
        return result.get("language"), segments

    def _transcribe_api(
        self, audio: np.ndarray, word_timestamps: bool
    ) -> Tuple[Optional[str], List[SegmentDict]]:
        step = API_CHUNK_SECONDS * SAMPLE_RATE
        language: Optional[str] = None
        segments: List[SegmentDict] = []
        for offset in range(0, len(audio), step):
            request: Dict[str, Any] = {
                "model": self.model_name,
                "file": self._wav_bytes(audio[offset : offset + step]),
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment", "word"] if word_timestamps else ["segment"],
            }
            if self.language:
                request["language"] = self.language
            response = self.client.audio.transcriptions.create(**request)
            language = language or getattr(response, "language", None)

            t0 = offset / SAMPLE_RATE
            words: List[WordDict] = []
            if word_timestamps:
                words = [
                    self._word(t0 + float(w.start), t0 + float(w.end), w.word)
                    for w in getattr(response, "words", None) or []
                    if w.word.strip()
                ]
            raw_segments = getattr(response, "segments", None) or []
            for i, s in enumerate(raw_segments):
                start, end = t0 + float(s.start), t0 + float(s.end)
                last = i == len(raw_segments) - 1
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "text": s.text.strip(),
                        "avg_logprob": float(getattr(s, "avg_logprob", 0.0)),
                        "no_speech_prob": float(getattr(s, "no_speech_prob", 0.0)),
                        "compression_ratio": float(getattr(s, "compression_ratio", 0.0)),
                        "words": [
                            w for w in words if w["start"] >= start - 0.01 and (last or w["start"] < end)
                        ],
                    }
                )
        return self._normalize_language(language), segments

    @staticmethod
    def _word(start: float, end: float, word: str, probability: Optional[float] = None) -> WordDict:
        return {
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "word": word.strip(),
            "probability": round(float(probability), 4) if probability is not None else None,
        }

    @staticmethod
    def _wav_bytes(audio: np.ndarray) -> io.BytesIO:
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        buffer.seek(0)
        buffer.name = "audio.wav"
        return buffer

    @staticmethod
    def _normalize_language(language: Optional[str]) -> Optional[str]:
        if not language or len(language) <= 3:
            return language
        try:
            from whisper.tokenizer import TO_LANGUAGE_CODE
        except ImportError:
            return language
        return TO_LANGUAGE_CODE.get(language.lower(), language)

    @staticmethod
    def _drop_reason(segment: SegmentDict) -> Optional[str]:
        if not segment["text"]:
            return "empty"
        if HALLUCINATION_PATTERNS.search(segment["text"]):
            return "known_hallucination"
        if segment["no_speech_prob"] > NO_SPEECH_THRESHOLD and segment["avg_logprob"] < LOGPROB_THRESHOLD:
            return "no_speech"
        if segment["compression_ratio"] > COMPRESSION_RATIO_THRESHOLD:
            return "repetitive"
        return None
