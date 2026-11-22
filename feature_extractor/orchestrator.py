from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .audio_feature_extractor import AudioFeatureExtractor
    from .deepfake_extractor import DeepfakeDetector
    from .face_identity_extractor import FaceIdentityExtractor, ReferenceFaceDatabase
    from .keyframe_extractor import LightweightKeyframeExtractor
    from .named_entity_extractor import NamedEntityExtractor
    from .ocr_text_extractor import TextExtractor
    from .paragraph_extractor import ParagraphExtractor
    from .sentence_rhythm_extractor import SentenceRhythmExtractor
    from .spectrogram_extractor import SpectrogramExtractor
    from .speech_extractor import SpeechExtractor
    from .text_chunk_embedder import DocumentEmbedder
    from .video_feature_extractor import VideoFeatureExtractor
    from .visual_feature_extractor import ImageHasher, VisualEmbedder
    from .web_infrastructure_extractor import WebInfrastructureExtractor

VECTOR_KEYS = frozenset({"embedding", "face_embeddings", "visual_embedding"})
TEXT_STEPS = ("named_entities", "text_embeddings")


@dataclass
class OrchestratorConfig:
    """Ustawienia modeli i ścieżek. Wartości domyślne działają na CPU."""

    use_gpu: bool = False
    ocr_languages: List[str] = field(default_factory=lambda: ["pl", "en"])
    min_ocr_confidence: float = 0.3
    embedder_model: str = "sdadas/mmlw-roberta-large"
    embedder_prefixes: Optional[Dict[str, str]] = None
    spacy_model: str = "pl_core_news_lg"
    entity_patterns: List[Tuple[str, str]] = field(default_factory=list)
    clip_model: str = "clip-ViT-B-32"
    keyframes_dir: str = "keyframes_out"
    reference_faces_dir: Optional[str] = None
    face_threshold: float = 0.45
    whisper_model: Optional[str] = None
    speech_language: Optional[str] = None
    spectrograms_dir: Optional[str] = None
    include_vectors: bool = True
    models_dir: Optional[str] = None
    deepfake: bool = False
    deepfake_image_model: Optional[str] = None
    deepfake_audio_model: Optional[str] = None
    deepfake_audio_fake_label: Optional[str] = None


@dataclass
class StepResult:
    name: str
    source: str
    status: str
    duration_sec: float
    data: Any = None
    error: Optional[str] = None


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Nie da się zapisać do JSON: {type(value).__name__}")


def _without_vectors(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: None if k in VECTOR_KEYS else _without_vectors(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_vectors(item) for item in value]
    return value


def _video_text(ocr_text: Optional[str], transcript: Optional[str]) -> Tuple[Optional[str], str]:
    ocr_text, transcript = (ocr_text or "").strip(), (transcript or "").strip()
    if ocr_text and transcript:
        return f"{ocr_text}\n\n{transcript}", "unknown"
    if transcript:
        return transcript, "transcript"
    return ocr_text or None, "ocr"


@dataclass
class Report:
    """Wynik jednego uruchomienia: wejścia i kroki w kolejności wykonania."""

    created_at: str
    inputs: Dict[str, str]
    steps: List[StepResult] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(step.status == "error" for step in self.steps)

    def step(self, name: str, source: Optional[str] = None) -> Optional[StepResult]:
        """Pierwszy krok o danej nazwie (opcjonalnie z danego źródła)."""
        return next(
            (s for s in self.steps if s.name == name and (source is None or s.source == source)),
            None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=_json_default)

    def save_json(self, path: str | Path) -> Path:
        """Zapisuje raport jako UTF-8, tworząc brakujące katalogi. Zwraca ścieżkę pliku."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target


class Orchestrator:
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Uruchamia ekstraktory zależnie od podanych wejść: url, tekst, obraz, wideo, audio.
        Modele ładują się dopiero przy pierwszym użyciu, a błąd jednego kroku
        trafia do raportu i nie zatrzymuje pozostałych.
        """
        self.config = config or OrchestratorConfig()

    @property
    def _device(self) -> str:
        return "cuda" if self.config.use_gpu else "cpu"

    @cached_property
    def ocr(self) -> TextExtractor:
        from .ocr_text_extractor import TextExtractor

        return TextExtractor(self.config.ocr_languages, self.config.use_gpu, self.config.models_dir)

    @cached_property
    def text_embedder(self) -> DocumentEmbedder:
        from .text_chunk_embedder import DocumentEmbedder

        return DocumentEmbedder(
            model_name=self.config.embedder_model,
            device=self._device,
            prefixes=self.config.embedder_prefixes,
            models_dir=self.config.models_dir,
        )

    @cached_property
    def entity_extractor(self) -> NamedEntityExtractor:
        from .named_entity_extractor import NamedEntityExtractor

        return NamedEntityExtractor(self.config.spacy_model, self.config.entity_patterns)

    @cached_property
    def hasher(self) -> ImageHasher:
        from .visual_feature_extractor import ImageHasher

        return ImageHasher()

    @cached_property
    def visual_embedder(self) -> VisualEmbedder:
        from .visual_feature_extractor import VisualEmbedder

        return VisualEmbedder(
            self.config.clip_model, device=self._device, models_dir=self.config.models_dir
        )

    @cached_property
    def faces(self) -> FaceIdentityExtractor:
        from .face_identity_extractor import FaceIdentityExtractor

        return FaceIdentityExtractor(use_gpu=self.config.use_gpu, models_dir=self.config.models_dir)

    @cached_property
    def face_database(self) -> Optional[ReferenceFaceDatabase]:
        if not self.config.reference_faces_dir:
            return None
        return self.faces.build_reference_database(self.config.reference_faces_dir)

    @cached_property
    def keyframes(self) -> LightweightKeyframeExtractor:
        from .keyframe_extractor import LightweightKeyframeExtractor

        return LightweightKeyframeExtractor()

    @cached_property
    def video_extractor(self) -> VideoFeatureExtractor:
        from .video_feature_extractor import VideoFeatureExtractor

        return VideoFeatureExtractor(
            self.keyframes,
            self.hasher,
            ocr=self.ocr,
            faces=self.faces,
            embedder=self.visual_embedder,
        )

    @cached_property
    def web_extractor(self) -> WebInfrastructureExtractor:
        from .web_infrastructure_extractor import WebInfrastructureExtractor

        return WebInfrastructureExtractor(models_dir=self.config.models_dir)

    @cached_property
    def speech(self) -> SpeechExtractor:
        from .speech_extractor import SpeechExtractor

        return SpeechExtractor(
            model_name=self.config.whisper_model,
            language=self.config.speech_language,
            device=self._device,
            models_dir=self.config.models_dir,
        )

    @cached_property
    def spectrogram(self) -> SpectrogramExtractor:
        from .spectrogram_extractor import SpectrogramExtractor

        return SpectrogramExtractor()

    @cached_property
    def sentence_rhythm(self) -> SentenceRhythmExtractor:
        from .sentence_rhythm_extractor import SentenceRhythmExtractor

        return SentenceRhythmExtractor()

    @cached_property
    def paragraphs(self) -> ParagraphExtractor:
        from .paragraph_extractor import ParagraphExtractor

        return ParagraphExtractor(rhythm=self.sentence_rhythm)

    @cached_property
    def audio_extractor(self) -> AudioFeatureExtractor:
        from .audio_feature_extractor import AudioFeatureExtractor

        return AudioFeatureExtractor(self.speech, self.spectrogram, self.sentence_rhythm)

    @cached_property
    def deepfake_detector(self) -> DeepfakeDetector:
        from .deepfake_extractor import AUDIO_MODEL, IMAGE_MODEL, DeepfakeDetector

        return DeepfakeDetector(
            image_model=self.config.deepfake_image_model or IMAGE_MODEL,
            audio_model=self.config.deepfake_audio_model or AUDIO_MODEL,
            device=self._device,
            models_dir=self.config.models_dir,
            audio_fake_label=self.config.deepfake_audio_fake_label,
        )

    def prefetch(self) -> Dict[str, str]:
        """Pobiera wszystkie modele z góry. Zwraca status każdego: "ok" albo opis błędu."""
        loaders: Dict[str, Callable[[], Any]] = {
            "ocr": lambda: self.ocr,
            "text_embedder": lambda: self.text_embedder,
            "entity_extractor": lambda: self.entity_extractor,
            "visual_embedder": lambda: self.visual_embedder,
            "faces": lambda: self.faces,
            "speech": lambda: self.speech,
            "browser": lambda: self.web_extractor.ensure_browser(),
        }
        if self.config.deepfake:
            loaders["deepfake_image"] = lambda: self.deepfake_detector.load_image_model()
            loaders["deepfake_audio"] = lambda: self.deepfake_detector.load_audio_model()
        statuses: Dict[str, str] = {}
        for name, load in loaders.items():
            try:
                load()
                statuses[name] = "ok"
            except Exception as exc:
                statuses[name] = f"{type(exc).__name__}: {exc}"
        return statuses

    def run(
        self,
        url: Optional[str] = None,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        audio: Optional[str] = None,
    ) -> Report:
        """
        Kroki idą w kolejności url, tekst, obraz, wideo, audio. Tekst z OCR, ze strony
        lub z mowy przechodzi dodatkowo przez rozpoznawanie nazw własnych i embeddingi.
        """
        inputs = (("url", url), ("text", text), ("image", image), ("video", video), ("audio", audio))
        given = {name: value for name, value in inputs if value and value.strip()}
        if not given:
            raise ValueError("Podaj co najmniej jedno wejście: url, text, image, video lub audio.")

        report = Report(
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            inputs=given,
        )
        if "url" in given:
            self._run_url(report, given["url"])
        if "text" in given:
            self._run_text(report, given["text"])
        if "image" in given:
            self._run_image(report, given["image"])
        if "video" in given:
            self._run_video(report, given["video"])
        if "audio" in given:
            self._run_audio(report, given["audio"])
        return report

    def _step(self, report: Report, name: str, source: str, action: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        try:
            data, status, error = action(), "ok", None
        except Exception as exc:
            data, status, error = None, "error", f"{type(exc).__name__}: {exc}"

        recorded = data if self.config.include_vectors else _without_vectors(data)
        report.steps.append(
            StepResult(name, source, status, round(time.perf_counter() - started, 3), recorded, error)
        )
        return data

    def _text_steps(self, report: Report, source: str, text: Optional[str], source_type: str) -> None:
        if not text or not text.strip():
            for name in TEXT_STEPS:
                report.steps.append(StepResult(name, source, "skipped", 0.0, None, "brak tekstu"))
            return

        self._step(
            report, "named_entities", source, lambda: self.entity_extractor.extract_entities(text)
        )
        self._step(
            report,
            "text_embeddings",
            source,
            lambda: [
                asdict(chunk)
                for chunk in self.text_embedder.extract_embeddings(text, source_type=source_type)
            ],
        )

    def _run_text(self, report: Report, text: str) -> None:
        self._text_steps(report, "text", text, "unknown")
        self._step(report, "sentence_rhythm", "text", lambda: self.sentence_rhythm.analyze_text(text))
        self._step(report, "paragraphs", "text", lambda: self.paragraphs.analyze_text(text))

    def _run_url(self, report: Report, url: str) -> None:
        page: Dict[str, Any] = {}
        data = self._step(report, "web_infrastructure", "url", lambda: self._analyze_page(url, page))
        if data and data.get("errors"):
            report.steps[-1].status = "partial"
            report.steps[-1].error = "; ".join(data["errors"])
        self._text_steps(report, "url", (data or {}).get("page_text"), "webpage")
        if not page.get("html"):
            report.steps.append(StepResult("paragraphs", "url", "skipped", 0.0, None, "brak HTML strony"))
            return
        self._step(
            report,
            "paragraphs",
            "url",
            lambda: self.paragraphs.analyze_html(page["html"], url=data.get("final_url") or url),
        )

    def _analyze_page(self, url: str, page: Dict[str, Any]) -> Dict[str, Any]:
        data = self.web_extractor.analyze(url).to_dict()
        page["html"] = data.pop("page_html", None)
        return data

    def _run_image(self, report: Report, path: str) -> None:
        ocr = self._step(report, "ocr", "image", lambda: self._read_image_text(path))
        self._text_steps(report, "image", (ocr or {}).get("text"), "ocr")
        self._step(report, "visual_hashes", "image", lambda: self.hasher.compute(path))
        self._step(
            report, "visual_embedding", "image", lambda: self.visual_embedder.extract_embedding(path)
        )
        self._step(report, "faces", "image", lambda: self._find_faces(path))
        if self.config.deepfake:
            self._step(
                report, "deepfake_image", "image", lambda: self.deepfake_detector.analyze_image(path)
            )

    def _run_video(self, report: Report, path: str) -> None:
        data = self._step(
            report,
            "video_features",
            "video",
            lambda: self.video_extractor.analyze(path, self.config.keyframes_dir),
        )
        if data and self.config.reference_faces_dir:
            self._step(
                report,
                "face_identities",
                "video",
                lambda: self._identify(data["face_embeddings"]),
            )
        if self.config.deepfake and data and data.get("keyframes"):
            self._step(report, "deepfake_frames", "video", lambda: self._deepfake_frames(path, data))
        audio = self._step(
            report,
            "audio_features",
            "video",
            lambda: self.audio_extractor.analyze(path, self.config.spectrograms_dir),
        )
        if self.config.deepfake:
            self._step(
                report, "deepfake_audio", "video", lambda: self.deepfake_detector.analyze_audio(path)
            )
        text, source_type = _video_text((data or {}).get("ocr_text"), (audio or {}).get("text"))
        self._text_steps(report, "video", text, source_type)

    def _run_audio(self, report: Report, path: str) -> None:
        audio = self._step(
            report,
            "audio_features",
            "audio",
            lambda: self.audio_extractor.analyze(path, self.config.spectrograms_dir),
        )
        if self.config.deepfake:
            self._step(
                report, "deepfake_audio", "audio", lambda: self.deepfake_detector.analyze_audio(path)
            )
        self._text_steps(report, "audio", (audio or {}).get("text"), "transcript")

    def _deepfake_frames(self, video_path: str, video: Dict[str, Any]) -> Dict[str, Any]:
        paths = [frame["file_path"] for frame in video["keyframes"]]
        return {"file": Path(video_path).name, **self.deepfake_detector.analyze_images(paths)}

    def _read_image_text(self, path: str) -> Dict[str, Any]:
        lines = self.ocr.extract_with_details(path, min_confidence=self.config.min_ocr_confidence)
        return {"text": "\n".join(line["text"] for line in lines), "lines": lines}

    def _find_faces(self, path: str) -> List[Dict[str, Any]]:
        database = self.face_database
        if database is None:
            return self.faces.extract_faces(path)
        return self.faces.identify_faces(path, database, self.config.face_threshold)

    def _identify(self, embeddings: Sequence[Sequence[float]]) -> List[Optional[Dict[str, Any]]]:
        database = self.face_database
        return [database.identify(list(e), self.config.face_threshold) for e in embeddings]
