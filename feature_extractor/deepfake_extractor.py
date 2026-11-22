import os
import re
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from .model_cache import HUGGINGFACE, model_dir
from .speech_extractor import SAMPLE_RATE, load_audio

IMAGE_MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"
AUDIO_MODEL = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"

FAKE_TOKENS = {"fake", "deepfake", "spoof", "spoofed", "synthetic", "generated", "ai"}
REAL_TOKENS = {"real", "realism", "bonafide", "human", "genuine", "authentic"}


class DeepfakeScore(TypedDict):
    file: str
    model: str
    label: str
    fake_probability: Optional[float]
    scores: Dict[str, float]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _tokens(label: str) -> set:
    return set(re.findall(r"[a-z]+", label.lower()))


def find_fake_label(labels: Sequence[str], preferred: Optional[str] = None) -> Optional[str]:
    """
    Etykieta oznaczająca fałszerstwo. Modele nazywają ją różnie (Deepfake, fake, spoof),
    a bywa że tylko LABEL_0/LABEL_1. Wtedy zwraca None, a wynik podaje same prawdopodobieństwa.
    """
    if preferred:
        return preferred if preferred in labels else None
    fakes = [l for l in labels if _tokens(l) & FAKE_TOKENS and not _tokens(l) & REAL_TOKENS]
    return fakes[0] if len(fakes) == 1 else None


class DeepfakeDetector:
    def __init__(
        self,
        image_model: str = IMAGE_MODEL,
        audio_model: str = AUDIO_MODEL,
        device: str = "cpu",
        models_dir: Optional[str] = None,
        audio_fake_label: Optional[str] = None,
    ):
        """
        Gotowe klasyfikatory z Hugging Face: obraz (ViT) i głos (wav2vec2). Wynik to wskaźnik,
        nie dowód: modele uczono na konkretnych zbiorach i na nowych generatorach mylą się częściej.
        Modele ładują się dopiero przy pierwszym użyciu i trafiają do wspólnego katalogu modeli.
        """
        self.image_model_name = image_model
        self.audio_model_name = audio_model
        self.device = device
        self.models_dir = models_dir
        self.audio_fake_label = audio_fake_label
        self._image: Optional[Tuple[Any, Any]] = None
        self._audio: Optional[Tuple[Any, Any]] = None

    def _cache_dir(self) -> str:
        return str(model_dir(HUGGINGFACE, self.models_dir))

    def load_image_model(self) -> Tuple[Any, Any]:
        if self._image is None:
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            cache = self._cache_dir()
            processor = AutoImageProcessor.from_pretrained(self.image_model_name, cache_dir=cache)
            model = AutoModelForImageClassification.from_pretrained(self.image_model_name, cache_dir=cache)
            self._image = (processor, model.to(self.device).eval())
        return self._image

    def load_audio_model(self) -> Tuple[Any, Any]:
        if self._audio is None:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            cache = self._cache_dir()
            extractor = AutoFeatureExtractor.from_pretrained(self.audio_model_name, cache_dir=cache)
            model = AutoModelForAudioClassification.from_pretrained(self.audio_model_name, cache_dir=cache)
            self._audio = (extractor, model.to(self.device).eval())
        return self._audio

    def _logits(self, model: Any, inputs: Any) -> np.ndarray:
        import torch

        batch = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            return model(**batch).logits[0].detach().cpu().numpy()

    def _score(
        self, file: str, model_name: str, model: Any, logits: np.ndarray, preferred: Optional[str] = None
    ) -> DeepfakeScore:
        id2label = model.config.id2label
        probabilities = _softmax(logits)
        scores = {str(id2label[i]): round(float(p), 4) for i, p in enumerate(probabilities)}
        fake = find_fake_label(list(scores), preferred)
        return {
            "file": file,
            "model": model_name,
            "label": max(scores, key=scores.get),
            "fake_probability": scores[fake] if fake else None,
            "scores": scores,
        }

    def analyze_image(self, image_path: str) -> DeepfakeScore:
        """Prawdopodobieństwo, że obraz jest wygenerowany lub podmieniony."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Nie znaleziono pliku pod ścieżką: {image_path}")

        processor, model = self.load_image_model()
        with Image.open(image_path) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        logits = self._logits(model, inputs)
        return self._score(os.path.basename(image_path), self.image_model_name, model, logits)

    def analyze_images(self, image_paths: Sequence[str]) -> Dict[str, Any]:
        """Wynik dla każdej klatki oraz średnia i maksimum (jedna podejrzana klatka też się liczy)."""
        frames = [self.analyze_image(path) for path in image_paths]
        probabilities = [f["fake_probability"] for f in frames if f["fake_probability"] is not None]
        return {
            "model": self.image_model_name,
            "frames": frames,
            "mean_fake_probability": round(float(np.mean(probabilities)), 4) if probabilities else None,
            "max_fake_probability": round(float(np.max(probabilities)), 4) if probabilities else None,
        }

    @staticmethod
    def _windows(samples: np.ndarray, window_sec: float, max_windows: int) -> List[Tuple[float, np.ndarray]]:
        size = int(window_sec * SAMPLE_RATE)
        starts = list(range(0, max(len(samples) - SAMPLE_RATE, 1), size)) or [0]
        if len(starts) > max_windows:
            starts = [starts[i] for i in np.linspace(0, len(starts) - 1, max_windows).astype(int)]
        return [(start / SAMPLE_RATE, samples[start : start + size]) for start in starts]

    def analyze_audio(
        self, media_path: str, window_sec: float = 10.0, max_windows: int = 12
    ) -> Optional[Dict[str, Any]]:
        """
        Fałszywy głos w nagraniu lub ścieżce dźwiękowej filmu. Nagranie jest cięte na okna,
        a przy długich plikach analizowane są równomiernie rozłożone okna. Brak dźwięku daje None.
        """
        samples, _ = load_audio(media_path)
        if samples is None or len(samples) == 0:
            return None

        extractor, model = self.load_audio_model()
        windows = []
        for start, chunk in self._windows(samples, window_sec, max_windows):
            inputs = extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            logits = self._logits(model, inputs)
            score = self._score(os.path.basename(media_path), self.audio_model_name, model, logits, self.audio_fake_label)
            windows.append({"start_sec": round(start, 2), **score})

        probabilities = [w["fake_probability"] for w in windows if w["fake_probability"] is not None]
        return {
            "file": os.path.basename(media_path),
            "model": self.audio_model_name,
            "windows": windows,
            "mean_fake_probability": round(float(np.mean(probabilities)), 4) if probabilities else None,
            "max_fake_probability": round(float(np.max(probabilities)), 4) if probabilities else None,
            "labels": sorted(windows[0]["scores"]),
        }
