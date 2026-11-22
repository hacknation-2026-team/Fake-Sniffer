import json
import numpy as np
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias, TypedDict

os.environ["MPLBACKEND"] = "Agg"

from matplotlib.figure import Figure  # noqa: E402

from .speech_extractor import SAMPLE_RATE, SILENCE_PEAK, load_audio, sha256_file

EmbeddingVector: TypeAlias = List[float]
SpectrogramKind: TypeAlias = Literal["mel", "linear"]


class SpectrogramFeatures(TypedDict):
    embedding: EmbeddingVector
    silence_ratio: float
    spectral_centroid_hz_mean: float
    spectral_centroid_hz_std: float
    spectral_flatness_mean: float
    dynamic_range_db: float


class SpectrogramRecord(TypedDict):
    source: Dict[str, Any]
    status: str
    params: Optional[Dict[str, Any]]
    shape: Optional[List[int]]
    duration_s: float
    features: Optional[SpectrogramFeatures]
    files: Dict[str, str]
    processing_seconds: float


@dataclass(frozen=True)
class Spectrogram:
    power_db: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    features: SpectrogramFeatures


def _hz_to_mel(freq: Any) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: Any) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


class SpectrogramExtractor:
    def __init__(
        self,
        kind: SpectrogramKind = "mel",
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        top_db: float = 80.0,
    ):
        if kind not in ("mel", "linear"):
            raise ValueError(f"Nieznany rodzaj spektrogramu: {kind!r} (mel, linear)")
        if win_length > n_fft:
            raise ValueError("win_length nie może być większe niż n_fft")

        self.kind = kind
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.top_db = top_db

    def params(self, sample_rate: int = SAMPLE_RATE) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": float(self.fmax if self.fmax is not None else sample_rate / 2),
            "top_db": self.top_db,
        }

    def compute(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Spectrogram:
        fmax = self.fmax if self.fmax is not None else sample_rate / 2
        power = self._power_spectrogram(audio)
        fft_freqs = np.linspace(0, sample_rate / 2, self.n_fft // 2 + 1).astype(np.float32)

        if self.kind == "mel":
            filters, freqs = self._mel_filterbank(sample_rate, fmax)
            spectrum = filters @ power
        else:
            band = (fft_freqs >= self.fmin) & (fft_freqs <= fmax)
            spectrum, freqs = power[band], fft_freqs[band]

        power_db = self._to_db(spectrum)
        times = (np.arange(power_db.shape[1]) * self.hop_length / sample_rate).astype(np.float32)
        return Spectrogram(
            power_db=power_db,
            freqs=freqs,
            times=times,
            features=self._features(power, power_db, fft_freqs),
        )

    def extract(
        self,
        media_path: str,
        output_dir: Optional[str] = None,
        save_png: bool = True,
        save_npz: bool = True,
    ) -> SpectrogramRecord:
        started = time.perf_counter()
        audio, meta = load_audio(media_path)
        sha = sha256_file(media_path)
        record: SpectrogramRecord = {
            "source": {
                "path": str(media_path),
                "sha256": sha,
                "size_bytes": os.path.getsize(media_path),
                **meta,
            },
            "status": "ok",
            "params": None,
            "shape": None,
            "duration_s": 0.0,
            "features": None,
            "files": {},
            "processing_seconds": 0.0,
        }

        if audio is None:
            record["status"] = "no_audio"
        elif len(audio) < SAMPLE_RATE * 0.1 or float(np.abs(audio).max()) < SILENCE_PEAK:
            record["status"] = "silence"
        else:
            spectrogram = self.compute(audio)
            record["params"] = self.params()
            record["shape"] = list(spectrogram.power_db.shape)
            record["duration_s"] = round(len(audio) / SAMPLE_RATE, 3)
            record["features"] = spectrogram.features

            if output_dir is not None and (save_png or save_npz):
                os.makedirs(output_dir, exist_ok=True)
                file_name = os.path.basename(media_path)
                base_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_{sha[:8]}")
                if save_npz:
                    record["files"]["npz"] = self._save_npz(f"{base_path}.npz", spectrogram)
                if save_png:
                    record["files"]["png"] = self._save_png(
                        f"{base_path}.png", spectrogram, f"{file_name} ({self.kind})"
                    )

        record["processing_seconds"] = round(time.perf_counter() - started, 3)
        return record

    def extract_embedding(self, media_path: str) -> Optional[EmbeddingVector]:
        features = self.extract(media_path)["features"]
        return features["embedding"] if features is not None else None

    @staticmethod
    def cosine_similarity(vec_a: EmbeddingVector, vec_b: EmbeddingVector) -> float:
        a, b = np.asarray(vec_a, dtype=np.float64), np.asarray(vec_b, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(f"Różne wymiary embeddingów: {a.shape} vs {b.shape}")
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    def _mel_filterbank(self, sample_rate: int, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
        if not 0 <= self.fmin < fmax <= sample_rate / 2:
            raise ValueError(
                f"Nieprawidłowy zakres częstotliwości: fmin={self.fmin}, fmax={fmax}, sr={sample_rate}"
            )
        fft_freqs = np.linspace(0, sample_rate / 2, self.n_fft // 2 + 1)
        edges = _mel_to_hz(np.linspace(_hz_to_mel(self.fmin), _hz_to_mel(fmax), self.n_mels + 2))
        lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
        rising = (fft_freqs - lower) / (center - lower)
        falling = (upper - fft_freqs) / (upper - center)
        filters = np.maximum(0.0, np.minimum(rising, falling)) * (2.0 / (upper - lower))
        return filters.astype(np.float32), edges[1:-1].astype(np.float32)

    def _power_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        pad = self.n_fft // 2
        mode = "reflect" if len(audio) > pad else "constant"
        padded = np.pad(audio.astype(np.float32), pad, mode=mode)
        n_frames = 1 + (len(padded) - self.n_fft) // self.hop_length
        frames = np.lib.stride_tricks.sliding_window_view(padded, self.n_fft)[:: self.hop_length][:n_frames]
        window = np.zeros(self.n_fft, np.float32)
        offset = (self.n_fft - self.win_length) // 2
        window[offset : offset + self.win_length] = np.hanning(self.win_length + 1)[:-1]
        spectrum = np.fft.rfft(frames * window, axis=1)
        return (np.abs(spectrum) ** 2).T.astype(np.float32)

    def _to_db(self, power: np.ndarray) -> np.ndarray:
        db = 10.0 * np.log10(np.maximum(power, 1e-10))
        db -= db.max()
        return np.maximum(db, -self.top_db).astype(np.float32)

    @staticmethod
    def _features(power: np.ndarray, power_db: np.ndarray, fft_freqs: np.ndarray) -> SpectrogramFeatures:
        frame_energy = power.sum(axis=0)
        energy_db = 10 * np.log10(np.maximum(frame_energy, 1e-10))
        active = energy_db > energy_db.max() - 40.0
        centroid = (fft_freqs[:, None] * power).sum(axis=0) / np.maximum(frame_energy, 1e-10)
        flatness = np.exp(np.mean(np.log(np.maximum(power, 1e-10)), axis=0)) / np.maximum(
            power.mean(axis=0), 1e-10
        )

        voiced = power_db[:, active] if active.any() else power_db
        delta = np.diff(voiced, axis=1) if voiced.shape[1] > 1 else np.zeros_like(voiced)
        embedding = np.concatenate([voiced.std(axis=1), delta.std(axis=1)])
        embedding -= embedding.mean()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        has_active = bool(active.any())
        return {
            "embedding": [round(float(v), 5) for v in embedding],
            "silence_ratio": round(float(1 - active.mean()), 4),
            "spectral_centroid_hz_mean": round(float(centroid[active].mean()), 1) if has_active else 0.0,
            "spectral_centroid_hz_std": round(float(centroid[active].std()), 1) if has_active else 0.0,
            "spectral_flatness_mean": round(float(flatness[active].mean()), 5) if has_active else 0.0,
            "dynamic_range_db": (
                round(float(energy_db[active].max() - energy_db[active].min()), 2) if has_active else 0.0
            ),
        }

    def _save_npz(self, path: str, spectrogram: Spectrogram) -> str:
        np.savez_compressed(
            path,
            power_db=spectrogram.power_db,
            freqs=spectrogram.freqs,
            times=spectrogram.times,
            params=json.dumps(self.params()),
        )
        return path

    def _save_png(self, path: str, spectrogram: Spectrogram, title: str) -> str:
        power_db, freqs, times = spectrogram.power_db, spectrogram.freqs, spectrogram.times
        duration = float(times[-1]) if len(times) else 0.0

        figure = Figure(figsize=(10, 3.6), dpi=110)
        ax = figure.subplots()
        image = ax.imshow(
            power_db,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[0, max(duration, 1e-3), 0, power_db.shape[0]],
            vmin=-self.top_db,
            vmax=0,
        )
        ticks = np.linspace(0, power_db.shape[0] - 1, 6).astype(int)
        ax.set_yticks(ticks + 0.5)
        ax.set_yticklabels(
            [f"{freqs[i] / 1000:.1f}k" if freqs[i] >= 1000 else f"{freqs[i]:.0f}" for i in ticks]
        )
        ax.set_xlabel("czas [s]")
        ax.set_ylabel("częstotliwość [Hz]")
        ax.set_title(title, fontsize=10)
        figure.colorbar(image, ax=ax, format="%+.0f dB", pad=0.01)
        figure.tight_layout()
        figure.savefig(path)
        return path
