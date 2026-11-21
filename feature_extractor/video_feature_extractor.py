from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

if TYPE_CHECKING: 
    from .face_identity_extractor import FaceIdentityExtractor
    from .keyframe_extractor import LightweightKeyframeExtractor
    from .ocr_text_extractor import TextExtractor
    from .visual_feature_extractor import ImageHasher, VisualEmbedder


class VideoFeatureExtractor:
    def __init__(
        self,
        keyframes: LightweightKeyframeExtractor,
        hasher: ImageHasher,
        ocr: Optional[TextExtractor] = None,
        faces: Optional[FaceIdentityExtractor] = None,
        embedder: Optional[VisualEmbedder] = None,
    ):
        """
        Zbiera cechy filmu z jego klatek kluczowych. Hashe są obowiązkowe, reszta opcjonalna:
        brak OCR/twarzy/CLIP-a po prostu oznacza puste pole w wyniku.
        """
        self.keyframes = keyframes
        self.hasher = hasher
        self.ocr = ocr
        self.faces = faces
        self.embedder = embedder

    def analyze(self, video_path: str, output_dir: str) -> Dict:
        frames = self.keyframes.extract(video_path, output_dir)

        ocr_lines: List[str] = []
        face_embeddings: List[List[float]] = []
        for frame in frames:
            path = frame["file_path"]
            frame["hashes"] = self.hasher.compute(path)

            if self.ocr is not None:
                frame["ocr_lines"] = self.ocr.extract_text(path, join_lines=False)
                ocr_lines.extend(frame["ocr_lines"])
            if self.faces is not None:
                frame_faces = self.faces.extract_faces(path)
                frame["faces"] = frame_faces
                face_embeddings.extend(f["embedding"] for f in frame_faces)

        return {
            "video": video_path,
            "keyframes": frames,
            "dhashes": [f["hashes"]["dhash"] for f in frames],
            "ocr_text": "\n".join(self._unique_lines(ocr_lines)),
            "face_embeddings": face_embeddings,
            "visual_embedding": self._mean_embedding(frames),
        }

    def _mean_embedding(self, frames: Sequence[Dict]) -> Optional[List[float]]:
        """Jeden wektor CLIP na film: średnia z klatek, znormalizowana."""
        if self.embedder is None or not frames:
            return None
        vectors = np.asarray(
            self.embedder.extract_embeddings([f["file_path"] for f in frames]), dtype=np.float64
        )
        mean = vectors.mean(axis=0)
        norm = np.linalg.norm(mean)
        return [float(x) for x in (mean / norm if norm else mean)]

    @staticmethod
    def _unique_lines(lines: Sequence[str]) -> List[str]:
        """Napisy i CTA powtarzają się w kolejnych klatkach - zostawiamy pierwsze wystąpienie."""
        seen, unique = set(), []
        for line in lines:
            key = " ".join(line.split()).casefold()
            if key and key not in seen:
                seen.add(key)
                unique.append(line.strip())
        return unique


def shared_scene_ratio(dhashes_a: Sequence[str], dhashes_b: Sequence[str], max_distance: int = 8) -> float:
    """
    Jaka część ujęć z krótszego filmu ma odpowiednik w drugim (odległość Hamminga dHash <= max_distance).
    Wychwytuje przemontowania, zmianę kolejności i dołożone wstawki, na których porównanie
    całych filmów zawodzi. Zwraca 0.0, gdy któryś film nie ma klatek.
    """
    from .visual_feature_extractor import ImageHasher

    if not dhashes_a or not dhashes_b:
        return 0.0
    shorter, longer = sorted((dhashes_a, dhashes_b), key=len)
    matched = sum(
        any(ImageHasher.hamming_distance(h, other) <= max_distance for other in longer)
        for h in shorter
    )
    return matched / len(shorter)
