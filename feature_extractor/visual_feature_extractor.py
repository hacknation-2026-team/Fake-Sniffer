import imagehash
import numpy as np
import os
from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import List, Sequence, TypeAlias, TypedDict

EmbeddingVector: TypeAlias = List[float]


class PerceptualHashes(TypedDict):
    phash: str
    dhash: str
    ahash: str


def _open_rgb(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Nie znaleziono pliku pod ścieżką: {image_path}")
    
    return Image.open(image_path).convert("RGB")


class ImageHasher:
    def __init__(self, hash_size: int = 8):
        """
        Hashe percepcyjne łapią duplikaty, kadrowania, rekompresję i lekkie zmiany kolorów.
        Hash ma hash_size^2 bitów: większy jest czulszy na szczegóły, ale mniej odporny na zmiany.
        """
        self.hash_size = hash_size

    @property
    def n_bits(self) -> int:
        return self.hash_size * self.hash_size

    def compute(self, image_path: str) -> PerceptualHashes:
        """pHash, dHash i aHash jako ciągi szesnastkowe (wygodne do JSON-a i bazy)."""
        image = _open_rgb(image_path)
        return {
            "phash": str(imagehash.phash(image, hash_size=self.hash_size)),
            "dhash": str(imagehash.dhash(image, hash_size=self.hash_size)),
            "ahash": str(imagehash.average_hash(image, hash_size=self.hash_size)),
        }

    @staticmethod
    def hamming_distance(hash_a: str, hash_b: str) -> int:
        if len(hash_a) != len(hash_b):
            raise ValueError("Hashe mają różną długość - policzono je z innym hash_size.")
        return int(imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b))

    def similarity(self, hash_a: str, hash_b: str) -> float:
        """1 - odległość Hamminga / liczba bitów, czyli 0.0-1.0."""
        return 1.0 - self.hamming_distance(hash_a, hash_b) / self.n_bits

    def is_near_duplicate(
        self,
        hashes_a: PerceptualHashes,
        hashes_b: PerceptualHashes,
        max_distance: int = 6,
    ) -> bool:
        """
        Duplikat, gdy zarówno pHash, jak i dHash mieszczą się w progu (dwa hashe naraz
        ograniczają fałszywe trafienia). Próg 6 jest dobrany pod hash_size=8.
        """
        return (
            self.hamming_distance(hashes_a["phash"], hashes_b["phash"]) <= max_distance
            and self.hamming_distance(hashes_a["dhash"], hashes_b["dhash"]) <= max_distance
        )


class VisualEmbedder:
    def __init__(self, model_name: str = "clip-ViT-B-32", device: str | None = None):
        """
        Embeddingi CLIP przez sentence-transformers. Łapią wspólny szablon reklamy
        (np. fałszywy baner TVP Info), nawet gdy zdjęcie w środku jest inne.
        SigLIP trzeba ładować inaczej (przez transformers), tu go nie ma.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def extract_embedding(self, image_path: str) -> EmbeddingVector:
        return self.extract_embeddings([image_path])[0]

    def extract_embeddings(
        self, image_paths: Sequence[str], batch_size: int = 16
    ) -> List[EmbeddingVector]:
        """Znormalizowane wektory w kolejności podanych ścieżek."""
        if not image_paths:
            return []

        images = [_open_rgb(p) for p in image_paths]
        vectors: np.ndarray = self.model.encode(
            images,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(x) for x in vec] for vec in vectors]

    @staticmethod
    def cosine_similarity(vec_a: EmbeddingVector, vec_b: EmbeddingVector) -> float:
        a, b = np.asarray(vec_a, dtype=np.float64), np.asarray(vec_b, dtype=np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0
