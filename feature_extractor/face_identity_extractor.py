import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from typing import Dict, List, Optional, Sequence, Tuple, TypeAlias, TypedDict

from .model_cache import INSIGHTFACE, model_dir

EmbeddingVector: TypeAlias = List[float]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FaceDict(TypedDict):
    bbox: List[float]  # x1, y1, x2, y2 w pikselach
    det_score: float
    embedding: EmbeddingVector  # ArcFace 512-d, znormalizowany


class IdentityMatch(TypedDict):
    name: str
    similarity: float


class FaceMatchPair(TypedDict):
    key_a: str
    key_b: str
    similarity: float


class ReferenceFaceDatabase:
    """
    Mała lokalna baza osób: dla każdej jedna uśredniona, znormalizowana sygnatura.
    Kilka wyraźnych, zróżnicowanych zdjęć na osobę zwykle wystarcza.
    """

    def __init__(self) -> None:
        self._signatures: Dict[str, np.ndarray] = {}

    @property
    def names(self) -> List[str]:
        return sorted(self._signatures)

    def __len__(self) -> int:
        return len(self._signatures)

    def add_person(self, name: str, embeddings: Sequence[EmbeddingVector]) -> None:
        if not embeddings:
            raise ValueError(f"Brak embeddingów dla osoby: {name}")
        mean = np.mean(np.asarray(embeddings, dtype=np.float64), axis=0)
        norm = np.linalg.norm(mean)
        if norm == 0:
            raise ValueError(f"Zerowy wektor średni dla osoby: {name}")
        self._signatures[name] = mean / norm

    def identify(
        self, embedding: EmbeddingVector, threshold: float = 0.45
    ) -> Optional[IdentityMatch]:
        """
        Najlepiej pasująca osoba albo None, gdy nikt nie przekracza progu. 0.45 to punkt startowy
        dla ArcFace na cosinusie - warto skalibrować na własnych danych (niżej więcej fałszywych
        rozpoznań, wyżej giną ujęcia z profilu i przy słabym świetle).
        """
        if not self._signatures:
            return None

        query = np.asarray(embedding, dtype=np.float64)
        query = query / np.linalg.norm(query)
        best_name, best_sim = max(
            ((name, float(np.dot(query, sig))) for name, sig in self._signatures.items()),
            key=lambda pair: pair[1],
        )
        if best_sim < threshold:
            return None
        return {"name": best_name, "similarity": round(best_sim, 4)}


class FaceIdentityExtractor:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        use_gpu: bool = True,
        det_size: Tuple[int, int] = (640, 640),
        min_det_score: float = 0.5,
        models_dir: Optional[str] = None,
    ):
        """
        Detekcja twarzy i embeddingi ArcFace przez InsightFace. Większe det_size wykrywa
        mniejsze twarze, ale działa wolniej; GPU wymaga onnxruntime-gpu. Brakujące modele
        pobierają się do models_dir przy pierwszym uruchomieniu.

        Licencja: modele InsightFace (w tym buffalo_l) są tylko do użytku niekomercyjnego.
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.det_size = det_size
        self.min_det_score = min_det_score

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(
            name=model_name,
            root=str(model_dir(INSIGHTFACE, models_dir)),
            providers=providers,
        )
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)

    @staticmethod
    def _load_bgr(image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Nie znaleziono pliku pod ścieżką: {image_path}")
        data = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None  # InsightFace oczekuje BGR
        if image is None:
            raise ValueError(f"Nie udało się odczytać obrazu: {image_path}")
        return image

    def extract_faces(self, image_path: str) -> List[FaceDict]:
        """Twarze od największej do najmniejszej (największa to zwykle główna postać reklamy)."""
        image = self._load_bgr(image_path)

        faces: List[FaceDict] = []
        for face in self.app.get(image):
            score = float(face.det_score)
            if score < self.min_det_score:
                continue
            faces.append(
                {
                    "bbox": [float(v) for v in face.bbox],
                    "det_score": round(score, 4),
                    "embedding": [float(v) for v in face.normed_embedding],
                }
            )

        faces.sort(
            key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
            reverse=True,
        )
        return faces

    def build_reference_database(self, root_dir: str) -> ReferenceFaceDatabase:
        """
        Buduje bazę z katalogu w układzie root_dir/Donald_Tusk/1.jpg, root_dir/Elon_Musk/a.png...
        Z każdego zdjęcia bierze największą twarz. Zdjęcia bez twarzy są pomijane,
        a osoba, której nie udało się wykryć nigdzie, nie trafia do bazy.
        """
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Nie znaleziono katalogu: {root_dir}")

        database = ReferenceFaceDatabase()
        for person in sorted(os.listdir(root_dir)):
            person_dir = os.path.join(root_dir, person)
            if not os.path.isdir(person_dir):
                continue

            embeddings: List[EmbeddingVector] = []
            for file_name in sorted(os.listdir(person_dir)):
                if os.path.splitext(file_name)[1].lower() not in IMAGE_EXTENSIONS:
                    continue
                faces = self.extract_faces(os.path.join(person_dir, file_name))
                if faces:
                    embeddings.append(faces[0]["embedding"])

            if embeddings:
                database.add_person(person.replace("_", " "), embeddings)
        return database

    def identify_faces(
        self,
        image_path: str,
        database: ReferenceFaceDatabase,
        threshold: float = 0.45,
    ) -> List[Dict]:
        """Twarze z obrazu, każda z polem 'identity' (IdentityMatch albo None)."""
        return [
            {**face, "identity": database.identify(face["embedding"], threshold)}
            for face in self.extract_faces(image_path)
        ]

    @staticmethod
    def find_matching_pairs(
        embeddings_by_key: Dict[str, Sequence[EmbeddingVector]],
        threshold: float = 0.5,
    ) -> List[FaceMatchPair]:
        """
        Porównuje twarze nieznane między sobą, bez bazy referencyjnej: wskazuje incydenty,
        w których pojawia się ta sama osoba (np. wygenerowany "lekarz" w wielu reklamach).
        Wejście to {id_incydentu: [embedding, ...]}, wynik to pary posortowane malejąco.
        """
        keys = sorted(embeddings_by_key)
        pairs: List[FaceMatchPair] = []
        for i, key_a in enumerate(keys):
            faces_a = np.asarray(embeddings_by_key[key_a], dtype=np.float64)
            if faces_a.size == 0:
                continue
            for key_b in keys[i + 1 :]:
                faces_b = np.asarray(embeddings_by_key[key_b], dtype=np.float64)
                if faces_b.size == 0:
                    continue
                best = float(np.max(faces_a @ faces_b.T))
                if best >= threshold:
                    pairs.append({"key_a": key_a, "key_b": key_b, "similarity": round(best, 4)})

        pairs.sort(key=lambda p: p["similarity"], reverse=True)
        return pairs
