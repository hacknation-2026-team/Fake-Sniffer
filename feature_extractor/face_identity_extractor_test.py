import numpy as np
import pytest
import cv2
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from face_identity_extractor import FaceIdentityExtractor, ReferenceFaceDatabase


def unit(*values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def fake_face(bbox, score, embedding):
    return SimpleNamespace(
        bbox=np.asarray(bbox, dtype=np.float32),
        det_score=np.float32(score),
        normed_embedding=embedding,
    )


@pytest.fixture
def mock_app():
    """Podmienia insightface FaceAnalysis atrapą - bez pobierania modeli buffalo_l."""
    with patch("face_identity_extractor.FaceAnalysis") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield mock_cls, instance


@pytest.fixture
def sample_image(tmp_path: Path) -> str:
    path = tmp_path / "ad.png"
    cv2.imwrite(str(path), np.zeros((64, 64, 3), dtype=np.uint8))
    return str(path)


# --- INICJALIZACJA ---

@pytest.mark.parametrize(
    "use_gpu, providers, ctx_id",
    [
        (False, ["CPUExecutionProvider"], -1),
        (True, ["CUDAExecutionProvider", "CPUExecutionProvider"], 0),
    ],
)
def test_initialization_passes_providers_and_ctx(mock_app, use_gpu, providers, ctx_id):
    mock_cls, instance = mock_app
    FaceIdentityExtractor(use_gpu=use_gpu, det_size=(320, 320))

    mock_cls.assert_called_once_with(name="buffalo_l", providers=providers)
    instance.prepare.assert_called_once_with(ctx_id=ctx_id, det_size=(320, 320))


# --- extract_faces ---

def test_extract_faces_types_filtering_and_sorting(mock_app, sample_image):
    _, instance = mock_app
    instance.get.return_value = [
        fake_face([0, 0, 10, 10], 0.99, unit(1, 0, 0)),       # mała twarz
        fake_face([0, 0, 40, 40], 0.90, unit(0, 1, 0)),       # największa
        fake_face([0, 0, 50, 50], 0.20, unit(0, 0, 1)),       # odrzucona: niska pewność
    ]
    faces = FaceIdentityExtractor(use_gpu=False).extract_faces(sample_image)

    assert len(faces) == 2
    assert faces[0]["bbox"] == [0.0, 0.0, 40.0, 40.0]  # największa pierwsza
    assert faces[0]["det_score"] == 0.9
    assert all(isinstance(x, float) for f in faces for x in f["embedding"] + f["bbox"])
    assert isinstance(faces[0]["det_score"], float)


def test_no_faces_returns_empty_list(mock_app, sample_image):
    _, instance = mock_app
    instance.get.return_value = []
    assert FaceIdentityExtractor(use_gpu=False).extract_faces(sample_image) == []


def test_missing_file_raises(mock_app):
    with pytest.raises(FileNotFoundError, match="brak.jpg"):
        FaceIdentityExtractor(use_gpu=False).extract_faces("brak.jpg")


def test_unreadable_image_raises_value_error(mock_app, tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"to nie jest obraz")
    with pytest.raises(ValueError, match="odczytać"):
        FaceIdentityExtractor(use_gpu=False).extract_faces(str(bad))


# --- ReferenceFaceDatabase ---

def test_database_identifies_best_match_above_threshold():
    db = ReferenceFaceDatabase()
    db.add_person("Donald Tusk", [unit(1, 0, 0).tolist()])
    db.add_person("Elon Musk", [unit(0, 1, 0).tolist()])

    match = db.identify(unit(0.95, 0.1, 0).tolist(), threshold=0.5)
    assert match is not None and match["name"] == "Donald Tusk"
    assert match["similarity"] > 0.9


def test_database_returns_none_below_threshold_or_when_empty():
    db = ReferenceFaceDatabase()
    assert db.identify(unit(1, 0, 0).tolist()) is None

    db.add_person("Donald Tusk", [unit(1, 0, 0).tolist()])
    assert db.identify(unit(0, 1, 0).tolist(), threshold=0.45) is None


def test_database_averages_and_renormalizes_signature():
    db = ReferenceFaceDatabase()
    db.add_person("X", [unit(1, 0).tolist(), unit(0, 1).tolist()])
    # Średnia leży pośrodku, więc obie "skrajne" twarze mają cosinus ~0.707
    match = db.identify(unit(1, 0).tolist(), threshold=0.0)
    assert match["similarity"] == pytest.approx(0.7071, abs=1e-3)


def test_database_rejects_empty_embeddings():
    with pytest.raises(ValueError):
        ReferenceFaceDatabase().add_person("Nikt", [])


# --- budowanie bazy z katalogu i identyfikacja ---

def test_build_reference_database_from_directory(mock_app, tmp_path):
    _, instance = mock_app
    for person, files in {"Donald_Tusk": ["a.jpg", "b.png", "notes.txt"], "Pusty_Folder": ["x.jpg"]}.items():
        (tmp_path / person).mkdir()
        for f in files:
            if f.endswith(("jpg", "png")):
                cv2.imwrite(str(tmp_path / person / f), np.zeros((16, 16, 3), dtype=np.uint8))
            else:
                (tmp_path / person / f).write_text("x")
    (tmp_path / "luzem.jpg").write_bytes(b"ignored")  # plik poza folderem osoby

    # Kolejność wywołań: Donald_Tusk/a.jpg, Donald_Tusk/b.png, Pusty_Folder/x.jpg (bez twarzy)
    results_by_call = [
        [fake_face([0, 0, 10, 10], 0.99, unit(1, 0, 0))],
        [fake_face([0, 0, 10, 10], 0.99, unit(1, 0.1, 0))],
        [],
    ]
    instance.get.side_effect = results_by_call

    db = FaceIdentityExtractor(use_gpu=False).build_reference_database(str(tmp_path))

    assert db.names == ["Donald Tusk"]  # podkreślenia -> spacje; osoba bez twarzy pominięta


def test_build_reference_database_missing_dir_raises(mock_app):
    with pytest.raises(FileNotFoundError):
        FaceIdentityExtractor(use_gpu=False).build_reference_database("/nie/ma/takiego")


def test_identify_faces_attaches_identity(mock_app, sample_image):
    _, instance = mock_app
    instance.get.return_value = [
        fake_face([0, 0, 40, 40], 0.9, unit(1, 0, 0)),
        fake_face([0, 0, 10, 10], 0.9, unit(0, 0, 1)),
    ]
    db = ReferenceFaceDatabase()
    db.add_person("Donald Tusk", [unit(1, 0, 0).tolist()])

    results = FaceIdentityExtractor(use_gpu=False).identify_faces(sample_image, db)

    assert results[0]["identity"]["name"] == "Donald Tusk"
    assert results[1]["identity"] is None


# --- porównanie twarzy nieznanych ---

def test_find_matching_pairs_links_incidents_sharing_a_face():
    same = unit(1, 0.05, 0).tolist()
    embeddings = {
        "ad_1": [unit(1, 0, 0).tolist()],
        "ad_2": [unit(0, 0, 1).tolist(), same],  # druga twarz = ta sama osoba co w ad_1
        "ad_3": [unit(0, 1, 0).tolist()],
        "ad_4": [],                              # brak twarzy - pomijany
    }
    pairs = FaceIdentityExtractor.find_matching_pairs(embeddings, threshold=0.8)

    assert len(pairs) == 1
    assert (pairs[0]["key_a"], pairs[0]["key_b"]) == ("ad_1", "ad_2")
    assert pairs[0]["similarity"] > 0.99


def test_find_matching_pairs_sorted_descending():
    embeddings = {
        "a": [unit(1, 0).tolist()],
        "b": [unit(1, 0.1).tolist()],
        "c": [unit(1, 0.5).tolist()],
    }
    pairs = FaceIdentityExtractor.find_matching_pairs(embeddings, threshold=0.0)
    sims = [p["similarity"] for p in pairs]
    assert sims == sorted(sims, reverse=True)
    assert len(pairs) == 3
