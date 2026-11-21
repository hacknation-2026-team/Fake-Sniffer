import cv2
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch

from feature_extractor.model_cache import EASYOCR, model_dir
from feature_extractor.ocr_text_extractor import TextExtractor


@pytest.fixture
def mock_reader():
    """Podmienia easyocr.Reader na atrapę, aby testy nie ładowały sieci neuronowej."""
    with patch("feature_extractor.ocr_text_extractor.easyocr.Reader") as mock_reader_cls:
        mock_instance = MagicMock()
        mock_reader_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_image(tmp_path: Path) -> str:
    """Tworzy tymczasowy plik udający obrazek (automatycznie usuwany po teście)."""
    img_file = tmp_path / "test_image.png"
    img_file.write_bytes(b"fake_image_data")
    return str(img_file)


# --- TESTY INICJALIZACJI I WALIDACJI ---

def test_extractor_initialization():
    """Sprawdza przekazanie parametrów languages i use_gpu do easyocr.Reader."""
    with patch("feature_extractor.ocr_text_extractor.easyocr.Reader") as mock_cls:
        extractor = TextExtractor(languages=["pl", "de"], use_gpu=False)
        
        assert extractor.languages == ["pl", "de"]
        assert extractor.use_gpu is False
        mock_cls.assert_called_once_with(
            ["pl", "de"], gpu=False, model_storage_directory=str(model_dir(EASYOCR))
        )


def test_models_are_stored_in_requested_directory(tmp_path):
    target = tmp_path / "moje_modele"
    with patch("feature_extractor.ocr_text_extractor.easyocr.Reader") as mock_cls:
        TextExtractor(use_gpu=False, models_dir=str(target))

    assert mock_cls.call_args[1]["model_storage_directory"] == str(target / "easyocr")
    assert (target / "easyocr").is_dir()


def test_file_not_found_raises_error(mock_reader):
    """Sprawdza, czy obie metody rzucają FileNotFoundError dla nieistniejącego pliku."""
    extractor = TextExtractor()
    
    with pytest.raises(FileNotFoundError, match="brak_pliku.jpg"):
        extractor.extract_text("brak_pliku.jpg")
        
    with pytest.raises(FileNotFoundError, match="brak_pliku.jpg"):
        extractor.extract_with_details("brak_pliku.jpg")


# --- TESTY METODY extract_text ---

@pytest.mark.parametrize("join_lines, expected_output", [
    (True, "Linia 1\nLinia 2\nLinia 3"),
    (False, ["Linia 1", "Linia 2", "Linia 3"]),
])
def test_extract_text(mock_reader, sample_image, join_lines, expected_output):
    """Testuje zwracanie tekstu jako połączony string lub lista linii."""
    mock_reader.readtext.return_value = ["Linia 1", "Linia 2", "Linia 3"]
    extractor = TextExtractor()

    result = extractor.extract_text(sample_image, join_lines=join_lines)

    mock_reader.readtext.assert_called_once_with(b"fake_image_data", detail=0)
    assert result == expected_output


def test_webp_from_path_with_polish_characters_is_passed_as_bytes(mock_reader, tmp_path):
    path = tmp_path / "zażółć gęślą" / "reklama.webp"
    path.parent.mkdir()
    Image.new("RGB", (32, 16), (255, 255, 255)).save(path, "WEBP")
    mock_reader.readtext.return_value = ["Zarób 5000 zł"]

    text = TextExtractor().extract_text(str(path))

    image = mock_reader.readtext.call_args[0][0]
    assert text == "Zarób 5000 zł"
    assert image == path.read_bytes()
    assert cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR).shape == (16, 32, 3)


# --- TESTY METODY extract_with_details ---

def test_extract_with_details_structure_and_types(mock_reader, sample_image):
    """
    Sprawdza strukturę zwracanego słownika oraz rzutowanie typów numpy
    (np.int32, np.float64) na czyste pythonowe floaty.
    """
    mock_reader.readtext.return_value = [
        (
            [
                [np.int32(10), np.int32(20)],
                [np.int32(100), np.int32(20)],
                [np.int32(100), np.int32(50)],
                [np.int32(10), np.int32(50)],
            ],
            "PARAGON FISKALNY",
            np.float64(0.9876543)
        )
    ]

    extractor = TextExtractor()
    results = extractor.extract_with_details(sample_image)

    mock_reader.readtext.assert_called_once_with(b"fake_image_data", detail=1)
    assert len(results) == 1
    
    item = results[0]
    assert item["text"] == "PARAGON FISKALNY"
    assert item["confidence"] == 0.9877
    assert isinstance(item["confidence"], float)
    
    # Sprawdzenie współrzędnych i czystych typów float wewnątrz bbox
    assert item["bbox"] == [[10.0, 20.0], [100.0, 20.0], [100.0, 50.0], [10.0, 50.0]]
    assert all(isinstance(coord, float) for point in item["bbox"] for coord in point)


@pytest.mark.parametrize("min_conf, expected_texts", [
    (0.0, ["Wysoka", "Średnia", "Niska"]),
    (0.5, ["Wysoka", "Średnia"]),
    (0.85, ["Wysoka"]),
    (0.99, []),
])
def test_extract_with_details_confidence_filtering(
    mock_reader, sample_image, min_conf, expected_texts
):
    """Sprawdza filtrowanie wyników na podstawie progu min_confidence."""
    dummy_box = [[0, 0], [10, 0], [10, 10], [0, 10]]
    mock_reader.readtext.return_value = [
        (dummy_box, "Wysoka", 0.92),
        (dummy_box, "Średnia", 0.60),
        (dummy_box, "Niska", 0.25),
    ]

    extractor = TextExtractor()
    results = extractor.extract_with_details(sample_image, min_confidence=min_conf)

    assert [r["text"] for r in results] == expected_texts

