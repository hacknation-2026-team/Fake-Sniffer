import easyocr
import os
from typing import List, Optional, Union, TypedDict, TypeAlias

from .model_cache import EASYOCR, model_dir

# lista 4 punktów [x, y]
Point: TypeAlias = List[Union[int, float]]
BoundingBox: TypeAlias = List[Point]

class OCRResultDict(TypedDict):
    text: str
    confidence: float
    bbox: BoundingBox

class TextExtractor:
    def __init__(
        self,
        languages: List[str] = ['pl', 'en'],
        use_gpu: bool = True,
        models_dir: Optional[str] = None,
    ):
        """
        Inicjalizuje czytnik EasyOCR.

        :param languages: Lista kodów języków do rozpoznawania (domyślnie polski i angielski).
        :param use_gpu: Czy używać karty graficznej (CUDA) do przyspieszenia obliczeń.
        :param models_dir: Gdzie trzymać modele; brakujące pobiera się przy pierwszym uruchomieniu.
        """
        self.languages = languages
        self.use_gpu = use_gpu
        self.reader = easyocr.Reader(
            self.languages,
            gpu=self.use_gpu,
            model_storage_directory=str(model_dir(EASYOCR, models_dir)),
        )

    def _validate_image(self, image_path: str) -> None:
        """Sprawdza, czy plik istnieje przed próbą odczytu."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Nie znaleziono pliku pod ścieżką: {image_path}")

    def extract_text(self, image_path: str, join_lines: bool = True) -> Union[str, List[str]]:
        """
        Odczytuje sam tekst z obrazu.
        
        :param image_path: Ścieżka do pliku graficznego.
        :param join_lines: Jeśli True, zwraca jeden połączony ciąg znaków. 
                           Jeśli False, zwraca listę wykrytych fragmentów tekstu.
        :return: Wyodrębniony tekst.
        """
        self._validate_image(image_path)
        
        # detail=0 zwraca wyłącznie rozpoznany tekst bez współrzędnych ramek
        results = self.reader.readtext(image_path, detail=0)
        
        if join_lines:
            return "\n".join(results)
        return results

    def extract_with_details(self, image_path: str, min_confidence: float = 0.0) -> List[OCRResultDict]:
        """
        Odczytuje tekst wraz ze współrzędnymi (bounding box) oraz poziomem pewności.
        
        :param image_path: Ścieżka do pliku graficznego.
        :param min_confidence: Minimalny próg pewności (od 0.0 do 1.0), poniżej którego wyniki są odrzucane.
        :return: Lista słowników zawierających tekst, pewność i współrzędne.
        """
        self._validate_image(image_path)
        raw_results = self.reader.readtext(image_path, detail=1)
        
        parsed_results: List[OCRResultDict] = []
        for bbox, text, confidence in raw_results:
            if confidence >= min_confidence:
                # Konwertujemy współrzędne na czyste listy intów/floatów 
                clean_bbox: BoundingBox = [[float(coord) for coord in point] for point in bbox]
                
                parsed_results.append({
                    "text": str(text),
                    "confidence": round(float(confidence), 4),
                    "bbox": clean_bbox
                })
                
        return parsed_results