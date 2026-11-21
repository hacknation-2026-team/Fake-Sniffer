import os
from pathlib import Path
from typing import Optional, Union

MODELS_DIR_ENV = "FAKE_SNIFFER_MODELS_DIR"

EASYOCR = "easyocr"
HUGGINGFACE = "huggingface"
INSIGHTFACE = "insightface"
WHISPER = "whisper"
PLAYWRIGHT = "playwright"


def default_models_dir() -> Path:
    """Katalog na modele: wartość FAKE_SNIFFER_MODELS_DIR albo `models/` obok pakietu."""
    configured = os.environ.get(MODELS_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent.parent / "models"


def model_dir(name: str, models_dir: Optional[Union[str, Path]] = None) -> Path:
    """Podkatalog modeli jednej biblioteki. Tworzy go, jeśli nie istnieje."""
    base = Path(models_dir).expanduser() if models_dir else default_models_dir()
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path
