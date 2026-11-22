from typing import Any, Dict

import numpy as np
from PIL import Image

MAX_SIDE = 256
MAX_ASPECT = 4.0
FLAT_SHARE = 0.9
FEW_COLORS = 48
TRANSPARENT_SHARE = 0.02


def _has_transparency(image: Image.Image) -> float:
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    return float((alpha < 250).mean())


def classify_image(path: str) -> Dict[str, Any]:
    """
    Fotografia czy grafika (logo, ikona, odznaka, ilustracja, pasek reklamowy). Detektory
    deepfake i obrazów AI uczono na zdjęciach, więc na grafice dają przypadkowe wyniki.
    Ocena z samych pikseli: przezroczyste tło, skrajne proporcje, mało kolorów, płaskie plamy.
    """
    with Image.open(path) as image:
        width, height = image.size
        transparent = _has_transparency(image)
        small = image.convert("RGB")
    small.thumbnail((MAX_SIDE, MAX_SIDE))

    pixels = np.asarray(small).reshape(-1, 3) >> 3
    codes = (pixels[:, 0].astype(np.int32) << 10) | (pixels[:, 1].astype(np.int32) << 5) | pixels[:, 2]
    _, counts = np.unique(codes, return_counts=True)
    top_share = float(np.sort(counts)[::-1][:16].sum() / counts.sum())
    aspect = max(width, height) / max(min(width, height), 1)

    reason = None
    if transparent >= TRANSPARENT_SHARE:
        reason = "przezroczyste tło"
    elif aspect > MAX_ASPECT:
        reason = "pasek lub baner"
    elif len(counts) < FEW_COLORS:
        reason = "kilka płaskich kolorów"
    elif top_share >= FLAT_SHARE:
        reason = "płaskie plamy koloru"
    return {
        "photographic": reason is None,
        "reason": reason,
        "colors": int(len(counts)),
        "top_share": round(top_share, 3),
        "aspect": round(aspect, 2),
    }
