import numpy as np
import pytest
from PIL import Image, ImageDraw

from feature_extractor.image_kind import classify_image


def save(tmp_path, image, name="a.png"):
    path = tmp_path / name
    image.save(path)
    return str(path)


def photo_like(size=(400, 300)):
    rng = np.random.default_rng(1)
    base = np.linspace(0, 255, size[0])[None, :, None] * np.ones((size[1], 1, 3))
    noise = rng.normal(0, 40, (size[1], size[0], 3))
    return Image.fromarray(np.clip(base + noise, 0, 255).astype("uint8"))


def test_noisy_photo_is_photographic(tmp_path):
    result = classify_image(save(tmp_path, photo_like(), "p.jpg"))
    assert result["photographic"] is True and result["reason"] is None


def test_flat_logo_on_white_is_graphic(tmp_path):
    image = Image.new("RGB", (300, 300), "white")
    ImageDraw.Draw(image).ellipse((60, 60, 240, 240), fill=(30, 90, 200))
    result = classify_image(save(tmp_path, image))
    assert result["photographic"] is False
    assert result["reason"] in {"kilka płaskich kolorów", "płaskie plamy koloru"}


def test_transparent_png_is_graphic(tmp_path):
    image = photo_like().convert("RGBA")
    image.putalpha(Image.new("L", image.size, 0))
    assert classify_image(save(tmp_path, image))["reason"] == "przezroczyste tło"


def test_wide_strip_is_graphic(tmp_path):
    result = classify_image(save(tmp_path, photo_like((1200, 150)), "b.jpg"))
    assert result["reason"] == "pasek lub baner"


def test_smooth_illustration_with_few_flat_regions_is_graphic(tmp_path):
    image = Image.new("RGB", (400, 300), (250, 240, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 150, 400, 300), fill=(120, 180, 90))
    draw.ellipse((300, 20, 360, 80), fill=(250, 210, 40))
    assert classify_image(save(tmp_path, image))["photographic"] is False


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        classify_image(str(tmp_path / "brak.png"))
