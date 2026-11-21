import numpy as np
import pytest
from pathlib import Path
from PIL import Image, ImageDraw
from unittest.mock import MagicMock, patch

from feature_extractor.model_cache import HUGGINGFACE, model_dir
from feature_extractor.visual_feature_extractor import ImageHasher, VisualEmbedder


def make_banner(seed: int, size=(320, 200)) -> Image.Image:
    """Syntetyczny 'baner': kolorowe bloki o losowym układzie (deterministyczne dla seed)."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for _ in range(8):
        x0, y0 = rng.integers(0, size[0] - 60), rng.integers(0, size[1] - 40)
        w, h = rng.integers(30, 140), rng.integers(20, 90)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=color)
    return img


@pytest.fixture
def banner_files(tmp_path: Path) -> dict:
    original = make_banner(seed=1)
    paths = {
        "original": tmp_path / "original.png",
        "resized": tmp_path / "resized.png",
        "jpeg": tmp_path / "recompressed.jpg",
        "other": tmp_path / "other.png",
    }
    original.save(paths["original"])
    original.resize((160, 100)).save(paths["resized"])
    original.save(paths["jpeg"], quality=40)
    make_banner(seed=99).save(paths["other"])
    return {k: str(v) for k, v in paths.items()}


# --- ImageHasher ---

def test_hash_format_and_determinism(banner_files):
    hasher = ImageHasher(hash_size=8)
    first = hasher.compute(banner_files["original"])
    second = hasher.compute(banner_files["original"])

    assert first == second
    assert set(first) == {"phash", "dhash", "ahash"}
    assert all(len(h) == 16 for h in first.values())  # 64 bity = 16 znaków hex


def test_larger_hash_size_gives_longer_hash(banner_files):
    assert len(ImageHasher(hash_size=16).compute(banner_files["original"])["phash"]) == 64


def test_resized_and_recompressed_are_near_duplicates(banner_files):
    hasher = ImageHasher()
    original = hasher.compute(banner_files["original"])

    assert hasher.is_near_duplicate(original, hasher.compute(banner_files["resized"]))
    assert hasher.is_near_duplicate(original, hasher.compute(banner_files["jpeg"]))


def test_different_image_is_not_near_duplicate(banner_files):
    hasher = ImageHasher()
    original = hasher.compute(banner_files["original"])
    other = hasher.compute(banner_files["other"])

    assert not hasher.is_near_duplicate(original, other)
    assert hasher.hamming_distance(original["phash"], other["phash"]) > 6


def test_hamming_distance_and_similarity():
    hasher = ImageHasher(hash_size=8)
    assert hasher.hamming_distance("0000000000000000", "0000000000000000") == 0
    assert hasher.hamming_distance("0000000000000000", "000000000000000f") == 4
    assert hasher.similarity("0000000000000000", "ffffffffffffffff") == 0.0
    assert hasher.similarity("ffffffffffffffff", "ffffffffffffffff") == 1.0


def test_hamming_distance_rejects_different_lengths():
    with pytest.raises(ValueError, match="hash_size"):
        ImageHasher.hamming_distance("00" * 8, "00" * 32)


def test_rgba_and_grayscale_images_are_accepted(tmp_path):
    rgba, gray = tmp_path / "a.png", tmp_path / "g.png"
    make_banner(3).convert("RGBA").save(rgba)
    make_banner(3).convert("L").save(gray)
    hasher = ImageHasher()
    assert hasher.compute(str(rgba))["phash"]
    assert hasher.compute(str(gray))["phash"]


def test_hasher_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="brak.png"):
        ImageHasher().compute("brak.png")


# --- VisualEmbedder ---

@pytest.fixture
def mock_clip():
    """Podmienia SentenceTransformer atrapą zwracającą znormalizowane wektory 4-wymiarowe."""
    with patch("feature_extractor.visual_feature_extractor.SentenceTransformer") as mock_cls:
        instance = MagicMock()

        def fake_encode(images, **kwargs):
            vecs = np.array([[1.0 + i, 0.5, 0.25, 0.1] for i in range(len(images))], dtype=np.float32)
            return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        instance.encode.side_effect = fake_encode
        mock_cls.return_value = instance
        yield mock_cls, instance


def test_embedder_passes_model_and_device(mock_clip):
    mock_cls, _ = mock_clip
    VisualEmbedder(model_name="clip-ViT-L-14", device="cpu")
    mock_cls.assert_called_once_with(
        "clip-ViT-L-14", device="cpu", cache_folder=str(model_dir(HUGGINGFACE))
    )


def test_embedder_downloads_into_requested_directory(mock_clip, tmp_path):
    mock_cls, _ = mock_clip
    VisualEmbedder(models_dir=str(tmp_path / "m"))
    assert mock_cls.call_args[1]["cache_folder"] == str(tmp_path / "m" / "huggingface")


def test_extract_embeddings_batch(mock_clip, banner_files):
    _, instance = mock_clip
    embedder = VisualEmbedder()

    vectors = embedder.extract_embeddings([banner_files["original"], banner_files["other"]])

    assert len(vectors) == 2
    assert all(isinstance(x, float) for v in vectors for x in v)
    assert all(np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5) for v in vectors)

    sent_images = instance.encode.call_args[0][0]
    assert all(isinstance(img, Image.Image) and img.mode == "RGB" for img in sent_images)
    assert instance.encode.call_args[1]["normalize_embeddings"] is True


def test_extract_embedding_single_returns_flat_vector(mock_clip, banner_files):
    vector = VisualEmbedder().extract_embedding(banner_files["original"])
    assert len(vector) == 4


def test_empty_input_does_not_call_model(mock_clip):
    _, instance = mock_clip
    assert VisualEmbedder().extract_embeddings([]) == []
    instance.encode.assert_not_called()


def test_embedder_missing_file_raises(mock_clip):
    with pytest.raises(FileNotFoundError, match="brak.png"):
        VisualEmbedder().extract_embedding("brak.png")


def test_cosine_similarity():
    assert VisualEmbedder.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert VisualEmbedder.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert VisualEmbedder.cosine_similarity([0, 0], [1, 0]) == 0.0
