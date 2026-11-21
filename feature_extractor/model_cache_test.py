from pathlib import Path

from feature_extractor import model_cache
from feature_extractor.model_cache import (
    EASYOCR,
    MODELS_DIR_ENV,
    PLAYWRIGHT,
    default_models_dir,
    model_dir,
)


def test_default_is_models_folder_next_to_the_package(monkeypatch):
    monkeypatch.delenv(MODELS_DIR_ENV)

    expected = Path(model_cache.__file__).resolve().parent.parent / "models"
    assert default_models_dir() == expected


def test_environment_variable_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path / "z_env"))
    assert default_models_dir() == tmp_path / "z_env"


def test_environment_variable_expands_home(monkeypatch):
    monkeypatch.setenv(MODELS_DIR_ENV, "~/fake_sniffer_models")
    assert default_models_dir() == Path.home() / "fake_sniffer_models"


def test_model_dir_creates_nested_folders(tmp_path):
    path = model_dir(EASYOCR, tmp_path / "a" / "b")

    assert path == tmp_path / "a" / "b" / "easyocr"
    assert path.is_dir()


def test_explicit_directory_wins_over_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path / "env"))

    path = model_dir(PLAYWRIGHT, tmp_path / "explicit")

    assert path == tmp_path / "explicit" / "playwright"
    assert not (tmp_path / "env").exists()


def test_model_dir_is_idempotent(tmp_path):
    first = model_dir(EASYOCR, tmp_path)
    (first / "model.pth").write_bytes(b"waga")

    assert model_dir(EASYOCR, tmp_path) == first
    assert (first / "model.pth").read_bytes() == b"waga"
