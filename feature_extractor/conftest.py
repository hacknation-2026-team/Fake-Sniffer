import pytest

from feature_extractor.model_cache import MODELS_DIR_ENV


def pytest_configure(config: pytest.Config) -> None:
    """Rejestruje własne markery, żeby pytest nie ostrzegał o 'Unknown pytest.mark.*'."""
    config.addinivalue_line(
        "markers",
        "integration: testy z prawdziwymi modelami/siecią (wolne; pomiń przez -m 'not integration')",
    )
    config.addinivalue_line(
        "markers",
        "network: testy wymagające dostępu do internetu (RDAP, DNS, przeglądarka)",
    )


@pytest.fixture(autouse=True)
def isolated_models_dir(tmp_path, monkeypatch):
    """Kieruje katalog modeli do tymczasowego, żeby testy nie tworzyły ./models w repozytorium."""
    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path / "models"))
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "")
    monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH")
