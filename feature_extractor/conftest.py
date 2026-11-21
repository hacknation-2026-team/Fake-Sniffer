import pytest


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
