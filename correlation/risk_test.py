import pytest

from correlation.risk import assess, describe_risk


def web(**data):
    return {"name": "web_infrastructure", "status": "ok", "data": data}


def report(*steps):
    return {"steps": list(steps)}


def texts(assessment):
    return [s.text for s in assessment.signals]


def test_old_plain_site_is_low_risk():
    result = assess(report(web(
        final_url="https://www.google.com/", registered_domain="google.com",
        redirect_chain=[{"url": "https://google.com"}, {"url": "https://www.google.com/"}],
        rdap={"age_days": 10000}, page_text="Szukaj w Google",
    )))
    assert result.level == "low" and result.score == 0.0
    assert texts(result) == ["domena istnieje od 27 lat"]
    assert result.label == "Raczej nie fake"


def scam_report():
    return report(
        web(
            final_url="http://orlen-invest-pl.top/x", registered_domain="orlen-invest-pl.top",
            redirect_chain=[{"url": "https://bit.ly/a"}, {"url": "https://t.example/r"}, {"url": "http://orlen-invest-pl.top/x"}],
            rdap={"age_days": 5}, page_text="SZOK! Zarób 5000 zł dziennie, gwarantowany zysk, tylko dziś!",
            media={"emojis": {"total": 8, "density": 30}},
        ),
        {"name": "ai_image", "status": "ok", "data": {"fake_probability": 0.95}},
    )


def test_fresh_domain_with_pressure_and_ai_image_is_high_risk():
    result = assess(scam_report())
    assert result.level == "high" and result.score >= 0.5
    joined = " | ".join(texts(result))
    for expected in ("5 dni", "3 różnych domen", "skracacz", ".top", "HTTPS", "emotikon", "AI (95%)", "presja czasu", "obietnica zysku"):
        assert expected in joined


def test_polish_text_matches_without_diacritics_and_case():
    result = assess(report(web(page_text="WYGRAŁEŚ nagrodę! Podaj numer PESEL")))
    joined = " ".join(texts(result))
    assert "nagroda" in joined and "wyłudzenie danych" in joined


def test_text_alone_cannot_make_a_page_high_risk():
    page = "Szok, sekret, tylko dziś, wygrales, gwarantowany zysk, podaj pesel"
    result = assess(report(web(page_text=page)))
    assert result.score <= 0.36 and result.level != "high"


def test_public_person_next_to_money_promise_adds_signal():
    steps = [
        web(page_text="Zarób na inwestycji w bitcoin"),
        {"name": "face_identities", "status": "ok", "data": [{"name": "Donald Tusk", "similarity": 0.9}]},
    ]
    result = assess(report(*steps))
    assert any("osoba publiczna" in t and "Donald Tusk" in t for t in texts(result))


def test_deepfake_and_voice_signals_need_threshold():
    low = assess(report({"name": "deepfake_image", "status": "ok", "data": {"fake_probability": 0.5}}))
    assert low.signals == []
    high = assess(report(
        {"name": "deepfake_frames", "status": "ok", "data": {"max_fake_probability": 0.8}},
        {"name": "deepfake_audio", "status": "ok", "data": {"max_fake_probability": 0.75}},
    ))
    assert len(high.signals) == 2 and high.level == "mid"


def test_no_data_is_unknown_and_score_is_clamped():
    empty = assess(report())
    assert empty.level == "unknown" and empty.label == "Za mało danych" and empty.coverage == []
    assert assess(scam_report()).score <= 1.0


def test_failed_steps_are_ignored():
    broken = {"name": "web_infrastructure", "status": "error", "data": None}
    assert assess(report(broken)).level == "unknown"


def test_describe_risk_lists_signals_with_signs():
    lines = describe_risk(assess(scam_report()))
    assert lines[0].startswith("  ryzyko: Prawdopodobnie fake")
    assert any(l.startswith("  + ") for l in lines[1:])
    old = describe_risk(assess(report(web(registered_domain="a.pl", rdap={"age_days": 4000}))))
    assert any(l.startswith("  - ") for l in old)


def test_http_site_only_gets_https_signal():
    result = assess(report(web(final_url="http://a.pl/", registered_domain="a.pl")))
    assert texts(result) == ["strona bez szyfrowania HTTPS"]


def test_ordinary_login_prompt_is_not_a_signal():
    result = assess(report(web(page_text="Google Szukaj Zaloguj się Poczta Grafika", rdap={"age_days": 10000})))
    assert [s.text for s in result.signals] == ["domena istnieje od 27 lat"]
    assert result.level == "low"
