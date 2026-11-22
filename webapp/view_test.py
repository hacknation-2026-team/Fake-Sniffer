import numpy as np

from correlation.profile import Profile
from webapp.view import profile_view, step_view, summarize_step, trim


def test_trim_shortens_text_lists_and_drops_vectors():
    data = {"text": "x" * 500, "items": list(range(30)), "embedding": [0.1, 0.2], "arr": np.array([1, 2])}
    out = trim(data)
    assert out["text"].endswith("…") and len(out["text"]) == 301
    assert len(out["items"]) == 21 and "jeszcze 10" in out["items"][-1]
    assert out["embedding"] is None
    assert out["arr"] == [1, 2]


def test_summaries_per_extractor():
    web = {"final_url": "https://a.pl/x", "redirect_chain": [{}, {}], "dns": {"a": ["1.1.1.1"]}}
    assert summarize_step({"name": "web_infrastructure", "status": "ok", "data": web}) == (
        "https://a.pl/x, przekierowania: 1, IP: 1"
    )
    entities = [{"text": "Orlen"}, {"text": "Tusk"}]
    assert summarize_step({"name": "named_entities", "status": "ok", "data": entities}) == "2 encji: Orlen, Tusk"
    faces = [{"identity": {"name": "Ewa"}}, {"identity": None}]
    assert summarize_step({"name": "faces", "status": "ok", "data": faces}) == "2 twarzy (Ewa)"
    assert summarize_step({"name": "ocr", "status": "ok", "data": {"text": "abc", "lines": [1]}}) == "3 znaków, 1 linii"
    assert summarize_step({"name": "visual_embedding", "status": "ok", "data": [0.0] * 512}) == "wektor 512-wymiarowy"


def test_summary_falls_back_when_data_has_unexpected_shape():
    assert summarize_step({"name": "video_features", "status": "ok", "data": {}}) == "ok"
    assert summarize_step({"name": "inny", "status": "ok", "data": {"a": 1}}) == "ok"


def test_summary_of_errors_and_partial_steps():
    assert summarize_step({"name": "ocr", "status": "error", "data": None, "error": "boom"}) == "boom"
    assert summarize_step({"name": "x", "status": "skipped", "data": None}) == "skipped"
    partial = {"name": "inny", "status": "partial", "data": {"a": 1}, "error": "RDAP"}
    assert summarize_step(partial) == "ok (RDAP)"


def test_step_view_contains_trimmed_data():
    view = step_view({"name": "ocr", "source": "image", "status": "ok", "duration_sec": 1.5, "data": {"text": "a"}})
    assert view["summary"] == "1 znaków, 0 linii" and view["data"] == {"text": "a"}


def test_profile_view_shortens_hashes():
    view = profile_view(Profile(dom_hash="a" * 64, cert_sha256="b" * 64, face_embeddings=[[1.0]], domains=["a.pl"]))
    assert view["dom_hash"] == "a" * 16 and view["faces"] == 1 and view["has_text_embedding"] is False


def test_deepfake_summaries():
    image = {"name": "deepfake_image", "status": "ok", "data": {"fake_probability": 0.834}}
    assert summarize_step(image) == "prawdopodobieństwo fałszerstwa 83%"
    frames = {"name": "deepfake_frames", "status": "ok", "data": {"frames": [1, 2], "max_fake_probability": 0.5}}
    assert summarize_step(frames) == "2 klatek, prawdopodobieństwo fałszerstwa max 50%"
    unknown = {"name": "deepfake_audio", "status": "ok", "data": {"max_fake_probability": None}}
    assert summarize_step(unknown) == "nie rozpoznano etykiety fałszerstwa"
    assert summarize_step({"name": "deepfake_audio", "status": "ok", "data": None}) == "brak danych"


def test_risk_levels():
    from webapp.view import risk_level

    assert [risk_level(p) for p in (None, 0.1, 0.4, 0.69, 0.7, 1.0)] == [
        "unknown", "low", "mid", "mid", "high", "high",
    ]


def test_deepfake_rows():
    from webapp.view import deepfake_rows

    steps = [
        {"name": "ocr", "status": "ok", "data": {}},
        {"name": "deepfake_image", "source": "image", "status": "ok",
         "data": {"file": "a.png", "fake_probability": 0.9, "model": "m"}},
        {"name": "deepfake_frames", "source": "video", "status": "ok",
         "data": {"file": "f.mp4", "mean_fake_probability": 0.2, "max_fake_probability": 0.5}},
        {"name": "deepfake_audio", "source": "audio", "status": "ok", "data": None},
        {"name": "deepfake_audio", "source": "video", "status": "error", "data": None, "error": "boom"},
    ]
    rows = deepfake_rows(steps)
    assert [(r["item"], r["kind"], r["level"]) for r in rows] == [
        ("a.png", "Podmieniona twarz (zdjęcie)", "high"),
        ("f.mp4", "Podmieniona twarz (klatki filmu)", "mid"),
        ("video", "Sklonowany lub syntetyczny głos", "unknown"),
    ]
    assert rows[1]["mean"] == 0.2 and rows[1]["max"] == 0.5
    assert rows[2]["error"] == "boom" and rows[0]["verdict"] == "podejrzane"


def test_summary_of_long_error_is_one_short_line():
    error = "browser: TargetClosedError: " + "--flag " * 200 + "\nCall log:\n- <kill>"
    text = summarize_step({"name": "web_infrastructure", "status": "error", "data": None, "error": error})
    assert "\n" not in text and len(text) <= 161


def test_page_media_summaries():
    pages = [
        {"kind": "image", "status": "ok"},
        {"kind": "video", "status": "ok"},
        {"kind": "image", "status": "skipped"},
    ]
    assert summarize_step({"name": "page_media", "status": "ok", "data": pages}) == "pobrano obrazy: 1, filmy: 1, pominięto: 1"
    web = {"final_url": "https://a.pl/", "media": {"images": [1, 2], "videos": [1], "emojis": {"total": 4}}}
    text = summarize_step({"name": "web_infrastructure", "status": "ok", "data": web})
    assert "obrazy: 2, filmy: 1" in text and "emotikony: 4" in text
    ai = {"name": "ai_image", "status": "ok", "data": {"fake_probability": 0.5}}
    assert "50%" in summarize_step(ai)


def test_page_media_view_without_download_lists_found_items():
    from webapp.view import page_media_view

    media = {"images": [{"url": "https://a/x.jpg"}], "videos": [{"url": "https://a/v.mp4"}], "inline_images": 1,
             "emojis": {"total": 0, "unique": 0, "density": 0, "top": []}}
    steps = [{"name": "web_infrastructure", "data": {"media": media}, "status": "ok"}]
    view = page_media_view(steps)
    assert view["downloaded"] is False
    assert [(i["kind"], i["status_label"]) for i in view["items"]] == [("image", "znaleziono"), ("video", "znaleziono")]


def test_page_media_view_is_none_for_empty_page():
    from webapp.view import page_media_view

    empty = {"images": [], "videos": [], "inline_images": 0, "emojis": {"total": 0, "unique": 0, "density": 0, "top": []}}
    assert page_media_view([{"name": "web_infrastructure", "data": {"media": empty}, "status": "ok"}]) is None
    assert page_media_view([{"name": "web_infrastructure", "data": {}, "status": "ok"}]) is None
    assert page_media_view([]) is None


def test_page_media_view_reports_download_error():
    from webapp.view import page_media_view

    media = {"images": [{"url": "https://a/x.jpg"}], "videos": [], "inline_images": 0,
             "emojis": {"total": 0, "unique": 0, "density": 0, "top": []}}
    steps = [
        {"name": "web_infrastructure", "data": {"media": media}, "status": "ok"},
        {"name": "page_media", "data": None, "status": "error", "error": "OSError: dysk pełny"},
    ]
    view = page_media_view(steps)
    assert view["error"] == "OSError: dysk pełny" and view["downloaded"] is False


def test_step_view_has_readable_labels():
    from webapp.view import step_view

    view = step_view({"name": "ai_image", "source": "page_image", "status": "ok", "data": {"fake_probability": 0.9}})
    assert view["label"] == "Wygenerowany przez AI (zdjęcie)"
    assert view["source_label"] == "obraz ze strony"
    assert step_view({"name": "nowy_krok", "source": "x", "status": "ok", "data": {}})["label"] == "nowy_krok"


def test_metadata_steps_have_summaries():
    from webapp.view import summarize_step

    image = {"name": "file_metadata", "status": "ok", "data": {
        "ai_hints": ["manifest C2PA: obraz wygenerowany przez AI (Google)"], "tools": ["Adobe Photoshop 26.2"],
        "export_apps": ["Adobe Photoshop"], "gps": None, "created": None}}
    page = {"name": "page_metadata", "status": "ok", "data": {
        "page_domain": "civicdaybook.live", "trackers": [{"kind": "facebook_pixel", "id": "123"}],
        "ad_ids": ["1", "2"], "claimed_domains": ["gazeta.pl"], "generator": None}}
    assert summarize_step(image).startswith("manifest C2PA: obraz wygenerowany przez AI (Google); narzędzia:")
    assert summarize_step(page) == (
        "civicdaybook.live; trackery: facebook_pixel 123; identyfikatory reklam: 2; podszywa się pod: gazeta.pl"
    )


def test_report_summary_orders_findings_by_severity():
    from correlation.profile import Profile
    from webapp.view import report_summary

    steps = [
        {"name": "ai_image", "source": "image", "status": "ok", "data": {"file": "a.png", "fake_probability": 0.95}},
        {"name": "page_metadata", "source": "url", "status": "ok", "data": {"claimed_domains": ["gazeta.pl"]}},
        {"name": "ocr", "source": "image", "status": "error", "data": None, "error": "AttributeError: x"},
    ]
    verdict = {
        "verdict": "KNOWN_CAMPAIGN", "campaign": "campaign-1", "reasons": [],
        "neighbours": [{"id": "inc-2", "label": "Brzoska", "score": 0.91, "hard": ["wspólny identyfikator Facebook Pixel 1"]}],
    }
    profile = Profile(domains=["civicdaybook.live"], ai_hints=["manifest C2PA: obraz wygenerowany przez AI (Google)"])

    summary = report_summary(steps, verdict, profile, campaign_size=3)

    assert summary["headline"]["text"] == (
        "Materiał pasuje do znanej kampanii: campaign-1, 3 incydentów. "
        "Najbliżej: Brzoska (inc-2), podobieństwo 0.91."
    )
    assert summary["level"] == "high"
    severities = [f["severity"] for f in summary["findings"]]
    assert severities == sorted(severities, key={"high": 0, "mid": 1, "info": 2}.get)
    texts = [f["text"] for f in summary["findings"]]
    assert "Twardy ślad łączący z inc-2: wspólny identyfikator Facebook Pixel 1" in texts
    assert "Strona podszywa się pod: gazeta.pl" in texts
    assert "Metadane: manifest C2PA: obraz wygenerowany przez AI (Google)" in texts
    assert any(t.startswith("Wygenerowany przez AI (zdjęcie): 95%") for t in texts)
    assert "Błąd ekstraktorów (1): Tekst odczytany z obrazu (OCR)" in texts
    assert summary["counts"] == {"ok": 2, "partial": 0, "skipped": 0, "error": 1}


def test_report_summary_without_verdict():
    from correlation.profile import Profile
    from webapp.view import report_summary

    summary = report_summary([], None, Profile())

    assert summary["headline"]["verdict"] is None
    assert summary["findings"] == [] and summary["level"] == "info"


def test_report_summary_known_without_campaign():
    from correlation.profile import Profile
    from webapp.view import report_summary

    verdict = {"verdict": "KNOWN_CAMPAIGN", "campaign": None, "reasons": ["podobny tekst"],
               "neighbours": [{"id": "inc-2", "label": "2", "score": 0.83, "hard": []}]}

    text = report_summary([], verdict, Profile())["headline"]["text"]

    assert text.startswith("Materiał jest powiązany ze znanymi incydentami, ale za słabo")
    assert text.endswith("Najbliżej: 2 (inc-2), podobieństwo 0.83.")


def test_deepfake_rows_show_skipped_detectors_with_reason():
    from webapp.view import deepfake_rows

    steps = [{"name": "ai_image", "source": "page_image", "status": "skipped",
              "data": {"file": "logo.png"}, "error": "grafika, nie fotografia (przezroczyste tło)"}]
    row = deepfake_rows(steps)[0]
    assert (row["item"], row["level"], row["verdict"]) == ("logo.png", "skip", "pominięto")
    assert "grafika" in row["note"] and row["error"] is None and row["max"] is None
