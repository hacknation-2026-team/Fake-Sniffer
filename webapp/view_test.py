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
        ("a.png", "obraz", "high"),
        ("f.mp4", "klatki wideo", "mid"),
        ("video", "głos", "unknown"),
    ]
    assert rows[1]["mean"] == 0.2 and rows[1]["max"] == 0.5
    assert rows[2]["error"] == "boom" and rows[0]["verdict"] == "podejrzane"
