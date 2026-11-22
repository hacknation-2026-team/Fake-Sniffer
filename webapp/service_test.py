from pathlib import Path
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from correlation.base import Base
from correlation.profile import Profile
from webapp.service import Settings, Workspace, _safe_name


class FakeOrchestrator:
    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace(keyframes_dir="")

    def run(self, **inputs):
        self.calls.append(inputs)
        (kind, value), = inputs.items()
        if kind == "image":
            image_steps = [
                {"name": "visual_hashes", "source": "image", "status": "ok",
                 "data": {"phash": "ab" * 8, "dhash": "cd" * 8, "ahash": "ef" * 8}},
                {"name": "deepfake_image", "source": "image", "status": "ok",
                 "data": {"file": Path(value).name, "fake_probability": 0.9, "model": "m"}},
            ]
            return SimpleNamespace(to_dict=lambda: {"steps": image_steps})
        steps = {
            "url": [{"name": "web_infrastructure", "source": "url", "status": "ok", "duration_sec": 1.0,
                     "data": {"registered_domain": "scam.top", "final_url": value}}],
            "text": [{"name": "named_entities", "source": "text", "status": "ok",
                      "data": [{"text": "Orlen", "label": "orgName", "lemma": "Orlen"}]}],
            "image": [{"name": "visual_hashes", "source": "image", "status": "ok",
                       "data": {"phash": "ab" * 8, "dhash": "cd" * 8, "ahash": "ef" * 8}}],
            "audio": [{"name": "audio_features", "source": "audio", "status": "ok", "data": {"text": "dzień dobry"}}],
            "video": [{"name": "video_features", "source": "video", "status": "ok",
                       "data": {"dhashes": ["12" * 8], "face_embeddings": [], "keyframes": []}}],
        }[kind]
        return SimpleNamespace(to_dict=lambda: {"steps": steps})


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path / "base.json", tmp_path / "clusters.json", tmp_path / "data")


@pytest.fixture
def fake():
    return FakeOrchestrator()


@pytest.fixture
def workspace(settings, fake):
    return Workspace(settings, lambda: fake)


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (1600, 900), "red").save(buffer, "PNG")
    return io.BytesIO(buffer.getvalue())


def seed(settings):
    base = Base(settings.base_path)
    for i in range(2):
        base.add(Profile(domains=["scam.top"], entities=["orgName:orlen"]), f"a{i}", f"Orlen {i}")
    for i in range(2):
        base.add(Profile(domains=["other.xyz"]), f"b{i}", f"Inna {i}")
    base.save()


def test_safe_name_strips_paths_and_odd_characters():
    assert _safe_name("../../etc/pass wd?.png") == "pass wd_.png"
    assert _safe_name("...") == "plik"


def test_analyze_needs_input(workspace):
    with pytest.raises(ValueError):
        workspace.analyze("  ", [])


def test_analyze_url_against_empty_base(workspace):
    result = workspace.analyze("https://scam.top/x", [])
    assert result["name"] == "https://scam.top/x"
    assert result["verdict"]["verdict"] == "NO_MATCH"
    assert result["steps"][0]["summary"] == "https://scam.top/x"
    assert result["has_features"]


def test_analyze_known_campaign_reports_cluster_and_sizes(settings, workspace):
    seed(settings)
    result = workspace.analyze("https://scam.top/x", [])
    assert result["verdict"]["verdict"] == "KNOWN_CAMPAIGN"
    assert result["verdict"]["campaign"] == "campaign-1"
    assert result["campaign_size"] == 2
    assert settings.clusters_path.exists()


def test_analyze_files_go_to_matching_extractors(workspace, fake):
    uploads = [
        ("notatka.txt", io.BytesIO("Orlen oferuje dochód".encode())),
        ("nagranie.mp3", io.BytesIO(b"x")),
        ("zrzut.png", png_bytes()),
        ("film.mp4", io.BytesIO(b"x")),
    ]
    result = workspace.analyze("", uploads)
    kinds = sorted(next(iter(c)) for c in fake.calls)
    assert kinds == ["audio", "image", "text", "video"]
    assert result["name"] == "notatka.txt, nagranie.mp3, zrzut.png, film.mp4"
    assert sorted(result["inputs"]["files"]) == sorted(["film.mp4", "nagranie.mp3", "notatka.txt", "zrzut.png"])


def test_analysis_without_features_cannot_be_added(workspace):
    result = workspace.analyze("", [("pusty.txt", io.BytesIO(b"   "))])
    assert not result["has_features"] and result["verdict"] is None
    with pytest.raises(ValueError):
        workspace.add_to_base(result["id"])


def test_add_to_base_copies_preview_and_blocks_duplicates(settings, workspace):
    result = workspace.analyze("", [("zrzut.png", png_bytes())])
    added = workspace.add_to_base(result["id"], "Moja kampania")
    assert added == {"id": "inc-1"}

    incident = Base(settings.base_path).get("inc-1")
    assert incident.label == "Moja kampania"
    assert workspace.preview_path("inc-1").is_file()
    with Image.open(workspace.preview_path("inc-1")) as thumb:
        assert max(thumb.size) == 720
    with pytest.raises(ValueError):
        workspace.add_to_base(result["id"])
    assert workspace.analyze("", [("zrzut.png", png_bytes())])["verdict"]["verdict"] == "KNOWN_CAMPAIGN"


def test_add_unknown_analysis(workspace):
    with pytest.raises(LookupError):
        workspace.add_to_base("brak")


def test_campaigns_view_groups_and_edges(settings, workspace):
    seed(settings)
    view = workspace.campaigns_view()
    assert view["total"] == 4 and not view["outdated"]
    sizes = [c["size"] for c in view["campaigns"]]
    assert sizes == [2, 2]
    first = view["campaigns"][0]
    assert {n["id"] for n in first["nodes"]} == {"inc-1", "inc-2"}
    assert first["edges"][0]["hard"] is True


def test_campaigns_view_flags_incidents_added_after_clustering(settings, workspace):
    seed(settings)
    workspace.campaigns_view()
    base = Base(settings.base_path)
    base.add(Profile(domains=["nowa.pl"]), "c", "Nowa")
    base.save()
    view = workspace.campaigns_view()
    assert view["outdated"] and view["campaigns"][-1]["id"] == "unassigned"
    workspace.recluster()
    assert not workspace.campaigns_view()["outdated"]


def test_empty_base_views(workspace):
    assert workspace.campaigns_view() == {"campaigns": [], "total": 0, "outdated": False}
    assert workspace.database_view() == []
    with pytest.raises(ValueError):
        workspace.recluster()


def test_incident_details_compare_and_delete(settings, workspace):
    seed(settings)
    workspace.recluster()
    details = workspace.incident_view("inc-1")
    assert details["campaign"] == "campaign-1" and details["profile"]["domains"] == ["scam.top"]

    same = workspace.compare("inc-1", "inc-2")
    assert same["hard"] == ["wspólna domena scam.top"] and same["a"]["label"] == "Orlen 0"
    assert workspace.compare("inc-1", "inc-3")["hard"] == []

    workspace.delete("inc-1")
    assert [i["id"] for i in workspace.database_view()] == ["inc-2", "inc-3", "inc-4"]
    for call in (lambda: workspace.incident_view("inc-1"), lambda: workspace.compare("inc-1", "inc-2"),
                 lambda: workspace.delete("inc-1")):
        with pytest.raises(LookupError):
            call()


def test_preview_missing_returns_none(settings, workspace):
    seed(settings)
    assert workspace.preview_path("inc-1") is None


def test_analysis_view_has_deepfake_table(workspace):
    result = workspace.analyze("", [("zrzut.png", png_bytes())])
    assert result["deepfake_enabled"] is True
    row = result["deepfake"][0]
    assert (row["item"], row["kind"], row["level"], row["verdict"]) == ("zrzut.png", "Podmieniona twarz (zdjęcie)", "high", "podejrzane")


def test_deepfake_can_be_disabled_in_settings(settings, fake):
    settings.deepfake = False
    settings.ai_image = False
    result = Workspace(settings, lambda: fake).analyze("https://scam.top/x", [])
    assert result["deepfake_enabled"] is False and result["deepfake"] == []


def test_page_media_steps_keep_their_page_sources(settings):
    page_steps = [
        {"name": "web_infrastructure", "source": "url", "status": "ok",
         "data": {"registered_domain": "scam.top", "final_url": "https://scam.top/",
                  "media": {"images": [{"url": "https://cdn/a.jpg"}], "videos": [], "inline_images": 0,
                            "emojis": {"total": 2, "unique": 1, "density": 3.0, "top": [["🔥", 2]]}}}},
        {"name": "page_media", "source": "url", "status": "ok",
         "data": [{"kind": "image", "url": "https://cdn/a.jpg", "status": "ok", "file_path": "/x/image_1.jpg",
                   "width": 640, "height": 480}]},
        {"name": "ai_image", "source": "page_image", "status": "ok",
         "data": {"file": "image_1.jpg", "fake_probability": 0.95, "model": "m"}},
    ]

    class PageOrchestrator(FakeOrchestrator):
        def run(self, **inputs):
            return SimpleNamespace(to_dict=lambda: {"steps": page_steps})

    result = Workspace(settings, lambda: PageOrchestrator()).analyze("https://scam.top/", [])
    assert [s["source"] for s in result["steps"]] == ["url", "url", "page_image"]
    row = result["deepfake"][0]
    assert (row["item"], row["kind"], row["origin"]) == ("https://cdn/a.jpg", "Wygenerowany przez AI (zdjęcie)", "strona")
    media = result["page_media"]
    assert media["images"] == 1 and media["downloaded"] and media["emojis"]["total"] == 2
    assert media["items"][0]["status_label"] == "przeanalizowano"
    assert result["page_media_enabled"] is True


def test_default_orchestrator_gets_page_media_and_ai_settings(settings):
    from unittest.mock import patch

    from webapp.service import _default_orchestrator

    with patch("feature_extractor.orchestrator.Orchestrator") as orchestrator:
        _default_orchestrator(settings)
    config = orchestrator.call_args[0][0]
    assert config.ai_image and config.page_media
    assert config.page_media_dir == str(settings.data_dir / "page_media")


def test_analysis_view_has_summary(workspace):
    result = workspace.analyze("https://scam.top/x", [])
    summary = result["summary"]
    assert summary["headline"]["text"]
    assert any(f["text"] == "Domeny: scam.top" for f in summary["findings"])


def test_analysis_view_has_risk_assessment(workspace):
    result = workspace.analyze("https://scam.top/x", [])
    risk = result["risk"]
    assert risk["level"] in {"low", "mid", "high", "unknown"}
    assert {"score", "label", "signals", "coverage"} <= set(risk)
