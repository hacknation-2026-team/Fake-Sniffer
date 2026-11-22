import io

import pytest
from fastapi.testclient import TestClient

from webapp.server import create_app
from webapp.service import Settings
from webapp.service_test import FakeOrchestrator, png_bytes, seed


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path / "base.json", tmp_path / "clusters.json", tmp_path / "data")


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings, FakeOrchestrator))


def test_index_and_static_files(client):
    page = client.get("/")
    assert page.status_code == 200 and "Fake Sniffer" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_analyze_with_link_and_file(client):
    response = client.post(
        "/api/analyze",
        data={"url": "https://scam.top/x"},
        files=[("files", ("zrzut.png", png_bytes().getvalue(), "image/png"))],
    )
    body = response.json()
    assert response.status_code == 200
    assert {s["name"] for s in body["steps"]} == {"web_infrastructure", "visual_hashes", "deepfake_image"}
    assert body["deepfake"][0]["item"] == "zrzut.png"
    assert body["verdict"]["verdict"] == "NEW_CAMPAIGN_CANDIDATE"


def test_analyze_without_input_is_400(client):
    response = client.post("/api/analyze", data={"url": ""})
    assert response.status_code == 400 and "link" in response.json()["detail"]


def test_add_analysis_and_browse_database(client, settings):
    analysis = client.post("/api/analyze", data={"url": "https://scam.top/x"}).json()
    added = client.post(f"/api/analyses/{analysis['id']}/add", json={"label": "Orlen"})
    assert added.json() == {"id": "inc-1"}
    assert client.post(f"/api/analyses/{analysis['id']}/add", json={}).status_code == 400
    assert client.post("/api/analyses/nie-ma/add", json={}).status_code == 404

    items = client.get("/api/database").json()
    assert [i["label"] for i in items] == ["Orlen"]


def test_campaign_endpoints(client, settings):
    seed(settings)
    assert client.post("/api/cluster").json() == {"campaigns": 2, "families": 2}
    data = client.get("/api/campaigns").json()
    assert data["total"] == 4 and len(data["campaigns"]) == 2

    detail = client.get("/api/incidents/inc-1").json()
    assert detail["campaign"] == "campaign-1"
    comparison = client.get("/api/compare", params={"a": "inc-1", "b": "inc-2"}).json()
    assert comparison["hard"]


def test_unknown_incident_is_404_everywhere(client, settings):
    seed(settings)
    assert client.get("/api/incidents/x").status_code == 404
    assert client.get("/api/incidents/x/preview").status_code == 404
    assert client.get("/api/incidents/inc-1/preview").status_code == 404
    assert client.get("/api/compare", params={"a": "inc-1", "b": "x"}).status_code == 404
    assert client.delete("/api/incidents/x").status_code == 404


def test_preview_is_served_and_delete_works(client):
    analysis = client.post(
        "/api/analyze", files=[("files", ("zrzut.png", png_bytes().getvalue(), "image/png"))]
    ).json()
    client.post(f"/api/analyses/{analysis['id']}/add", json={})
    image = client.get("/api/incidents/inc-1/preview")
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert client.delete("/api/incidents/inc-1").json() == {"deleted": "inc-1"}
    assert client.get("/api/database").json() == []


def test_recluster_on_empty_base_is_400(client):
    assert client.post("/api/cluster").status_code == 400
