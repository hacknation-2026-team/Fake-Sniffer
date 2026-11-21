import json
import numpy as np
import pytest
from unittest.mock import MagicMock

from feature_extractor.orchestrator import Orchestrator, OrchestratorConfig, Report, StepResult
from feature_extractor.text_chunk_embedder import EmbeddedChunk


def chunk(text="treść", vector=(0.1, 0.2)):
    return EmbeddedChunk(
        chunk_index=0, text=text, embedding=list(vector), word_count=1, source_type="unknown"
    )


def make_orchestrator(config=None, **extractors):
    orchestrator = Orchestrator(config)
    for name, extractor in extractors.items():
        setattr(orchestrator, name, extractor)
    return orchestrator


def text_mocks():
    entities = MagicMock()
    entities.extract_entities.return_value = [{"text": "Tusk", "label": "persName"}]
    embedder = MagicMock()
    embedder.extract_embeddings.return_value = [chunk()]
    return {"entity_extractor": entities, "text_embedder": embedder}


def image_mocks(ocr_lines=None):
    ocr = MagicMock()
    ocr.extract_with_details.return_value = (
        [{"text": "Zarób 5000 zł", "confidence": 0.9, "bbox": []}, {"text": "Kliknij", "confidence": 0.8, "bbox": []}]
        if ocr_lines is None
        else ocr_lines
    )
    hasher = MagicMock()
    hasher.compute.return_value = {"phash": "a", "dhash": "b", "ahash": "c"}
    visual = MagicMock()
    visual.extract_embedding.return_value = [0.5, 0.5]
    faces = MagicMock()
    faces.extract_faces.return_value = [{"bbox": [0, 0, 1, 1], "det_score": 0.9, "embedding": [1.0, 0.0]}]
    return {"ocr": ocr, "hasher": hasher, "visual_embedder": visual, "faces": faces, **text_mocks()}


def names(report):
    return [step.name for step in report.steps]


def test_run_without_inputs_raises():
    with pytest.raises(ValueError, match="co najmniej jedno"):
        Orchestrator().run()


def test_blank_inputs_count_as_missing():
    with pytest.raises(ValueError):
        Orchestrator().run(url="  ", text="", image=None, video="\n")


def test_text_runs_only_text_extractors():
    mocks = text_mocks()
    report = make_orchestrator(**mocks).run(text="Tusk poleca platformę")

    assert names(report) == ["named_entities", "text_embeddings"]
    assert {s.source for s in report.steps} == {"text"}
    assert all(s.status == "ok" for s in report.steps)
    mocks["text_embedder"].extract_embeddings.assert_called_once_with("Tusk poleca platformę", source_type="unknown")


def test_text_embeddings_are_plain_dicts():
    report = make_orchestrator(**text_mocks()).run(text="abc")
    assert report.step("text_embeddings").data == [
        {"chunk_index": 0, "text": "treść", "embedding": [0.1, 0.2], "word_count": 1, "source_type": "unknown"}
    ]


def test_image_steps_in_order_with_ocr_text_fed_to_text_extractors():
    mocks = image_mocks()
    report = make_orchestrator(**mocks).run(image="ad.png")

    assert names(report) == [
        "ocr", "named_entities", "text_embeddings", "visual_hashes", "visual_embedding", "faces",
    ]
    assert report.step("ocr").data["text"] == "Zarób 5000 zł\nKliknij"
    mocks["ocr"].extract_with_details.assert_called_once_with("ad.png", min_confidence=0.3)
    mocks["entity_extractor"].extract_entities.assert_called_once_with("Zarób 5000 zł\nKliknij")
    mocks["text_embedder"].extract_embeddings.assert_called_once_with("Zarób 5000 zł\nKliknij", source_type="ocr")
    mocks["faces"].extract_faces.assert_called_once_with("ad.png")


def test_image_without_text_skips_text_steps():
    report = make_orchestrator(**image_mocks(ocr_lines=[])).run(image="ad.png")

    skipped = [s for s in report.steps if s.status == "skipped"]
    assert [s.name for s in skipped] == ["named_entities", "text_embeddings"]
    assert all(s.error == "brak tekstu" for s in skipped)
    assert report.step("faces").status == "ok"


def test_faces_are_identified_when_reference_database_is_configured():
    mocks = image_mocks()
    database = MagicMock()
    mocks["faces"].identify_faces.return_value = [{"identity": {"name": "Donald Tusk", "similarity": 0.8}}]
    config = OrchestratorConfig(reference_faces_dir="refs", face_threshold=0.5)

    report = make_orchestrator(config, face_database=database, **mocks).run(image="ad.png")

    mocks["faces"].identify_faces.assert_called_once_with("ad.png", database, 0.5)
    mocks["faces"].extract_faces.assert_not_called()
    assert report.step("faces").data[0]["identity"]["name"] == "Donald Tusk"


def test_failing_step_is_recorded_and_others_still_run():
    mocks = image_mocks()
    mocks["ocr"].extract_with_details.side_effect = RuntimeError("model nie wstał")

    report = make_orchestrator(**mocks).run(image="ad.png")

    ocr = report.step("ocr")
    assert ocr.status == "error" and ocr.error == "RuntimeError: model nie wstał" and ocr.data is None
    assert [s.status for s in report.steps if s.name in ("named_entities", "text_embeddings")] == ["skipped"] * 2
    assert report.step("visual_hashes").status == "ok"
    assert report.step("faces").status == "ok"
    assert report.has_errors


class OrchestratorWithoutSpacy(Orchestrator):
    @property
    def entity_extractor(self):
        raise OSError("brak modelu spaCy")


def test_unavailable_extractor_becomes_an_error_step():
    orchestrator = OrchestratorWithoutSpacy()
    orchestrator.text_embedder = text_mocks()["text_embedder"]

    report = orchestrator.run(text="abc")

    assert report.step("named_entities").error == "OSError: brak modelu spaCy"
    assert report.step("text_embeddings").status == "ok"


def test_url_steps_use_page_text_for_text_extractors():
    web = MagicMock()
    web.analyze.return_value.to_dict.return_value = {
        "input_url": "https://szybki.pl", "page_text": "Szok! Tusk poleca", "errors": [],
    }
    mocks = text_mocks()

    report = make_orchestrator(web_extractor=web, **mocks).run(url="https://szybki.pl")

    assert names(report) == ["web_infrastructure", "named_entities", "text_embeddings"]
    assert {s.source for s in report.steps} == {"url"}
    mocks["text_embedder"].extract_embeddings.assert_called_once_with("Szok! Tusk poleca", source_type="webpage")


def test_url_with_partial_failures_is_marked_partial():
    web = MagicMock()
    web.analyze.return_value.to_dict.return_value = {
        "page_text": None, "errors": ["tls: ConnectionRefusedError: 443", "rdap: HTTPError: 500"],
    }

    report = make_orchestrator(web_extractor=web, **text_mocks()).run(url="https://szybki.pl")

    step = report.step("web_infrastructure")
    assert step.status == "partial"
    assert step.error == "tls: ConnectionRefusedError: 443; rdap: HTTPError: 500"
    assert not report.has_errors
    assert [s.status for s in report.steps[1:]] == ["skipped", "skipped"]


def video_mocks(identify=None):
    video = MagicMock()
    video.analyze.return_value = {
        "video": "ad.mp4",
        "dhashes": ["00"],
        "ocr_text": "Tylko dziś",
        "face_embeddings": [[1.0, 0.0], [0.0, 1.0]],
        "visual_embedding": [0.7, 0.7],
        "keyframes": [{"faces": [{"embedding": [1.0, 0.0]}]}],
    }
    database = MagicMock()
    database.identify.side_effect = identify or [{"name": "Donald Tusk", "similarity": 0.9}, None]
    return {"video_extractor": video, "face_database": database, **text_mocks()}


def test_video_steps_and_face_identities():
    mocks = video_mocks()
    config = OrchestratorConfig(reference_faces_dir="refs", keyframes_dir="kf")

    report = make_orchestrator(config, **mocks).run(video="ad.mp4")

    assert names(report) == ["video_features", "face_identities", "named_entities", "text_embeddings"]
    mocks["video_extractor"].analyze.assert_called_once_with("ad.mp4", "kf")
    assert report.step("face_identities").data == [{"name": "Donald Tusk", "similarity": 0.9}, None]
    mocks["text_embedder"].extract_embeddings.assert_called_once_with("Tylko dziś", source_type="ocr")


def test_video_without_reference_database_has_no_identity_step():
    report = make_orchestrator(**video_mocks()).run(video="ad.mp4")
    assert "face_identities" not in names(report)


def test_include_vectors_false_blanks_vectors_but_keeps_pipeline_working():
    mocks = video_mocks()
    config = OrchestratorConfig(reference_faces_dir="refs", include_vectors=False)

    report = make_orchestrator(config, **mocks).run(video="ad.mp4")

    features = report.step("video_features").data
    assert features["face_embeddings"] is None and features["visual_embedding"] is None
    assert features["keyframes"][0]["faces"][0]["embedding"] is None
    assert features["dhashes"] == ["00"]
    assert report.step("face_identities").data[0]["name"] == "Donald Tusk"
    assert report.step("text_embeddings").data[0]["embedding"] is None
    assert report.step("text_embeddings").data[0]["text"] == "treść"


def test_all_inputs_run_in_documented_order():
    web = MagicMock()
    web.analyze.return_value.to_dict.return_value = {"page_text": "Treść strony", "errors": []}
    mocks = {**image_mocks(), **video_mocks(), "web_extractor": web}

    report = make_orchestrator(**mocks).run(url="https://a.pl", text="tekst", image="a.png", video="a.mp4")

    sources = [s.source for s in report.steps]
    assert [s for i, s in enumerate(sources) if i == 0 or s != sources[i - 1]] == ["url", "text", "image", "video"]
    assert list(report.inputs) == ["url", "text", "image", "video"]


def test_report_json_round_trip_keeps_polish_characters(tmp_path):
    report = make_orchestrator(**text_mocks()).run(text="Zażółć gęślą jaźń")

    target = report.save_json(tmp_path / "nested" / "dir" / "report.json")

    raw = target.read_text(encoding="utf-8")
    assert "Zażółć gęślą jaźń" in raw
    loaded = json.loads(raw)
    assert loaded["inputs"] == {"text": "Zażółć gęślą jaźń"}
    assert [s["name"] for s in loaded["steps"]] == ["named_entities", "text_embeddings"]
    assert loaded["created_at"].endswith("+00:00")


def test_report_json_handles_numpy_sets_and_dataclasses():
    report = Report(created_at="2026-11-21T00:00:00+00:00", inputs={"text": "x"})
    report.steps.append(
        StepResult(
            "demo", "text", "ok", 0.1,
            {
                "score": np.float32(0.5),
                "vector": np.array([1, 2]),
                "keys": frozenset({"b", "a"}),
                "chunk": chunk(),
            },
        )
    )

    data = json.loads(report.to_json())["steps"][0]["data"]

    assert data["score"] == 0.5
    assert data["vector"] == [1, 2]
    assert data["keys"] == ["a", "b"]
    assert data["chunk"]["text"] == "treść"


def test_report_json_rejects_unknown_types():
    report = Report(created_at="x", inputs={})
    report.steps.append(StepResult("demo", "text", "ok", 0.0, {"obj": object()}))
    with pytest.raises(TypeError, match="object"):
        report.to_json()


def test_step_lookup_by_name_and_source():
    report = Report(created_at="x", inputs={})
    report.steps = [
        StepResult("named_entities", "url", "ok", 0.0, "z url"),
        StepResult("named_entities", "image", "ok", 0.0, "z obrazu"),
    ]
    assert report.step("named_entities").data == "z url"
    assert report.step("named_entities", "image").data == "z obrazu"
    assert report.step("brak") is None


def test_construction_does_not_load_any_model():
    orchestrator = Orchestrator(OrchestratorConfig(use_gpu=True))
    assert orchestrator._device == "cuda"
    assert not any(isinstance(v, MagicMock) for v in vars(orchestrator).values())
    assert set(vars(orchestrator)) == {"config"}
