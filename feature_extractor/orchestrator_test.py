import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

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

    assert names(report) == ["named_entities", "text_embeddings", "sentence_rhythm", "paragraphs"]
    assert {s.source for s in report.steps} == {"text"}
    assert all(s.status == "ok" for s in report.steps)
    mocks["text_embedder"].extract_embeddings.assert_called_once_with("Tusk poleca platformę", source_type="unknown")


def test_text_embeddings_are_plain_dicts():
    report = make_orchestrator(**text_mocks()).run(text="abc")
    assert report.step("text_embeddings").data == [
        {"chunk_index": 0, "text": "treść", "embedding": [0.1, 0.2], "word_count": 1, "source_type": "unknown"}
    ]


def test_text_gets_sentence_rhythm_and_paragraphs():
    report = make_orchestrator(**text_mocks()).run(text="Nagłówek\n\nPierwszy akapit. Drugie zdanie!")

    assert report.step("sentence_rhythm").data["n_sentences"] == 3
    assert report.step("paragraphs").data["structure"]["pattern"] == "Hs"


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

    assert names(report) == ["web_infrastructure", "named_entities", "text_embeddings", "paragraphs"]
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
    assert [s.status for s in report.steps[1:]] == ["skipped", "skipped", "skipped"]


def test_url_paragraphs_use_page_html_without_storing_it():
    html = "<html><body><article><h1>Szok!</h1><p>Tusk poleca platformę.</p></article></body></html>"
    web = MagicMock()
    web.analyze.return_value.to_dict.return_value = {
        "final_url": "https://szybki.pl/oferta", "page_text": "Szok! Tusk poleca", "page_html": html, "errors": [],
    }
    paragraphs = MagicMock()
    paragraphs.analyze_html.return_value = {"structure": {"pattern": "Hs"}}

    report = make_orchestrator(web_extractor=web, paragraphs=paragraphs, **text_mocks()).run(url="https://szybki.pl")

    paragraphs.analyze_html.assert_called_once_with(html, url="https://szybki.pl/oferta")
    assert report.step("paragraphs").data == {"structure": {"pattern": "Hs"}}
    assert "page_html" not in report.step("web_infrastructure").data


def test_url_without_html_skips_paragraphs():
    web = MagicMock()
    web.analyze.return_value.to_dict.return_value = {"page_text": "Szok!", "errors": []}

    report = make_orchestrator(web_extractor=web, **text_mocks()).run(url="https://szybki.pl")

    step = report.step("paragraphs")
    assert (step.status, step.error) == ("skipped", "brak HTML strony")


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
    return {
        "video_extractor": video,
        "face_database": database,
        "audio_extractor": audio_mock(text=""),
        **text_mocks(),
    }


def audio_mock(text="Kup teraz. Promocja trwa tylko dziś!"):
    audio = MagicMock()
    audio.analyze.return_value = {
        "text": text,
        "timestamps": {0.0: "Kup teraz."},
        "rhythm": {"n_sentences": 2},
        "speech": {"status": "ok"},
        "spectrogram": {"status": "ok", "features": {"embedding": [0.6, 0.8]}},
    }
    return audio


def test_video_steps_and_face_identities():
    mocks = video_mocks()
    config = OrchestratorConfig(reference_faces_dir="refs", keyframes_dir="kf")

    report = make_orchestrator(config, **mocks).run(video="ad.mp4")

    assert names(report) == [
        "video_features", "face_identities", "audio_features", "named_entities", "text_embeddings",
    ]
    mocks["video_extractor"].analyze.assert_called_once_with("ad.mp4", "kf")
    assert report.step("face_identities").data == [{"name": "Donald Tusk", "similarity": 0.9}, None]
    mocks["text_embedder"].extract_embeddings.assert_called_once_with("Tylko dziś", source_type="ocr")


def test_video_without_reference_database_has_no_identity_step():
    report = make_orchestrator(**video_mocks()).run(video="ad.mp4")
    assert "face_identities" not in names(report)


def test_video_transcript_joins_ocr_text():
    mocks = {**video_mocks(), "audio_extractor": audio_mock()}
    config = OrchestratorConfig(spectrograms_dir="spektrogramy")

    report = make_orchestrator(config, **mocks).run(video="ad.mp4")

    mocks["audio_extractor"].analyze.assert_called_once_with("ad.mp4", "spektrogramy")
    assert report.step("audio_features").source == "video"
    text = "Tylko dziś\n\nKup teraz. Promocja trwa tylko dziś!"
    mocks["entity_extractor"].extract_entities.assert_called_once_with(text)
    mocks["text_embedder"].extract_embeddings.assert_called_once_with(text, source_type="unknown")


def test_failed_audio_track_keeps_ocr_text_steps():
    mocks = video_mocks()
    mocks["audio_extractor"].analyze.side_effect = RuntimeError("brak dekodera")

    report = make_orchestrator(**mocks).run(video="ad.mp4")

    assert report.step("audio_features").error == "RuntimeError: brak dekodera"
    assert report.step("text_embeddings").status == "ok"
    mocks["text_embedder"].extract_embeddings.assert_called_once_with("Tylko dziś", source_type="ocr")


def test_audio_input_runs_audio_and_transcript_text_steps():
    mocks = {"audio_extractor": audio_mock(), **text_mocks()}

    report = make_orchestrator(**mocks).run(audio="nagranie.mp3")

    assert names(report) == ["audio_features", "named_entities", "text_embeddings"]
    assert {s.source for s in report.steps} == {"audio"}
    mocks["audio_extractor"].analyze.assert_called_once_with("nagranie.mp3", None)
    mocks["text_embedder"].extract_embeddings.assert_called_once_with(
        "Kup teraz. Promocja trwa tylko dziś!", source_type="transcript"
    )


def test_audio_without_speech_skips_text_steps():
    report = make_orchestrator(audio_extractor=audio_mock(text=""), **text_mocks()).run(audio="cisza.wav")

    assert [s.status for s in report.steps] == ["ok", "skipped", "skipped"]


def test_include_vectors_false_blanks_spectrogram_embedding():
    config = OrchestratorConfig(include_vectors=False)

    report = make_orchestrator(config, audio_extractor=audio_mock(), **text_mocks()).run(audio="a.mp3")

    assert report.step("audio_features").data["spectrogram"]["features"]["embedding"] is None


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

    report = make_orchestrator(**mocks).run(
        url="https://a.pl", text="tekst", image="a.png", video="a.mp4", audio="a.mp3"
    )

    sources = [s.source for s in report.steps]
    assert [s for i, s in enumerate(sources) if i == 0 or s != sources[i - 1]] == [
        "url", "text", "image", "video", "audio",
    ]
    assert list(report.inputs) == ["url", "text", "image", "video", "audio"]


def test_report_json_round_trip_keeps_polish_characters(tmp_path):
    report = make_orchestrator(**text_mocks()).run(text="Zażółć gęślą jaźń")

    target = report.save_json(tmp_path / "nested" / "dir" / "report.json")

    raw = target.read_text(encoding="utf-8")
    assert "Zażółć gęślą jaźń" in raw
    loaded = json.loads(raw)
    assert loaded["inputs"] == {"text": "Zażółć gęślą jaźń"}
    assert [s["name"] for s in loaded["steps"]] == [
        "named_entities", "text_embeddings", "sentence_rhythm", "paragraphs",
    ]
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


def test_speech_extractor_is_built_from_config():
    config = OrchestratorConfig(use_gpu=True, whisper_model="small", speech_language="pl")

    with patch("feature_extractor.speech_extractor.SpeechExtractor") as speech_cls:
        speech = Orchestrator(config).speech

    speech_cls.assert_called_once_with(model_name="small", language="pl", device="cuda", models_dir=None)
    assert speech is speech_cls.return_value


def prefetch_mocks():
    return {
        "ocr": MagicMock(),
        "text_embedder": MagicMock(),
        "entity_extractor": MagicMock(),
        "visual_embedder": MagicMock(),
        "faces": MagicMock(),
        "speech": MagicMock(),
        "web_extractor": MagicMock(),
    }


def test_prefetch_loads_every_model_and_the_browser():
    mocks = prefetch_mocks()

    statuses = make_orchestrator(**mocks).prefetch()

    assert statuses == {
        "ocr": "ok",
        "text_embedder": "ok",
        "entity_extractor": "ok",
        "visual_embedder": "ok",
        "faces": "ok",
        "speech": "ok",
        "browser": "ok",
    }
    mocks["web_extractor"].ensure_browser.assert_called_once_with()


def test_prefetch_reports_failures_without_stopping():
    mocks = prefetch_mocks()
    mocks["web_extractor"].ensure_browser.side_effect = OSError("brak sieci")

    class WithoutFaces(Orchestrator):
        @property
        def faces(self):
            raise RuntimeError("nie da się pobrać buffalo_l")

    orchestrator = WithoutFaces()
    for name, mock in mocks.items():
        if name != "faces":
            setattr(orchestrator, name, mock)

    statuses = orchestrator.prefetch()

    assert statuses["faces"] == "RuntimeError: nie da się pobrać buffalo_l"
    assert statuses["browser"] == "OSError: brak sieci"
    assert statuses["speech"] == "ok"
    assert list(statuses)[-1] == "browser"


def test_models_dir_is_passed_to_every_extractor_that_downloads():
    from unittest.mock import patch

    orchestrator = Orchestrator(OrchestratorConfig(models_dir="moje_modele", use_gpu=False))
    targets = {
        "ocr": "feature_extractor.ocr_text_extractor.TextExtractor",
        "text_embedder": "feature_extractor.text_chunk_embedder.DocumentEmbedder",
        "visual_embedder": "feature_extractor.visual_feature_extractor.VisualEmbedder",
        "faces": "feature_extractor.face_identity_extractor.FaceIdentityExtractor",
        "speech": "feature_extractor.speech_extractor.SpeechExtractor",
        "web_extractor": "feature_extractor.web_infrastructure_extractor.WebInfrastructureExtractor",
    }
    for attribute, target in targets.items():
        with patch(target) as constructor:
            getattr(orchestrator, attribute)
        passed = list(constructor.call_args[0]) + list(constructor.call_args[1].values())
        assert "moje_modele" in passed, attribute


def test_models_dir_defaults_to_none():
    assert OrchestratorConfig().models_dir is None


def deepfake_mock():
    detector = MagicMock()
    detector.analyze_image.return_value = {"file": "a.png", "fake_probability": 0.8}
    detector.analyze_images.return_value = {"frames": [{}], "max_fake_probability": 0.6}
    detector.analyze_audio.return_value = {"file": "a.mp3", "max_fake_probability": 0.2}
    return detector


def test_deepfake_is_off_by_default():
    report = make_orchestrator(**image_mocks()).run(image="a.png")
    assert not [n for n in names(report) if n.startswith("deepfake")]


def test_image_gets_deepfake_step_after_faces():
    detector = deepfake_mock()
    config = OrchestratorConfig(deepfake=True)
    report = make_orchestrator(config, deepfake_detector=detector, **image_mocks()).run(image="a.png")
    assert names(report)[-1] == "deepfake_image"
    assert report.step("deepfake_image").data["fake_probability"] == 0.8
    detector.analyze_image.assert_called_once_with("a.png")


def test_video_gets_frame_and_audio_deepfake_steps():
    detector = deepfake_mock()
    mocks = video_mocks()
    mocks["video_extractor"].analyze.return_value["keyframes"] = [
        {"file_path": "kf/1.jpg"},
        {"file_path": "kf/2.jpg"},
    ]
    config = OrchestratorConfig(deepfake=True, keyframes_dir="kf")
    report = make_orchestrator(config, deepfake_detector=detector, **mocks).run(video="/x/ad.mp4")

    frames = report.step("deepfake_frames")
    assert frames.data["file"] == "ad.mp4" and frames.data["max_fake_probability"] == 0.6
    detector.analyze_images.assert_called_once_with(["kf/1.jpg", "kf/2.jpg"])
    detector.analyze_audio.assert_called_once_with("/x/ad.mp4")
    assert names(report).index("deepfake_frames") < names(report).index("audio_features")
    assert report.step("deepfake_audio").source == "video"


def test_audio_input_gets_deepfake_audio_step():
    detector = deepfake_mock()
    config = OrchestratorConfig(deepfake=True)
    report = make_orchestrator(
        config, deepfake_detector=detector, audio_extractor=audio_mock(), **text_mocks()
    ).run(audio="a.mp3")
    assert report.step("deepfake_audio", "audio").data["max_fake_probability"] == 0.2


def test_failing_deepfake_step_does_not_stop_the_run():
    detector = deepfake_mock()
    detector.analyze_image.side_effect = RuntimeError("brak modelu")
    config = OrchestratorConfig(deepfake=True)
    report = make_orchestrator(config, deepfake_detector=detector, **image_mocks()).run(image="a.png")
    step = report.step("deepfake_image")
    assert step.status == "error" and "brak modelu" in step.error
    assert "text_embeddings" in names(report)


def test_deepfake_detector_is_built_from_config():
    config = OrchestratorConfig(
        deepfake=True,
        deepfake_image_model="m/obraz",
        deepfake_audio_model="m/glos",
        deepfake_audio_fake_label="spoof",
        models_dir="modele",
        use_gpu=True,
    )
    with patch("feature_extractor.deepfake_extractor.DeepfakeDetector") as detector_class:
        detector = Orchestrator(config).deepfake_detector
    detector_class.assert_called_once_with(
        image_model="m/obraz",
        audio_model="m/glos",
        device="cuda",
        models_dir="modele",
        audio_fake_label="spoof",
    )
    assert detector is detector_class.return_value


def test_prefetch_includes_deepfake_models_only_when_enabled():
    detector = MagicMock()
    mocks = prefetch_mocks()
    off = make_orchestrator(OrchestratorConfig(), deepfake_detector=detector, **mocks).prefetch()
    assert "deepfake_image" not in off

    on = make_orchestrator(OrchestratorConfig(deepfake=True), deepfake_detector=detector, **prefetch_mocks()).prefetch()
    assert on["deepfake_image"] == "ok" and on["deepfake_audio"] == "ok"
    detector.load_image_model.assert_called_once_with()
    detector.load_audio_model.assert_called_once_with()


def test_ai_image_is_off_by_default():
    report = make_orchestrator(deepfake_detector=deepfake_mock(), **image_mocks()).run(image="a.png")
    assert not [n for n in names(report) if n.startswith("ai_")]


def test_image_gets_ai_image_step_independent_of_deepfake():
    detector = deepfake_mock()
    detector.analyze_image.return_value = {"file": "a.png", "fake_probability": 0.9}
    config = OrchestratorConfig(ai_image=True)
    report = make_orchestrator(config, ai_image_detector=detector, **image_mocks()).run(image="a.png")
    assert names(report)[-1] == "ai_image"
    assert "deepfake_image" not in names(report)
    assert report.step("ai_image").data["fake_probability"] == 0.9


def test_video_gets_ai_frames_step():
    detector = deepfake_mock()
    mocks = video_mocks()
    mocks["video_extractor"].analyze.return_value["keyframes"] = [{"file_path": "kf/1.jpg"}]
    config = OrchestratorConfig(ai_image=True, keyframes_dir="kf")
    report = make_orchestrator(config, ai_image_detector=detector, **mocks).run(video="/x/ad.mp4")
    assert report.step("ai_frames").data["max_fake_probability"] == 0.6
    detector.analyze_images.assert_called_once_with(["kf/1.jpg"])
    assert "deepfake_frames" not in names(report)


def test_ai_image_detector_is_built_from_config():
    config = OrchestratorConfig(
        ai_image=True, ai_image_model="m/ai", ai_image_fake_label="LABEL_1", models_dir="modele", use_gpu=True
    )
    with patch("feature_extractor.ai_image_extractor.AiImageDetector") as detector_class:
        detector = Orchestrator(config).ai_image_detector
    detector_class.assert_called_once_with(model="m/ai", device="cuda", models_dir="modele", fake_label="LABEL_1")
    assert detector is detector_class.return_value


def test_prefetch_includes_ai_image_model_only_when_enabled():
    detector = MagicMock()
    off = make_orchestrator(OrchestratorConfig(), ai_image_detector=detector, **prefetch_mocks()).prefetch()
    assert "ai_image" not in off
    on = make_orchestrator(OrchestratorConfig(ai_image=True), ai_image_detector=detector, **prefetch_mocks()).prefetch()
    assert on["ai_image"] == "ok"
    detector.load_image_model.assert_called_once_with()


def page_web(media, final_url="https://szybki.pl/oferta"):
    web = MagicMock()
    web.analyze.return_value.to_dict.return_value = {
        "page_text": "Okazja",
        "page_html": "<html></html>",
        "final_url": final_url,
        "errors": [],
        "media": media,
    }
    return web


def page_media_mocks(items):
    downloader = MagicMock()
    downloader.download.return_value = items
    paragraphs = MagicMock()
    return {"page_media_downloader": downloader, "paragraphs": paragraphs, **image_mocks(), **video_mocks()}


MEDIA = {
    "images": [{"url": "https://cdn/a.jpg"}],
    "videos": [{"url": "https://cdn/v.mp4", "kind": "file"}],
    "inline_images": 0,
    "emojis": {"total": 0},
}


def test_page_media_is_off_by_default():
    mocks = page_media_mocks([])
    report = make_orchestrator(web_extractor=page_web(MEDIA), **mocks).run(url="https://szybki.pl")
    assert "page_media" not in names(report)
    mocks["page_media_downloader"].download.assert_not_called()


def test_page_media_runs_image_and_video_steps_with_page_sources(tmp_path):
    items = [
        {"kind": "image", "url": "https://cdn/a.jpg", "status": "ok", "file_path": "/m/image_1.jpg"},
        {"kind": "image", "url": "https://cdn/b.jpg", "status": "skipped", "error": "za mały"},
        {"kind": "video", "url": "https://cdn/v.mp4", "status": "ok", "file_path": "/m/video_1.mp4"},
    ]
    mocks = page_media_mocks(items)
    config = OrchestratorConfig(page_media=True, page_media_dir=str(tmp_path), ai_image=True)
    mocks["ai_image_detector"] = deepfake_mock()
    report = make_orchestrator(config, web_extractor=page_web(MEDIA), **mocks).run(url="https://szybki.pl")

    assert report.step("page_media", "url").data == items
    assert report.step("ocr", "page_image") is not None
    assert report.step("ai_image", "page_image").status == "ok"
    assert report.step("video_features", "page_video") is not None
    assert "ocr" not in [s.name for s in report.steps if s.source == "image"]
    assert [s.source for s in report.steps if s.name == "ocr"] == ["page_image"]
    assert mocks["ai_image_detector"].analyze_image.call_count == 1

    folder = mocks["page_media_downloader"].download.call_args[0][1]
    assert str(folder).startswith(str(tmp_path))
    keyframes_dir = mocks["video_extractor"].analyze.call_args[0][1]
    assert keyframes_dir.startswith(str(folder)) and keyframes_dir.endswith("video_1")


def test_page_media_skips_when_page_has_no_media(tmp_path):
    empty = {"images": [], "videos": [], "inline_images": 0, "emojis": {"total": 0}}
    mocks = page_media_mocks([])
    config = OrchestratorConfig(page_media=True, page_media_dir=str(tmp_path))
    report = make_orchestrator(config, web_extractor=page_web(empty), **mocks).run(url="https://szybki.pl")
    assert "page_media" not in names(report)


def test_page_media_download_failure_is_a_step_error(tmp_path):
    mocks = page_media_mocks([])
    mocks["page_media_downloader"].download.side_effect = OSError("dysk pełny")
    config = OrchestratorConfig(page_media=True, page_media_dir=str(tmp_path))
    report = make_orchestrator(config, web_extractor=page_web(MEDIA), **mocks).run(url="https://szybki.pl")
    assert report.step("page_media", "url").status == "error"
    assert "text_embeddings" in names(report)


def graphic_png(tmp_path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (300, 300), "white")
    ImageDraw.Draw(image).ellipse((60, 60, 240, 240), fill=(30, 90, 200))
    path = tmp_path / "logo.png"
    image.save(path)
    return str(path)


def photo_png(tmp_path, name="foto.png"):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(3)
    path = tmp_path / name
    Image.fromarray(rng.integers(0, 255, (200, 300, 3), dtype="uint8")).save(path)
    return str(path)


def test_ai_image_is_skipped_for_logos_and_graphics(tmp_path):
    detector = deepfake_mock()
    config = OrchestratorConfig(ai_image=True)
    report = make_orchestrator(config, ai_image_detector=detector, **image_mocks()).run(image=graphic_png(tmp_path))
    step = report.step("ai_image")
    assert step.status == "skipped" and "grafika" in step.error
    detector.analyze_image.assert_not_called()


def test_ai_image_runs_for_photographs(tmp_path):
    detector = deepfake_mock()
    config = OrchestratorConfig(ai_image=True)
    report = make_orchestrator(config, ai_image_detector=detector, **image_mocks()).run(image=photo_png(tmp_path))
    assert report.step("ai_image").status == "ok"


def test_deepfake_is_skipped_when_image_has_no_faces():
    detector = deepfake_mock()
    mocks = image_mocks()
    mocks["faces"].extract_faces.return_value = []
    config = OrchestratorConfig(deepfake=True)
    report = make_orchestrator(config, deepfake_detector=detector, **mocks).run(image="a.png")
    step = report.step("deepfake_image")
    assert step.status == "skipped" and "twarzy" in step.error
    detector.analyze_image.assert_not_called()


def test_deepfake_still_runs_when_face_step_failed():
    detector = deepfake_mock()
    mocks = image_mocks()
    mocks["faces"].extract_faces.side_effect = RuntimeError("brak modelu twarzy")
    config = OrchestratorConfig(deepfake=True)
    report = make_orchestrator(config, deepfake_detector=detector, **mocks).run(image="a.png")
    assert report.step("deepfake_image").status == "ok"


def test_video_deepfake_uses_only_frames_with_faces_and_skips_without():
    detector = deepfake_mock()
    mocks = video_mocks()
    mocks["video_extractor"].analyze.return_value["keyframes"] = [
        {"file_path": "kf/1.jpg", "faces": [{"embedding": [1.0, 0.0]}]},
        {"file_path": "kf/2.jpg"},
    ]
    config = OrchestratorConfig(deepfake=True, keyframes_dir="kf")
    make_orchestrator(config, deepfake_detector=detector, **mocks).run(video="/x/ad.mp4")
    detector.analyze_images.assert_called_once_with(["kf/1.jpg"])

    mocks = video_mocks()
    mocks["video_extractor"].analyze.return_value.update(keyframes=[{"file_path": "kf/1.jpg"}], face_embeddings=[])
    report = make_orchestrator(config, deepfake_detector=deepfake_mock(), **mocks).run(video="/x/ad.mp4")
    assert report.step("deepfake_frames").status == "skipped"


def test_video_ai_frames_skip_graphic_frames(tmp_path):
    detector = deepfake_mock()
    mocks = video_mocks()
    mocks["video_extractor"].analyze.return_value["keyframes"] = [
        {"file_path": graphic_png(tmp_path)},
        {"file_path": photo_png(tmp_path)},
    ]
    config = OrchestratorConfig(ai_image=True, keyframes_dir="kf")
    make_orchestrator(config, ai_image_detector=detector, **mocks).run(video="/x/ad.mp4")
    assert detector.analyze_images.call_args[0][0] == [str(tmp_path / "foto.png")]
