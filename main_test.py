import json
import pytest

import main
from feature_extractor.orchestrator import OrchestratorConfig, Report, StepResult


class FakeOrchestrator:
    instances = []
    steps = [StepResult("named_entities", "text", "ok", 0.2, [{"text": "Tusk"}])]

    def __init__(self, config):
        self.config = config
        self.calls = []
        FakeOrchestrator.instances.append(self)

    def run(self, **inputs):
        self.calls.append(inputs)
        given = {k: v for k, v in inputs.items() if v}
        return Report(created_at="2026-11-21T00:00:00+00:00", inputs=given, steps=list(self.steps))


@pytest.fixture(autouse=True)
def fake_orchestrator(monkeypatch):
    FakeOrchestrator.instances = []
    FakeOrchestrator.steps = [StepResult("named_entities", "text", "ok", 0.2, [{"text": "Tusk"}])]
    monkeypatch.setattr(main, "Orchestrator", FakeOrchestrator)
    return FakeOrchestrator


def test_no_input_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main.main([])
    assert exit_info.value.code == 2
    assert "--url" in capsys.readouterr().err


def test_text_run_writes_json_report_and_prints_summary(tmp_path, capsys):
    out = tmp_path / "out" / "raport.json"

    code = main.main(["--text", "Zażółć gęślą jaźń", "--out", str(out)])

    assert code == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["inputs"] == {"text": "Zażółć gęślą jaźń"}
    assert saved["steps"][0]["name"] == "named_entities"
    printed = capsys.readouterr().out
    assert "named_entities" in printed and str(out) in printed


def test_inputs_are_passed_to_orchestrator(tmp_path):
    main.main(["--url", "https://a.pl", "--image", "a.png", "--video", "a.mp4", "--out", str(tmp_path / "r.json")])

    assert FakeOrchestrator.instances[0].calls == [
        {"url": "https://a.pl", "text": None, "image": "a.png", "video": "a.mp4", "audio": None}
    ]


def test_text_and_text_file_are_combined(tmp_path):
    note = tmp_path / "artykul.txt"
    note.write_text("Treść z pliku", encoding="utf-8")

    main.main(["--text", "Wstęp", "--text-file", str(note), "--out", str(tmp_path / "r.json")])

    assert FakeOrchestrator.instances[0].calls[0]["text"] == "Wstęp\n\nTreść z pliku"


def test_missing_text_file_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main.main(["--text-file", str(tmp_path / "brak.txt")])
    assert exit_info.value.code == 2
    assert "brak.txt" in capsys.readouterr().err


def test_flags_are_mapped_to_config(tmp_path):
    main.main([
        "--text", "x", "--gpu", "--no-vectors", "--faces-dir", "refs", "--keyframes-dir", "kf",
        "--entity", "BRAND=Baltic Pipeline", "--entity", "BRAND=Onet News",
        "--embedder-model", "intfloat/multilingual-e5-large", "--out", str(tmp_path / "r.json"),
    ])

    config = FakeOrchestrator.instances[0].config
    assert config.use_gpu is True
    assert config.include_vectors is False
    assert config.reference_faces_dir == "refs"
    assert config.keyframes_dir == "kf"
    assert config.entity_patterns == [("BRAND", "Baltic Pipeline"), ("BRAND", "Onet News")]
    assert config.embedder_model == "intfloat/multilingual-e5-large"


def test_defaults_match_orchestrator_config(tmp_path):
    main.main(["--text", "x", "--out", str(tmp_path / "r.json")])

    config = FakeOrchestrator.instances[0].config
    defaults = OrchestratorConfig()
    assert config.use_gpu is False and config.include_vectors is True
    assert config.embedder_model == defaults.embedder_model
    assert config.keyframes_dir == defaults.keyframes_dir


def test_audio_alone_is_enough(tmp_path):
    code = main.main(["--audio", "nagranie.mp3", "--out", str(tmp_path / "r.json")])

    assert code == 0
    assert FakeOrchestrator.instances[0].calls[0]["audio"] == "nagranie.mp3"


def test_no_input_error_mentions_audio(capsys):
    with pytest.raises(SystemExit):
        main.main([])
    assert "--audio" in capsys.readouterr().err


def test_speech_flags_are_mapped_to_config(tmp_path):
    main.main([
        "--video", "a.mp4", "--whisper-model", "small", "--speech-language", "pl",
        "--spectrograms-dir", "spektrogramy", "--out", str(tmp_path / "r.json"),
    ])

    config = FakeOrchestrator.instances[0].config
    assert config.whisper_model == "small"
    assert config.speech_language == "pl"
    assert config.spectrograms_dir == "spektrogramy"


def test_speech_defaults_match_orchestrator_config(tmp_path):
    main.main(["--audio", "a.mp3", "--out", str(tmp_path / "r.json")])

    config = FakeOrchestrator.instances[0].config
    defaults = OrchestratorConfig()
    assert config.whisper_model == defaults.whisper_model
    assert config.speech_language == defaults.speech_language
    assert config.spectrograms_dir == defaults.spectrograms_dir


@pytest.mark.parametrize("value", ["BRAND", "=fraza", "BRAND=", " = "])
def test_invalid_entity_is_a_usage_error(value, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main.main(["--text", "x", "--entity", value])
    assert exit_info.value.code == 2
    assert "ETYKIETA=FRAZA" in capsys.readouterr().err


def test_exit_code_is_one_when_a_step_failed(tmp_path, capsys):
    FakeOrchestrator.steps = [StepResult("ocr", "image", "error", 1.0, None, "RuntimeError: boom")]

    code = main.main(["--image", "a.png", "--out", str(tmp_path / "r.json")])

    assert code == 1
    assert "RuntimeError: boom" in capsys.readouterr().out
    assert (tmp_path / "r.json").exists()


def test_partial_step_does_not_fail_the_run(tmp_path):
    FakeOrchestrator.steps = [StepResult("web_infrastructure", "url", "partial", 1.0, {}, "tls: timeout")]
    assert main.main(["--url", "https://a.pl", "--out", str(tmp_path / "r.json")]) == 0


def test_models_dir_is_mapped_to_config(tmp_path):
    main.main(["--text", "x", "--models-dir", "moje_modele", "--out", str(tmp_path / "r.json")])
    assert FakeOrchestrator.instances[0].config.models_dir == "moje_modele"


def test_download_models_needs_no_other_input(capsys):
    FakeOrchestrator.prefetch = lambda self: {"ocr": "ok", "browser": "ok"}

    code = main.main(["--download-models"])

    assert code == 0
    printed = capsys.readouterr().out
    assert "ocr" in printed and "browser" in printed
    assert FakeOrchestrator.instances[0].calls == []


def test_download_models_returns_one_when_something_failed(capsys):
    FakeOrchestrator.prefetch = lambda self: {"ocr": "ok", "faces": "RuntimeError: brak sieci"}

    assert main.main(["--download-models"]) == 1
    assert "RuntimeError: brak sieci" in capsys.readouterr().out


def test_download_models_respects_models_dir():
    FakeOrchestrator.prefetch = lambda self: {"ocr": "ok"}
    main.main(["--download-models", "--models-dir", "dysk"])
    assert FakeOrchestrator.instances[0].config.models_dir == "dysk"


def test_deepfake_flag_is_mapped_to_config(tmp_path):
    main.main(["--text", "x", "--out", str(tmp_path / "a.json")])
    main.main(["--text", "x", "--deepfake", "--out", str(tmp_path / "b.json")])
    assert [o.config.deepfake for o in FakeOrchestrator.instances] == [False, True]
