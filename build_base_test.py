import json
import sys
from types import SimpleNamespace

import build_base
from correlation.base import Base
from correlation.dataset_test import FakeOrchestrator, make_case


def patch_orchestrator(monkeypatch):
    fake = FakeOrchestrator()
    module = SimpleNamespace(
        Orchestrator=lambda config: fake, OrchestratorConfig=lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setitem(sys.modules, "feature_extractor.orchestrator", module)
    return fake


def test_builds_base_and_reports(tmp_path, monkeypatch, capsys):
    root = make_case(tmp_path)
    fake = patch_orchestrator(monkeypatch)
    base_path, reports = tmp_path / "base.json", tmp_path / "reports"

    code = build_base.main([str(root), "--base", str(base_path), "--reports-dir", str(reports)])
    out = capsys.readouterr().out

    assert code == 0
    assert "Dodano 1, pominięto 2" in out
    base = Base(base_path)
    assert len(base) == 1 and base.incidents[0].label == "1"
    assert base.incidents[0].profile.domains == ["scam.xyz", "track.top"]
    saved = json.loads((reports / "1.json").read_text(encoding="utf-8"))
    assert saved["inputs"]["case"] == "1"
    assert fake.calls


def test_second_run_reuses_reports_and_replaces_incident(tmp_path, monkeypatch, capsys):
    root = make_case(tmp_path)
    fake = patch_orchestrator(monkeypatch)
    args = [str(root), "--base", str(tmp_path / "base.json"), "--reports-dir", str(tmp_path / "reports")]
    build_base.main(args)
    calls = len(fake.calls)

    build_base.main(args)
    assert len(fake.calls) == calls
    assert "z pliku" in capsys.readouterr().out
    assert len(Base(tmp_path / "base.json")) == 1

    build_base.main([*args, "--force"])
    assert len(fake.calls) == 2 * calls


def test_only_and_skip_flags(tmp_path, monkeypatch):
    root = make_case(tmp_path)
    fake = patch_orchestrator(monkeypatch)
    args = [str(root), "--base", str(tmp_path / "b.json"), "--reports-dir", str(tmp_path / "r")]

    build_base.main([*args, "--only", "2", "10"])
    assert len(Base(tmp_path / "b.json")) == 0

    build_base.main([*args, "--only", "1", "--no-images", "--no-videos"])
    assert [list(c[0]) for c in fake.calls] == [["text"]]


def test_missing_dataset_directory_is_an_error(tmp_path, capsys):
    try:
        build_base.main([str(tmp_path / "nie-ma")])
    except SystemExit as exc:
        assert exc.code == 2
    assert "nie jest katalogiem" in capsys.readouterr().err
