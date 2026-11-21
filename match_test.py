import json

from match import main


def report(domain="scam.top", embedding=(1.0, 0.0)):
    return {
        "steps": [
            {"name": "web_infrastructure", "source": "url", "status": "ok", "data": {"registered_domain": domain}},
            {
                "name": "text_embeddings",
                "source": "url",
                "status": "ok",
                "data": [{"embedding": list(embedding), "word_count": 5}],
            },
        ]
    }


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_new_report_against_empty_base(tmp_path, capsys):
    path = write(tmp_path / "r.json", report())
    assert main([path, "--base", str(tmp_path / "base.json")]) == 0
    assert "Nowa kampania" in capsys.readouterr().out
    assert not (tmp_path / "base.json").exists()


def test_add_then_match_known(tmp_path, capsys):
    base = str(tmp_path / "base.json")
    first = write(tmp_path / "a.json", report())
    second = write(tmp_path / "b.json", report())
    out = tmp_path / "out" / "result.json"
    assert main([first, "--base", base, "--add", "--label", "Orlen"]) == 0
    assert main([second, "--base", base, "--out", str(out)]) == 0
    assert "Podobne do znanej kampanii" in capsys.readouterr().out
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result[0]["verdict"] == "KNOWN_CAMPAIGN"
    assert result[0]["neighbours"][0]["label"] == "Orlen"
    assert "wspólna domena scam.top" in result[0]["reasons"]


def test_added_reports_are_matched_against_earlier_ones(tmp_path, capsys):
    base = str(tmp_path / "base.json")
    paths = [write(tmp_path / f"{n}.json", report()) for n in "ab"]
    assert main([*paths, "--base", base, "--add"]) == 0
    out = capsys.readouterr().out
    assert "Nowa kampania" in out and "Podobne do znanej kampanii" in out
    incidents = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))["incidents"]
    assert len(incidents) == 2 and "campaign" not in incidents[0]


def test_unreadable_and_empty_reports_fail(tmp_path, capsys):
    empty = write(tmp_path / "e.json", {"steps": []})
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    code = main([empty, str(broken), str(tmp_path / "missing.json"), "--base", str(tmp_path / "b.json")])
    err = capsys.readouterr().err
    assert code == 1
    assert "nie zawiera cech" in err and "nie da się wczytać" in err
