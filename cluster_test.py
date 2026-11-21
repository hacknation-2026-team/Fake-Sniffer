import json

from cluster import main
from correlation.base import Base
from correlation.clustering import Clusters
from correlation.profile import Profile


def test_empty_base_fails(tmp_path, capsys):
    assert main(["--base", str(tmp_path / "b.json")]) == 1
    assert "pusta" in capsys.readouterr().err


def test_result_goes_to_separate_file_and_base_is_untouched(tmp_path, capsys):
    path = tmp_path / "b.json"
    base = Base(path)
    base.add(Profile(domains=["scam.top"]), "a")
    base.add(Profile(domains=["scam.top"]), "b")
    base.add(Profile(domains=["inna.pl"]), "c")
    base.save()
    before = path.read_text(encoding="utf-8")
    out = tmp_path / "clusters.json"

    assert main(["--base", str(path), "--out", str(out)]) == 0
    assert "Kampanie: 2" in capsys.readouterr().out
    assert path.read_text(encoding="utf-8") == before

    groups = json.loads(out.read_text(encoding="utf-8"))
    assert groups["campaigns"]["campaign-1"] == ["inc-1", "inc-2"]
    assert Clusters.load(out).campaigns["inc-3"] == "campaign-2"
