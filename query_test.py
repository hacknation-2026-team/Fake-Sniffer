import json

from cluster import main as cluster_main
from match import main as match_main
from query import main


def report(domain):
    return {
        "steps": [
            {"name": "web_infrastructure", "source": "url", "status": "ok", "data": {"registered_domain": domain}}
        ]
    }


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def prepared(tmp_path):
    base = str(tmp_path / "base.json")
    paths = [write(tmp_path / f"{n}.json", report(d)) for n, d in (("a", "scam.top"), ("b", "scam.top"), ("c", "inna.pl"))]
    assert match_main([*paths, "--base", base, "--add"]) == 0
    clusters = str(tmp_path / "clusters.json")
    assert cluster_main(["--base", base, "--out", clusters]) == 0
    return base, clusters


def test_requires_clustering_first(tmp_path, capsys):
    path = write(tmp_path / "r.json", report("scam.top"))
    assert main([path, "--base", str(tmp_path / "b.json"), "--clusters", str(tmp_path / "none.json")]) == 1
    assert "cluster.py" in capsys.readouterr().err


def test_new_report_gets_the_cluster_of_its_neighbours(tmp_path, capsys):
    base, clusters = prepared(tmp_path)
    capsys.readouterr()
    new = write(tmp_path / "new.json", report("scam.top"))
    out = tmp_path / "answer.json"
    assert main([new, "--base", base, "--clusters", clusters, "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "kampania: campaign-1 (2 incydentów)" in text
    answer = json.loads(out.read_text(encoding="utf-8"))[0]
    assert answer["campaign"] == "campaign-1"
    assert answer["verdict"] == "KNOWN_CAMPAIGN"


def test_unrelated_report_has_no_cluster(tmp_path, capsys):
    base, clusters = prepared(tmp_path)
    capsys.readouterr()
    new = write(tmp_path / "new.json", report("zupelnie-inna.pl"))
    assert main([new, "--base", base, "--clusters", clusters]) == 0
    text = capsys.readouterr().out
    assert "Brak dopasowania" in text and "kampania: brak" in text


def test_warns_when_base_is_newer_than_clusters(tmp_path, capsys):
    base, clusters = prepared(tmp_path)
    extra = write(tmp_path / "x.json", report("nowa.pl"))
    assert match_main([extra, "--base", base, "--add"]) == 0
    capsys.readouterr()
    assert main([extra, "--base", base, "--clusters", clusters]) == 0
    assert "uruchom cluster.py ponownie" in capsys.readouterr().out


def test_bad_reports_fail(tmp_path, capsys):
    base, clusters = prepared(tmp_path)
    empty = write(tmp_path / "e.json", {"steps": []})
    assert main([empty, str(tmp_path / "missing.json"), "--base", base, "--clusters", clusters]) == 1
    err = capsys.readouterr().err
    assert "nie zawiera cech" in err and "nie da się wczytać" in err
