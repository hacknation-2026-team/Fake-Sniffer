import json

from correlation.base import Base
from correlation.profile import Profile


def test_missing_file_gives_empty_base(tmp_path):
    assert len(Base(tmp_path / "none.json")) == 0


def test_add_assigns_sequential_ids(tmp_path):
    base = Base(tmp_path / "b.json")
    first = base.add(Profile(domains=["a.pl"]), "r1.json", label="Orlen")
    second = base.add(Profile(domains=["b.pl"]), "r2.json")
    assert (first.id, second.id) == ("inc-1", "inc-2")


def test_old_cluster_fields_in_file_are_ignored(tmp_path):
    path = tmp_path / "b.json"
    base = Base(path)
    base.add(Profile(domains=["a.pl"]), "r")
    data = json.loads(base.save().read_text(encoding="utf-8"))
    data["incidents"][0].update(campaign="campaign-1", family="family-1")
    path.write_text(json.dumps(data), encoding="utf-8")
    assert len(Base(path)) == 1


def test_save_and_reload(tmp_path):
    path = tmp_path / "sub" / "b.json"
    base = Base(path)
    base.add(Profile(domains=["a.pl"], text_embedding=[0.1, 0.2]), "r1.json", label="x")
    base.save()
    loaded = Base(path)
    assert len(loaded) == 1
    assert loaded.incidents[0].profile == base.incidents[0].profile
    assert loaded.get("inc-1").label == "x"
    assert loaded.get("inc-9") is None


def test_preview_is_stored_and_optional(tmp_path):
    base = Base(tmp_path / "b.json")
    base.add(Profile(domains=["a.pl"]), "r1", preview="/x/a.png")
    base.add(Profile(domains=["b.pl"]), "r2")
    base.save()
    loaded = Base(tmp_path / "b.json")
    assert [i.preview for i in loaded.incidents] == ["/x/a.png", None]
