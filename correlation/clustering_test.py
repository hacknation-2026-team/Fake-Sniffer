from correlation.base import Base
from correlation.clustering import Clusters, build_graph, cluster
from correlation.profile import Profile


def incidents(tmp_path):
    base = Base(tmp_path / "b.json")
    base.add(Profile(domains=["scam.top"], text_embedding=[1.0, 0.0]), "a")
    base.add(Profile(domains=["scam.top"], text_embedding=[1.0, 0.0]), "b")
    base.add(Profile(domains=["other.xyz"], text_embedding=[0.0, 1.0]), "c")
    base.add(Profile(domains=["third.pl"], text_embedding=[0.0, 1.0]), "d")
    return base.incidents


def test_graph_has_edges_for_linked_pairs_only(tmp_path):
    graph = build_graph(incidents(tmp_path))
    assert set(graph.nodes) == {"inc-1", "inc-2", "inc-3", "inc-4"}
    assert graph.has_edge("inc-1", "inc-2")
    assert graph["inc-1"]["inc-2"]["hard"] is True
    assert not graph.has_edge("inc-1", "inc-3")


def test_campaigns_and_families(tmp_path):
    result = cluster(incidents(tmp_path))
    assert result.campaigns["inc-1"] == result.campaigns["inc-2"] == "campaign-1"
    assert result.campaigns["inc-3"] != result.campaigns["inc-1"]
    assert len(set(result.campaigns.values())) == 3
    assert result.families["inc-1"] == result.families["inc-2"]
    assert set(result.families) == {"inc-1", "inc-2", "inc-3", "inc-4"}


def test_families_are_broader_than_campaigns(tmp_path):
    base = Base(tmp_path / "b.json")
    base.add(Profile(text_embedding=[1.0, 0.0]), "a")
    base.add(Profile(text_embedding=[0.9, 0.436]), "b")
    result = cluster(base.incidents, threshold=0.5, tight=0.99)
    assert result.campaigns["inc-1"] != result.campaigns["inc-2"]
    assert result.families["inc-1"] == result.families["inc-2"]


def test_clustering_is_deterministic(tmp_path):
    items = incidents(tmp_path)
    assert cluster(items, seed=1) == cluster(items, seed=1)


def test_clusters_round_trip(tmp_path):
    result = cluster(incidents(tmp_path))
    path = result.save(tmp_path / "sub" / "clusters.json")
    loaded = Clusters.load(path)
    assert loaded == result
    assert loaded.created_at == result.created_at != ""
    assert loaded.campaign_members()["campaign-1"] == ["inc-1", "inc-2"]
    assert (loaded.threshold, loaded.tight) == (result.threshold, result.tight)
