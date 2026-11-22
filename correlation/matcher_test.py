from correlation.base import Base
from correlation.clustering import Clusters
from correlation.matcher import KNOWN, NO_MATCH, WEAK, classify
from correlation.profile import Profile


def known_base(tmp_path):
    base = Base(tmp_path / "b.json")
    base.add(
        Profile(domains=["scam.top"], text_embedding=[1.0, 0.0], identities=["Ewa"]), "a.json", label="Orlen"
    )
    base.add(Profile(domains=["other.xyz"], text_embedding=[0.0, 1.0]), "b.json")
    return base


def known_clusters():
    return Clusters(
        {"inc-1": "campaign-1", "inc-2": "campaign-2"},
        {"inc-1": "family-1", "inc-2": "family-2"},
    )


def test_empty_base_means_no_match(tmp_path):
    verdict = classify(Profile(domains=["a.pl"]), Base(tmp_path / "b.json"))
    assert verdict.verdict == NO_MATCH
    assert verdict.reasons == ["baza jest pusta"]


def test_nothing_to_compare_means_no_match(tmp_path):
    verdict = classify(Profile(entities=["a:b"]), known_base(tmp_path))
    assert verdict.verdict == NO_MATCH
    assert verdict.reasons == ["brak wspólnych cech do porównania"]


def test_matching_profile_joins_known_campaign(tmp_path):
    profile = Profile(text_embedding=[1.0, 0.0], identities=["Ewa"])
    verdict = classify(profile, known_base(tmp_path), known_clusters())
    assert verdict.verdict == KNOWN
    assert (verdict.campaign, verdict.family) == ("campaign-1", "family-1")
    assert not verdict.outdated_clusters
    assert verdict.neighbours[0].incident.id == "inc-1"
    assert verdict.top_features[0]["feature"] == "twarz"
    assert any("Ewa" in r for r in verdict.reasons)


def test_hard_infrastructure_match_is_enough(tmp_path):
    profile = Profile(domains=["scam.top"], text_embedding=[0.0, 1.0])
    verdict = classify(profile, known_base(tmp_path))
    assert verdict.verdict == KNOWN
    assert verdict.neighbours[0].incident.id == "inc-1"
    assert "wspólna domena scam.top" in verdict.reasons


def test_family_only_when_below_tight_threshold(tmp_path):
    profile = Profile(text_embedding=[0.85, 0.5268])
    verdict = classify(profile, known_base(tmp_path), known_clusters(), threshold=0.5, tight=0.99)
    assert verdict.verdict == KNOWN
    assert verdict.campaign is None
    assert verdict.family is not None


def test_weak_similarity(tmp_path):
    profile = Profile(text_embedding=[0.9, 0.436], domains=["nowa.pl"])
    verdict = classify(profile, known_base(tmp_path), known_clusters(), threshold=0.95, weak=0.1)
    assert verdict.verdict == WEAK
    assert verdict.campaign is None


def test_unrelated_profile_is_new(tmp_path):
    profile = Profile(text_embedding=[0.7, 0.7], domains=["zupelnie.pl"], identities=["Nikt"])
    base = known_base(tmp_path)
    verdict = classify(profile, base, threshold=0.95, weak=0.9)
    assert verdict.verdict == NO_MATCH
    assert verdict.to_dict()["neighbours"][0]["id"].startswith("inc-")
    assert verdict.top_features == [] and verdict.campaign is None and verdict.family is None
    assert len(verdict.reasons) == 1 and "nic w bazie nie jest podobne" in verdict.reasons[0]


def test_without_clusters_no_group_is_given(tmp_path):
    verdict = classify(Profile(text_embedding=[1.0, 0.0], identities=["Ewa"]), known_base(tmp_path))
    assert verdict.verdict == KNOWN
    assert (verdict.campaign, verdict.family) == (None, None)


def test_clusters_older_than_base_are_flagged(tmp_path):
    base = known_base(tmp_path)
    base.add(Profile(domains=["nowa.pl"]), "c.json")
    verdict = classify(Profile(domains=["scam.top"]), base, known_clusters())
    assert verdict.outdated_clusters
    assert verdict.to_dict()["outdated_clusters"] is True
