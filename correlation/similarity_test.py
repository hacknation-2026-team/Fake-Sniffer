import pytest

from correlation.profile import Profile
from correlation.similarity import (
    THRESHOLD,
    WEIGHTS,
    compare,
    cosine,
    hamming,
    jaccard,
    min_hash_distance,
)


def test_helpers():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([0, 0], [1, 0]) == 0.0
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard([], []) == 0.0
    assert hamming("ff", "00") == 8
    assert hamming("ff", "fff") is None
    assert min_hash_distance(["ff", "0f"], ["0f"]) == 0
    assert min_hash_distance([], ["0f"]) is None


def test_no_shared_modalities_scores_zero():
    result = compare(Profile(domains=["a.pl"]), Profile(entities=["x:y"]))
    assert result.score == 0.0
    assert result.modalities == {}
    assert not result.linked()


def test_missing_modalities_do_not_lower_score():
    a = Profile(text_embedding=[1.0, 0.0])
    b = Profile(text_embedding=[1.0, 0.0], domains=["x.pl"])
    assert compare(a, b).score == pytest.approx(1.0)


def test_weighted_average_over_available_modalities():
    a = Profile(text_embedding=[1.0, 0.0], domains=["a.pl"])
    b = Profile(text_embedding=[1.0, 0.0], domains=["b.pl"])
    expected = WEIGHTS["text"] / (WEIGHTS["text"] + WEIGHTS["infra"])
    result = compare(a, b)
    assert result.modalities == {"infra": 0.0, "text": 1.0}
    assert result.score == pytest.approx(expected, abs=1e-3)


def test_shared_domain_is_hard_link_even_with_low_score():
    a = Profile(domains=["scam.top"], text_embedding=[1.0, 0.0], identities=["Ewa"])
    b = Profile(domains=["scam.top"], text_embedding=[0.0, 1.0], identities=["Jan"])
    result = compare(a, b)
    assert result.hard == ["wspólna domena scam.top"]
    assert result.score < THRESHOLD
    assert result.linked()


def test_shortener_domain_is_not_a_hard_link():
    result = compare(Profile(domains=["bit.ly"]), Profile(domains=["bit.ly"]))
    assert result.hard == []
    assert not result.linked()


@pytest.mark.parametrize(
    "field, value, text",
    [
        ("dom_hash", "h", "szablon"),
        ("cert_sha256", "c", "certyfikat"),
    ],
)
def test_other_hard_indicators(field, value, text):
    a = Profile(domains=["a.pl"], **{field: value})
    b = Profile(domains=["b.pl"], **{field: value})
    result = compare(a, b)
    assert len(result.hard) == 1 and text in result.hard[0]


def test_soft_infrastructure_signals():
    a = Profile(domains=["a.pl"], asns=["1"], nameservers=["ns.x"], dom_paths=["1", "2", "3", "4"])
    b = Profile(domains=["b.pl"], asns=["1"], nameservers=["ns.x"], dom_paths=["1", "2", "3", "9"])
    result = compare(a, b)
    assert result.hard == []
    assert result.modalities["infra"] == pytest.approx(0.6)
    assert len(result.reasons) == 3


def test_same_identity_wins_over_embeddings():
    a = Profile(identities=["Ewa"], face_embeddings=[[1.0, 0.0]])
    b = Profile(identities=["Ewa"], face_embeddings=[[0.0, 1.0]])
    assert compare(a, b).modalities["face"] == 1.0


def test_face_embeddings_are_compared_when_unidentified():
    close = compare(Profile(face_embeddings=[[1.0, 0.0]]), Profile(face_embeddings=[[1.0, 0.0]]))
    far = compare(Profile(face_embeddings=[[1.0, 0.0]]), Profile(face_embeddings=[[0.0, 1.0]]))
    assert close.modalities["face"] == 1.0
    assert far.modalities["face"] == 0.0


def test_visual_uses_hashes_and_embeddings():
    a = Profile(phashes=["0" * 16], visual_embedding=[1.0, 0.0])
    b = Profile(phashes=["0" * 15 + "1"], visual_embedding=[1.0, 0.0])
    result = compare(a, b)
    assert result.modalities["visual"] == pytest.approx((1 - 1 / 32 + 1) / 2, abs=1e-3)
    assert any("pHash" in r for r in result.reasons)


def test_visual_falls_back_to_dhash_for_video():
    a = Profile(dhashes=["0" * 16])
    b = Profile(dhashes=["f" * 16])
    assert compare(a, b).modalities["visual"] == 0.0


def test_text_blends_embedding_and_entities():
    a = Profile(text_embedding=[1.0, 0.0], entities=["orgName:orlen", "persName:tusk"])
    b = Profile(text_embedding=[1.0, 0.0], entities=["orgName:orlen"])
    result = compare(a, b)
    assert result.modalities["text"] == pytest.approx(0.7 + 0.3 * 0.5)
    assert any("orlen" in r for r in result.reasons)


def test_duplicate_rule():
    base = dict(phashes=["0" * 16], text_embedding=[1.0, 0.0])
    near = Profile(phashes=["0" * 15 + "3"], text_embedding=[1.0, 0.0])
    assert compare(Profile(**base), near).duplicate

    far = Profile(phashes=["0" * 12 + "ffff"], text_embedding=[1.0, 0.0])
    assert not compare(Profile(**base), far).duplicate

    different_text = Profile(phashes=["0" * 16], text_embedding=[0.8, 0.6])
    assert not compare(Profile(**base), different_text).duplicate


def test_contributions_sum_to_score_and_are_ranked():
    a = Profile(text_embedding=[1.0, 0.0], identities=["Ewa"])
    result = compare(a, Profile(text_embedding=[1.0, 0.0], identities=["Ewa"]))
    assert sum(result.contributions.values()) == pytest.approx(result.score, abs=1e-3)
    assert result.contributions["face"] > result.contributions["text"]


def test_shared_ip_is_only_a_soft_signal():
    result = compare(Profile(domains=["a.pl"], ips=["1.1.1.1"]), Profile(domains=["b.pl"], ips=["1.1.1.1"]))
    assert result.hard == []
    assert result.modalities["infra"] == 0.5
    assert not result.linked()


def test_shared_owner_tracker_is_hard_and_google_is_soft():
    from correlation.profile import Profile
    from correlation.similarity import compare

    a = Profile(trackers=["facebook_pixel:111", "google_analytics:G-AAA"])
    b = Profile(trackers=["facebook_pixel:111", "google_analytics:G-AAA"])
    c = Profile(trackers=["google_analytics:G-AAA"])

    assert compare(a, b).hard == ["wspólny identyfikator Facebook Pixel 111"]
    soft_only = compare(a, c)
    assert soft_only.hard == []
    assert "wspólny identyfikator Google Analytics G-AAA" in soft_only.reasons
    assert soft_only.modalities["infra"] == 0.8


def test_shared_redirect_signature_is_soft():
    from correlation.profile import Profile
    from correlation.similarity import compare

    a = Profile(param_signatures=["clickid,land_id,lpkey"])
    b = Profile(param_signatures=["clickid,land_id,lpkey"])

    result = compare(a, b)
    assert result.hard == [] and result.modalities["infra"] == 0.6


import pytest


@pytest.mark.parametrize("key, expected", [
    ("persName:rafał brzoska", True),
    ("orgName:inpost", True),
    ("placeName:polska", False),
    ("date:2026", False),
    ("orgName:3s", False),
    ("placeName:ue", False),
    ("persName:ab", False),
])
def test_informative_entity(key, expected):
    from correlation.similarity import informative_entity

    assert informative_entity(key) is expected


def _entity_profile(contexts, tones=None):
    from correlation.profile import Profile

    keys = ["persName:rafał brzoska", "placeName:polska", "date:2026"]
    return Profile(entities=keys, entity_contexts=contexts, entity_tones=tones or {})


def test_shared_entity_counts_only_in_similar_context():
    from correlation.similarity import compare

    same = compare(
        _entity_profile({"persName:rafał brzoska": [1.0, 0.0]}),
        _entity_profile({"persName:rafał brzoska": [0.95, 0.1]}),
    )
    other = compare(
        _entity_profile({"persName:rafał brzoska": [1.0, 0.0]}),
        _entity_profile({"persName:rafał brzoska": [0.2, 1.0]}),
    )

    assert "wspólne nazwy własne w podobnym kontekście: rafał brzoska" in same.reasons
    assert not any("polska" in r or "2026" in r for r in same.reasons)
    assert same.modalities["text"] == 1.0
    assert other.modalities["text"] == 0.0
    assert "te same nazwy w innym kontekście (nie liczą się): rafał brzoska" in other.reasons


def test_opposite_tone_does_not_count():
    from correlation.similarity import compare

    context = {"persName:rafał brzoska": [1.0, 0.0]}
    result = compare(
        _entity_profile(context, {"persName:rafał brzoska": "negatywny"}),
        _entity_profile(context, {"persName:rafał brzoska": "pozytywny"}),
    )

    assert result.modalities["text"] == 0.0
    assert "te same nazwy o przeciwnym wydźwięku (nie liczą się): rafał brzoska" in result.reasons


def test_profiles_without_context_still_match_informative_entities():
    from correlation.profile import Profile
    from correlation.similarity import compare

    a = Profile(entities=["persName:rafał brzoska", "placeName:polska"])
    b = Profile(entities=["persName:rafał brzoska", "placeName:polska"])

    assert compare(a, b).modalities["text"] == 1.0


def test_entity_without_context_on_one_side_does_not_count():
    from correlation.profile import Profile
    from correlation.similarity import match_entities

    a = Profile(entities=["placeName:kuwait", "orgName:inpost"], entity_contexts={"orgName:inpost": [1.0, 0.0]})
    b = Profile(entities=["placeName:kuwait", "orgName:inpost"], entity_contexts={"orgName:inpost": [1.0, 0.0]})

    matched = match_entities(a, b)

    assert matched.shared == ["orgName:inpost"]
    assert matched.in_list == ["placeName:kuwait"]
