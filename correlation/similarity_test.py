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
