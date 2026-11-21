import json

import pytest

from correlation.profile import Profile, load_report, profile_from_report, registered_domain


def step(name, data, status="ok", source="x"):
    return {"name": name, "source": source, "status": status, "data": data}


def test_registered_domain():
    assert registered_domain("https://www.Promo.Example.com.pl/a?b=1") == "example.com.pl"
    assert registered_domain("https://bit.ly/x") == "bit.ly"
    assert registered_domain(None) is None
    assert registered_domain("") is None


def test_empty_report_gives_empty_profile():
    assert profile_from_report({"steps": []}).is_empty


def test_failed_and_empty_steps_are_ignored():
    report = {"steps": [step("faces", [{"embedding": [1.0]}], status="error"), step("ocr", None)]}
    assert profile_from_report(report).is_empty


def test_infrastructure_features():
    data = {
        "registered_domain": "scam.top",
        "final_url": "https://www.landing.xyz/oferta",
        "redirect_chain": [{"url": "https://track.ads.com/c", "status": 302, "location": None}],
        "dns": {"a": ["1.2.3.4"], "aaaa": [], "asns": [{"asn": 13335}]},
        "rdap": {"nameservers": ["NS1.Host.com"], "registrar": "Reg"},
        "dom_fingerprint": {"tag_class_hash": "abc", "path_hashes": ["p1", "p2"]},
        "tls": {"sha256": "ff"},
    }
    profile = profile_from_report({"steps": [step("web_infrastructure", data, status="partial")]})
    assert profile.domains == ["ads.com", "landing.xyz", "scam.top"]
    assert profile.ips == ["1.2.3.4"]
    assert profile.asns == ["13335"]
    assert profile.nameservers == ["ns1.host.com"]
    assert (profile.registrar, profile.dom_hash, profile.cert_sha256) == ("Reg", "abc", "ff")
    assert profile.dom_paths == ["p1", "p2"]


def test_text_features_are_merged_across_chunks():
    chunks = [
        {"embedding": [1.0, 0.0], "word_count": 3},
        {"embedding": [0.0, 1.0], "word_count": 1},
    ]
    entities = [{"text": "Orlenu", "label": "orgName", "lemma": "Orlen"}]
    report = {"steps": [step("text_embeddings", chunks), step("named_entities", entities)]}
    profile = profile_from_report(report)
    assert profile.entities == ["orgName:orlen"]
    assert profile.text_embedding[0] > profile.text_embedding[1]
    assert sum(v * v for v in profile.text_embedding) == pytest.approx(1.0, abs=1e-3)


def test_image_and_face_features():
    faces = [{"embedding": [0.1, 0.2], "identity": {"name": "Jan Kowalski", "similarity": 0.7}}]
    report = {
        "steps": [
            step("visual_hashes", {"phash": "aa", "dhash": "bb", "ahash": "cc"}),
            step("visual_embedding", [0.5, 0.5]),
            step("faces", faces),
        ]
    }
    profile = profile_from_report(report)
    assert (profile.phashes, profile.dhashes) == (["aa"], ["bb"])
    assert profile.visual_embedding == [0.5, 0.5]
    assert profile.identities == ["Jan Kowalski"]
    assert profile.face_embeddings == [[0.1, 0.2]]


def test_video_features_and_identities():
    video = {"dhashes": ["a1", "a2"], "visual_embedding": [1.0], "face_embeddings": [[0.3, 0.4]]}
    report = {"steps": [step("video_features", video), step("face_identities", [None, {"name": "Ewa"}])]}
    profile = profile_from_report(report)
    assert profile.dhashes == ["a1", "a2"]
    assert profile.face_embeddings == [[0.3, 0.4]]
    assert profile.identities == ["Ewa"]


def test_stripped_vectors_do_not_break_profile():
    report = {"steps": [step("text_embeddings", [{"embedding": None, "word_count": 2}])]}
    assert profile_from_report(report).text_embedding is None


def test_dict_round_trip_ignores_unknown_keys():
    profile = Profile(domains=["a.pl"], dom_hash="h")
    data = {**profile.to_dict(), "nowe_pole": 1}
    assert Profile.from_dict(data) == profile


def test_load_report(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"steps": []}), encoding="utf-8")
    assert load_report(path) == {"steps": []}
