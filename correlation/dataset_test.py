import json
from types import SimpleNamespace

from correlation.dataset import build_case_report, collect_case, find_cases, har_infrastructure
from correlation.profile import profile_from_report

HTML = "<html><body><div class='a'><p>Orlen daje pasywny dochód</p></div><script>x()</script></body></html>"


def entry(url, status=200, mime="text/html", ip=None, redirect="", kind="document"):
    data = {
        "request": {"url": url},
        "response": {"status": status, "redirectURL": redirect, "content": {"mimeType": mime}},
        "_resourceType": kind,
    }
    if ip:
        data["serverIPAddress"] = ip
    return data


def write_har(path, entries):
    path.write_text(json.dumps({"log": {"entries": entries}}), encoding="utf-8")
    return path


def make_case(tmp_path):
    root = tmp_path / "data"
    for name in ("10", "2", "1"):
        (root / name).mkdir(parents=True)
    case = root / "1"
    write_har(
        case / "shop.har",
        [
            entry("https://go.track.top/c", 302, redirect="https://www.scam.xyz/oferta", ip="1.1.1.1"),
            entry("https://www.scam.xyz/oferta", ip="[2001:db8::1]"),
            entry("https://connect.facebook.net/fbevents.js", mime="application/javascript", kind="script"),
            entry("https://ads.other.com/frame", kind="document", ip="9.9.9.9"),
        ],
    )
    (case / "Strona.html").write_text(HTML, encoding="utf-8")
    (case / "Zrzut.PNG").write_bytes(b"x")
    (case / "film.mp4").write_bytes(b"x")
    (case / "Strona_files").mkdir()
    (case / "Strona_files" / "hero.webp").write_bytes(b"x")
    (case / "capture" / "resources" / "image").mkdir(parents=True)
    (case / "capture" / "resources" / "image" / "logo.png").write_bytes(b"x")
    (case / "capture" / "screenshot.png").write_bytes(b"x")
    return root


class FakeOrchestrator:
    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace(keyframes_dir="")

    def run(self, **inputs):
        self.calls.append((inputs, self.config.keyframes_dir))
        (kind, value), = inputs.items()
        steps = {
            "text": [{"name": "named_entities", "source": "text", "status": "ok", "data": [
                {"text": "Orlen", "label": "orgName", "lemma": "Orlen"}]}],
            "image": [{"name": "visual_hashes", "source": "image", "status": "ok",
                       "data": {"phash": "ab", "dhash": "cd", "ahash": "ef"}}],
            "video": [{"name": "video_features", "source": "video", "status": "ok",
                       "data": {"dhashes": ["12"], "face_embeddings": []}}],
        }[kind]
        return SimpleNamespace(to_dict=lambda: {"steps": steps})


def test_find_cases_in_natural_order_and_skip_resources(tmp_path):
    root = make_case(tmp_path)
    cases = find_cases(root)
    assert [c.name for c in cases] == ["1", "2", "10"]
    assert cases[1].is_empty

    case = cases[0]
    assert [p.name for p in case.html] == ["Strona.html"]
    assert [p.name for p in case.har] == ["shop.har"]
    assert sorted(p.name for p in case.images) == ["Zrzut.PNG", "screenshot.png"]
    assert [p.name for p in case.videos] == ["film.mp4"]


def test_har_follows_redirects_of_main_document_only(tmp_path):
    har = write_har(
        tmp_path / "a.har",
        [
            entry("https://go.track.top/c", 302, redirect="/next", ip="1.1.1.1"),
            entry("https://go.track.top/next", 302, redirect="https://www.scam.xyz/oferta"),
            entry("https://www.scam.xyz/oferta", ip="[2001:db8::1]"),
            entry("https://ads.other.com/frame", ip="9.9.9.9"),
        ],
    )
    data = har_infrastructure(har)
    assert data["registered_domain"] == "track.top"
    assert data["final_url"] == "https://www.scam.xyz/oferta"
    assert [h["status"] for h in data["redirect_chain"]] == [302, 302, 200]
    assert data["dns"]["a"] == ["1.1.1.1", "2001:db8::1"]


def test_har_redirect_to_missing_entry_keeps_target(tmp_path):
    har = write_har(tmp_path / "a.har", [entry("https://a.pl/", 301, redirect="https://b.pl/x")])
    assert har_infrastructure(har)["final_url"] == "https://b.pl/x"


def test_har_without_entries_and_without_resource_types(tmp_path):
    assert har_infrastructure(write_har(tmp_path / "e.har", [])) is None
    plain = {"request": {"url": "https://x.pl/"}, "response": {"status": 200, "content": {"mimeType": "text/html"}}}
    assert har_infrastructure(write_har(tmp_path / "p.har", [plain]))["final_url"] == "https://x.pl/"


def test_case_report_collects_all_sources(tmp_path):
    case = collect_case(make_case(tmp_path) / "1")
    orchestrator = FakeOrchestrator()
    report = build_case_report(case, orchestrator, keyframes_root=tmp_path / "frames")

    names = [(s["name"], s["source"]) for s in report["steps"]]
    assert names[:2] == [("web_infrastructure", "har"), ("web_infrastructure", "html")]
    assert ("named_entities", "html") in names
    assert names.count(("visual_hashes", "image")) == 2
    assert ("video_features", "video") in names
    assert report["inputs"]["case"] == "1"
    assert report["inputs"]["har"] == ["shop.har"]

    text_call = orchestrator.calls[0][0]["text"]
    assert "Orlen daje pasywny dochód" in text_call and "x()" not in text_call
    video_call = orchestrator.calls[-1]
    assert video_call[1] == str(tmp_path / "frames" / "1" / "film")

    profile = profile_from_report(report)
    assert profile.domains == ["scam.xyz", "track.top"]
    assert profile.dom_hash
    assert profile.entities == ["orgName:orlen"]
    assert profile.phashes == ["ab", "ab"]


def test_media_can_be_skipped(tmp_path):
    case = collect_case(make_case(tmp_path) / "1")
    orchestrator = FakeOrchestrator()
    report = build_case_report(case, orchestrator, images=False, videos=False)
    assert [list(c[0]) for c in orchestrator.calls] == [["text"]]
    assert not any(s["source"] in ("image", "video") for s in report["steps"])


def test_broken_har_becomes_error_step(tmp_path):
    root = tmp_path / "d" / "1"
    root.mkdir(parents=True)
    (root / "bad.har").write_text("{", encoding="utf-8")
    report = build_case_report(collect_case(root), FakeOrchestrator())
    assert report["steps"][0]["status"] == "error"
    assert profile_from_report(report).is_empty
