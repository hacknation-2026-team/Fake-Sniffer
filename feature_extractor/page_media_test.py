import io
from pathlib import Path

import pytest
from PIL import Image

from feature_extractor.page_media import (
    MediaDownloader,
    MediaLimits,
    extract_media,
    find_emojis,
    video_kind,
)

PUBLIC = [(None, None, None, None, ("93.184.216.34", 0))]


def png(width=200, height=200, color=(200, 10, 10)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, "PNG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, body=b"", status=200, location=None):
        self.body = body
        self.status_code = status
        self.headers = {"Location": location} if location else {}
        self.closed = False

    @property
    def is_redirect(self):
        return "Location" in self.headers

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, size):
        for i in range(0, len(self.body), size):
            yield self.body[i : i + size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        assert kwargs["allow_redirects"] is False
        return self.routes[url]


def downloader(routes, resolver=lambda host, port: PUBLIC, **limits):
    return MediaDownloader(MediaLimits(**limits), FakeSession(routes), resolver)


def test_find_emojis_counts_unique_and_density():
    result = find_emojis("Wygrałeś 🎁🎁 odbierz 🔥 teraz 👉")
    assert result["total"] == 4
    assert result["unique"] == 3
    assert result["top"][0] == ["🎁", 2]
    assert result["density"] > 0


def test_find_emojis_handles_flags_and_zwj_sequences():
    result = find_emojis("🇵🇱 i 👨‍👩‍👧")
    assert result["total"] == 2


def test_find_emojis_empty_text():
    assert find_emojis("") == {"total": 0, "unique": 0, "density": 0.0, "top": []}
    assert find_emojis("zwykły tekst")["total"] == 0


def test_extract_media_finds_images_videos_and_resolves_relative_urls():
    html = """
    <html><head><meta property="og:image" content="/og.jpg"></head><body>
      <img src="a.jpg" alt="promo" width="300" height="200">
      <img data-src="lazy.jpg">
      <img srcset="small.jpg 100w, big.jpg 800w">
      <img src="data:image/png;base64,AAAA">
      <div style="background: url('/bg.png')"></div>
      <video poster="/poster.jpg"><source src="clip.mp4"></video>
      <video src="https://cdn.example/live.m3u8"></video>
      <iframe src="https://www.youtube.com/embed/xyz"></iframe>
      <iframe src="https://other.example/page"></iframe>
    </body></html>
    """
    media = extract_media(html, "https://shop.example/offer/", text="🔥 okazja")
    urls = [i["url"] for i in media["images"]]
    assert "https://shop.example/offer/a.jpg" in urls
    assert "https://shop.example/offer/lazy.jpg" in urls
    assert "https://shop.example/offer/big.jpg" in urls
    assert "https://shop.example/offer/small.jpg" not in urls
    assert "https://shop.example/og.jpg" in urls
    assert "https://shop.example/bg.png" in urls
    assert "https://shop.example/poster.jpg" in urls
    first = next(i for i in media["images"] if i["url"].endswith("a.jpg"))
    assert (first["alt"], first["width"], first["height"]) == ("promo", 300, 200)
    assert media["inline_images"] == 1
    videos = {v["url"]: v["kind"] for v in media["videos"]}
    assert videos["https://shop.example/offer/clip.mp4"] == "file"
    assert videos["https://cdn.example/live.m3u8"] == "hls"
    assert videos["https://www.youtube.com/embed/xyz"] == "embed"
    assert "https://other.example/page" not in videos
    assert media["emojis"]["total"] == 1


def test_extract_media_skips_icons_and_deduplicates():
    html = '<img src="/logo.svg"><img src="/favicon.ico"><img src="/x.jpg"><img src="/x.jpg">'
    media = extract_media(html, "https://a.example/")
    assert [i["url"] for i in media["images"]] == ["https://a.example/x.jpg"]


def test_extract_media_adds_network_resources():
    resources = [
        {"url": "https://cdn.example/hero.jpg", "type": "image", "content_type": "image/jpeg"},
        {"url": "https://cdn.example/stream.m3u8", "type": "media", "content_type": "application/vnd.apple.mpegurl"},
        {"url": "https://cdn.example/clip", "type": "other", "content_type": "video/mp4"},
        {"url": "data:image/png;base64,AA", "type": "image", "content_type": None},
    ]
    media = extract_media("<p>x</p>", "https://a.example/", resources=resources)
    assert [i["url"] for i in media["images"]] == ["https://cdn.example/hero.jpg"]
    assert media["images"][0]["source"] == "network"
    assert {(v["url"], v["kind"]) for v in media["videos"]} == {
        ("https://cdn.example/stream.m3u8", "hls"),
        ("https://cdn.example/clip", "file"),
    }


def test_extract_media_without_html():
    media = extract_media("", "https://a.example/")
    assert media["images"] == [] and media["videos"] == []


def test_video_kind():
    assert video_kind("https://a/x.m3u8?token=1") == "hls"
    assert video_kind("https://a/x.mp4") == "file"


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1", "fd00::1"]
)
def test_private_addresses_are_rejected(tmp_path, address):
    resolver = lambda host, port: [(None, None, None, None, (address, 0))]
    d = downloader({}, resolver)
    result = d.download({"images": [{"url": "http://evil.example/a.png"}], "videos": []}, tmp_path)
    assert result[0]["status"] == "skipped"
    assert "niepubliczny" in result[0]["error"]
    assert d.session.calls == []


def test_mixed_resolution_with_private_address_is_rejected(tmp_path):
    resolver = lambda host, port: PUBLIC + [(None, None, None, None, ("10.0.0.1", 0))]
    d = downloader({}, resolver)
    result = d.download({"images": [{"url": "http://evil.example/a.png"}], "videos": []}, tmp_path)
    assert result[0]["status"] == "skipped"


def test_non_http_scheme_is_rejected(tmp_path):
    d = downloader({})
    result = d.download({"images": [{"url": "file:///etc/passwd"}], "videos": []}, tmp_path)
    assert result[0]["status"] == "skipped"
    assert "http" in result[0]["error"]


def test_redirect_to_private_host_is_blocked(tmp_path):
    def resolver(host, port):
        ip = "127.0.0.1" if host == "internal.local" else "93.184.216.34"
        return [(None, None, None, None, (ip, 0))]

    routes = {"http://ok.example/a.png": FakeResponse(status=302, location="http://internal.local/secret.png")}
    d = downloader(routes, resolver)
    result = d.download({"images": [{"url": "http://ok.example/a.png"}], "videos": []}, tmp_path)
    assert result[0]["status"] == "skipped"
    assert d.session.calls == ["http://ok.example/a.png"]


def test_redirect_loop_is_cut(tmp_path):
    routes = {"http://a.example/x.png": FakeResponse(status=302, location="http://a.example/x.png")}
    d = downloader(routes)
    result = d.download({"images": [{"url": "http://a.example/x.png"}], "videos": []}, tmp_path)
    assert "przekierowań" in result[0]["error"]


def test_download_saves_image_and_reports_size(tmp_path):
    routes = {"https://a.example/a.png": FakeResponse(png(300, 200))}
    result = downloader(routes).download({"images": [{"url": "https://a.example/a.png"}], "videos": []}, tmp_path)
    item = result[0]
    assert item["status"] == "ok"
    assert (item["width"], item["height"]) == (300, 200)
    assert Path(item["file_path"]).read_bytes() == png(300, 200)
    assert Path(item["file_path"]).suffix == ".png"


def test_download_skips_small_duplicate_broken_and_oversized(tmp_path):
    big = png(200, 200)
    routes = {
        "https://a.example/small.png": FakeResponse(png(50, 50)),
        "https://a.example/one.png": FakeResponse(big),
        "https://a.example/two.png": FakeResponse(big),
        "https://a.example/broken.png": FakeResponse(b"not an image"),
        "https://a.example/missing.png": FakeResponse(status=404),
        "https://a.example/huge.png": FakeResponse(big),
    }
    images = [{"url": u} for u in routes]
    d = downloader(routes, image_bytes=len(big) - 1)
    result = d.download({"images": images, "videos": []}, tmp_path)
    assert all(r["status"] == "skipped" for r in result)

    d = downloader({k: v for k, v in routes.items() if "huge" not in k})
    for response in d.session.routes.values():
        response.closed = False
    result = d.download({"images": [{"url": u} for u in d.session.routes], "videos": []}, tmp_path / "b")
    status = {r["url"].rsplit("/", 1)[1]: r["status"] for r in result}
    assert status == {
        "small.png": "skipped",
        "one.png": "ok",
        "two.png": "skipped",
        "broken.png": "skipped",
        "missing.png": "skipped",
    }
    assert "za mały" in next(r for r in result if r["url"].endswith("small.png"))["error"]
    assert "duplikat" in next(r for r in result if r["url"].endswith("two.png"))["error"]
    assert all(r.closed for r in d.session.routes.values())


def test_download_keeps_only_largest_images(tmp_path):
    sizes = {"a": 150, "b": 400, "c": 250}
    routes = {f"https://a.example/{k}.png": FakeResponse(png(v, v, (v % 255, 0, 0))) for k, v in sizes.items()}
    d = downloader(routes, max_images=2)
    result = d.download({"images": [{"url": u} for u in routes], "videos": []}, tmp_path)
    ok = sorted(r["width"] for r in result if r["status"] == "ok")
    assert ok == [250, 400]
    dropped = next(r for r in result if r["status"] == "skipped")
    assert "file_path" not in dropped
    assert len(list(tmp_path.iterdir())) == 2


def test_download_video_file_and_limits(tmp_path):
    routes = {
        "https://a.example/1.mp4": FakeResponse(b"x" * 1000),
        "https://a.example/2.webm": FakeResponse(b"y" * 1000),
        "https://a.example/3.mp4": FakeResponse(b"z" * 1000),
    }
    videos = [{"url": u, "kind": "file"} for u in routes]
    videos.append({"url": "https://www.youtube.com/embed/x", "kind": "embed"})
    result = downloader(routes, max_videos=2).download({"images": [], "videos": videos}, tmp_path)
    assert [r["status"] for r in result] == ["ok", "ok", "skipped", "skipped"]
    assert result[0]["bytes"] == 1000
    assert Path(result[1]["file_path"]).suffix == ".webm"
    assert "osadzone" in result[3]["error"]
    assert "zbyt wiele" in result[2]["error"]


def test_video_over_limit_is_removed(tmp_path):
    routes = {"https://a.example/1.mp4": FakeResponse(b"x" * 5000)}
    result = downloader(routes, video_bytes=1000).download(
        {"images": [], "videos": [{"url": "https://a.example/1.mp4", "kind": "file"}]}, tmp_path
    )
    assert result[0]["status"] == "skipped"
    assert list(tmp_path.glob("video_*")) == []


def test_hls_concatenates_segments_up_to_duration(tmp_path):
    playlist = "#EXTM3U\n#EXTINF:10.0,\ns1.ts\n#EXTINF:10.0,\ns2.ts\n#EXTINF:10.0,\ns3.ts\n#EXTINF:10.0,\ns4.ts\n#EXT-X-ENDLIST\n"
    routes = {"https://a.example/v/index.m3u8": FakeResponse(playlist.encode())}
    for i in range(1, 5):
        routes[f"https://a.example/v/s{i}.ts"] = FakeResponse(f"seg{i}".encode())
    d = downloader(routes, hls_seconds=25)
    result = d.download({"images": [], "videos": [{"url": "https://a.example/v/index.m3u8", "kind": "hls"}]}, tmp_path)
    assert result[0]["status"] == "ok"
    assert Path(result[0]["file_path"]).suffix == ".ts"
    assert Path(result[0]["file_path"]).read_bytes() == b"seg1seg2seg3"
    assert "https://a.example/v/s4.ts" not in d.session.calls


def test_hls_master_playlist_uses_first_variant(tmp_path):
    master = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=800000\nlow/index.m3u8\n#EXT-X-STREAM-INF:BANDWIDTH=2000000\nhigh/index.m3u8\n"
    media = "#EXTM3U\n#EXTINF:5,\nseg.ts\n"
    routes = {
        "https://a.example/master.m3u8": FakeResponse(master.encode()),
        "https://a.example/low/index.m3u8": FakeResponse(media.encode()),
        "https://a.example/low/seg.ts": FakeResponse(b"DATA"),
    }
    result = downloader(routes).download(
        {"images": [], "videos": [{"url": "https://a.example/master.m3u8", "kind": "hls"}]}, tmp_path
    )
    assert result[0]["status"] == "ok"
    assert Path(result[0]["file_path"]).read_bytes() == b"DATA"


@pytest.mark.parametrize(
    "playlist, reason",
    [
        ("#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI=\"k\"\n#EXTINF:5,\ns.ts\n", "szyfrowany"),
        ("#EXTM3U\n#EXT-X-MAP:URI=\"init.mp4\"\n#EXTINF:5,\ns.m4s\n", "fMP4"),
        ("#EXTM3U\n#EXT-X-ENDLIST\n", "pusta"),
    ],
)
def test_hls_unsupported_playlists_are_skipped(tmp_path, playlist, reason):
    routes = {"https://a.example/i.m3u8": FakeResponse(playlist.encode())}
    result = downloader(routes).download(
        {"images": [], "videos": [{"url": "https://a.example/i.m3u8", "kind": "hls"}]}, tmp_path
    )
    assert result[0]["status"] == "skipped"
    assert reason in result[0]["error"]
    assert list(tmp_path.glob("video_*")) == []


def test_img_with_srcset_gives_one_candidate():
    html = '<img src="s.jpg" srcset="s.jpg 1x, l.jpg 2x">'
    media = extract_media(html, "https://a.example/")
    assert [i["url"] for i in media["images"]] == ["https://a.example/l.jpg"]


def test_declared_small_images_are_dropped():
    html = (
        '<img src="/a.png" width="40" height="40">'
        '<img src="/b.png?w=64&h=64">'
        '<img src="/c.png?w=750&h=900">'
        '<img src="/d.png">'
    )
    urls = [i["url"] for i in extract_media(html, "https://a.example/")["images"]]
    assert urls == ["https://a.example/c.png?w=750&h=900", "https://a.example/d.png"]


def test_find_page_url_reads_canonical_in_any_attribute_order():
    from feature_extractor.page_media import find_page_url

    assert find_page_url('<link href="https://a.pl/x" rel="canonical">') == "https://a.pl/x"
    assert find_page_url('<meta property="og:url" content="https://b.pl/y">') == "https://b.pl/y"
    assert find_page_url('<link rel="canonical" href="/wzgledny">') is None
    assert find_page_url("<p>nic</p>") is None
