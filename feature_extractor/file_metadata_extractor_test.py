import av
import json
import numpy as np
import pytest
from fractions import Fraction
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from feature_extractor.file_metadata_extractor import FileMetadataExtractor, find_trackers, url_signals


@pytest.fixture
def extractor() -> FileMetadataExtractor:
    return FileMetadataExtractor()


def jpeg_with_exif(path: Path, software: str = "Adobe Photoshop 26.2", gps: bool = False) -> str:
    image = Image.new("RGB", (64, 48), (200, 30, 30))
    exif = image.getexif()
    exif[0x010F] = "Apple"
    exif[0x0110] = "iPhone 15"
    exif[0x0131] = software
    if gps:
        exif[0x8825] = {1: "N", 2: (52.0, 13.0, 30.0), 3: "E", 4: (21.0, 0.0, 36.0)}
    image.save(path, "JPEG", exif=exif)
    return str(path)


def write_mp4(path: Path, comment: str) -> str:
    with av.open(str(path), "w") as out:
        out.metadata["comment"] = comment
        out.metadata["encoder"] = "Lavf61.1.100"
        stream = out.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(5):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 40, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return str(path)


def test_image_exif_camera_software_and_gps(extractor, tmp_path):
    path = jpeg_with_exif(tmp_path / "zdjęcie.jpg", gps=True)

    result = extractor.extract(path)

    assert result["kind"] == "image" and (result["width"], result["height"]) == (64, 48)
    assert result["camera"] == "Apple iPhone 15"
    assert result["software"] == "Adobe Photoshop 26.2"
    assert result["gps"] == {"latitude": 52.225, "longitude": 21.01}
    assert result["export_apps"] == ["Adobe Photoshop"]
    assert result["ai_hints"] == []


def test_png_generation_parameters_are_an_ai_hint(extractor, tmp_path):
    info = PngInfo()
    info.add_text("parameters", "a doctor holding pills, Steps: 30, Sampler: Euler a, Model: sdxl")
    path = tmp_path / "baner.png"
    Image.new("RGB", (32, 32)).save(path, pnginfo=info)

    result = extractor.extract_image(str(path))

    assert "parametry generowania obrazu w polu 'parameters'" in result["ai_hints"]
    assert result["text_chunks"]["parameters"].startswith("a doctor")


def test_c2pa_manifest_marks_synthetic_image(extractor, tmp_path):
    path = tmp_path / "ai.png"
    Image.new("RGB", (32, 32)).save(path)
    manifest = b"jumbc2pa\x00claim_generator ChatGPT OpenAI c2pa.created digitalSourceType trainedAlgorithmicMedia"
    path.write_bytes(path.read_bytes() + manifest)

    result = extractor.extract_image(str(path))

    credentials = result["content_credentials"]
    assert credentials["synthetic"] is True
    assert credentials["generators"] == ["ChatGPT", "OpenAI"]
    assert credentials["actions"] == ["created"]
    assert result["ai_hints"][0] == "manifest C2PA: obraz wygenerowany przez AI (ChatGPT, OpenAI)"


def test_image_without_metadata(extractor, tmp_path):
    path = tmp_path / "czysty.webp"
    Image.new("RGB", (16, 16)).save(path, "WEBP")

    result = extractor.extract(str(path))

    assert result["software"] is None and result["gps"] is None and result["content_credentials"] is None
    assert result["tools"] == [] and result["ai_hints"] == []


def test_video_tags_and_editor_flags(extractor, tmp_path):
    flags = {"data": {"editType": "default", "is_use_ai_video_generation": 1, "is_use_ai_image_generation": 0}}
    path = write_mp4(tmp_path / "reklama.mp4", json.dumps(flags))

    result = extractor.extract(path)

    assert result["kind"] == "video" and result["streams"][0]["codec"] == "mpeg4"
    assert result["editor_flags"]["data.is_use_ai_video_generation"] == 1
    assert result["ai_hints"] == ["flaga edytora is_use_ai_video_generation=1"]


def test_video_export_app_from_tags(extractor, tmp_path):
    path = write_mp4(tmp_path / "film.mp4", "Made with CapCut")

    result = extractor.extract_media(path)

    assert result["export_apps"] == ["CapCut"]
    assert result["ai_hints"] == []


def test_missing_file_raises(extractor):
    with pytest.raises(FileNotFoundError, match="brak.mp4"):
        extractor.extract("brak.mp4")


PAGE = """<!-- saved from url=(0058)https://civicdaybook.live/?ad_id=120250477528820329&fbclid=X -->
<html lang="pl"><head>
<title> «Wiem, kto strzelał» — Gazeta.pl </title>
<meta name="description" content="Zamach na znanego przedsiębiorcę">
<meta name="generator" content="WordPress 6.4.2">
<meta property="og:url" content="https://wiadomosci.gazeta.pl/artykul">
<meta property="og:title" content="Wiem, kto strzelał">
<meta name="google-site-verification" content="abc123">
<link rel="canonical" href="https://wiadomosci.gazeta.pl/artykul">
<script async src="https://www.googletagmanager.com/gtag/js?id=G-ABCDEF1234"></script>
<script>fbq('init', '1376242401352160'); ttq.load('C4ABCDEFGHIJKLMNOPQR');</script>
<script src="/js/local.js"></script>
</head><body><img src="https://cdn.example-cdn.com/a.jpg"><noscript>
<img src="https://www.facebook.com/tr?id=1549107073347817&ev=PageView"></noscript></body></html>"""


def test_html_head_metadata(extractor):
    result = extractor.extract_html(PAGE)

    assert result["saved_from"].startswith("https://civicdaybook.live/")
    assert result["page_domain"] == "civicdaybook.live"
    assert result["title"] == "«Wiem, kto strzelał» — Gazeta.pl"
    assert result["lang"] == "pl" and result["generator"] == "WordPress 6.4.2"
    assert result["claimed_domains"] == ["gazeta.pl"]
    assert result["open_graph"]["og:title"] == "Wiem, kto strzelał"
    assert result["verification"] == {"google-site-verification": "abc123"}
    assert result["ad_ids"] == ["120250477528820329"]
    assert "facebook.com" in result["third_party_domains"]
    assert "civicdaybook.live" not in result["third_party_domains"]


def test_html_trackers(extractor):
    trackers = {(t["kind"], t["id"]) for t in extractor.extract_html(PAGE)["trackers"]}

    assert trackers == {
        ("google_analytics", "G-ABCDEF1234"),
        ("facebook_pixel", "1376242401352160"),
        ("facebook_pixel", "1549107073347817"),
        ("tiktok_pixel", "C4ABCDEFGHIJKLMNOPQR"),
    }


@pytest.mark.parametrize("text, expected", [
    ("gtag('config', 'UA-12345678-1')", ("google_analytics", "UA-12345678-1")),
    ("gtag('config', 'AW-987654321')", ("google_ads", "AW-987654321")),
    ("(function(w,d,s,l,i){})(window,document,'script','dataLayer','GTM-K9X2ABC');", ("google_tag_manager", "GTM-K9X2ABC")),
    ("ym(98765432, 'init', {clickmap:true})", ("yandex_metrica", "98765432")),
    ("h._hjSettings={hjid:3456789,hjsv:6}", ("hotjar", "3456789")),
    ("https://www.clarity.ms/tag/abcd1234ef", ("microsoft_clarity", "abcd1234ef")),
])
def test_find_trackers(text, expected):
    assert [(t["kind"], t["id"]) for t in find_trackers(text)] == [expected]


def test_url_signals_skip_secrets_and_build_signature():
    signals = url_signals([
        "https://ineptsolidoval.top/?pxl=1071875908635809&acc=hk77_tv6&ad_name=tvtext&ad_id=120250558396250564"
        "&access_token=EAAU123&clickid=db6a&lpkey=1791&land_id=25714&_camp_id=113&utm_source=ig",
    ])

    assert signals["trackers"] == [{"kind": "facebook_pixel", "id": "1071875908635809"}]
    assert signals["ad_ids"] == ["120250558396250564"]
    assert signals["ad_labels"] == ["acc=hk77_tv6", "ad_name=tvtext"]
    assert "access_token" in signals["param_names"]
    assert not any("EAAU123" in label for label in signals["ad_labels"])
    assert signals["param_signature"] == "_camp_id,acc,access_token,ad_id,ad_name,clickid,land_id,lpkey,pxl"


def test_page_without_metadata(extractor):
    result = extractor.extract_html("<html><body><p>tekst</p></body></html>")

    assert result["saved_from"] is None and result["title"] is None
    assert result["trackers"] == [] and result["ad_ids"] == [] and result["param_signature"] is None
