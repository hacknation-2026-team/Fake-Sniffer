import dns.exception
import dns.resolver
import os
import pytest
from feature_extractor.page_media import check_public_url
import requests
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from playwright.sync_api import Error as PlaywrightError

from feature_extractor.web_infrastructure_extractor import (
    InfrastructureReport,
    WebInfrastructureExtractor,
    install_chromium,
)

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def extractor() -> WebInfrastructureExtractor:
    return WebInfrastructureExtractor(timeout=1.0, settle_ms=0)


# URL i domena

@pytest.mark.parametrize("url", ["https://example.pl/x", "http://a.b.example.com:8080/"])
def test_validate_url_returns_host(url):
    assert WebInfrastructureExtractor.validate_url(url) in {"example.pl", "a.b.example.com"}


@pytest.mark.parametrize("url", ["ftp://example.com", "javascript:alert(1)", "example.com", "http://", ""])
def test_validate_url_rejects_non_http(url):
    with pytest.raises(ValueError, match="URL"):
        WebInfrastructureExtractor.validate_url(url)


@pytest.mark.parametrize(
    "host, expected",
    [
        ("login.example.com.pl", "example.com.pl"),  # sufiks wielopoziomowy
        ("www.onet-news.pl", "onet-news.pl"),
        ("a.b.c.example.co.uk", "example.co.uk"),
        ("192.168.1.10", None),
        ("localhost", None),
    ],
)
def test_registered_domain(host, expected):
    assert WebInfrastructureExtractor.registered_domain(host) == expected


# Odcisk DOM

TEMPLATE_A = """
<html><body><div class="hero big"><h1>Szok! Tusk poleca</h1><p>Zarób 5000 zł</p></div>
<form class="signup"><input name="email"><button class="cta">Dołącz</button></form></body></html>
"""
TEMPLATE_A_OTHER_TEXT = TEMPLATE_A.replace("Szok! Tusk poleca", "Uwaga! Lewandowski zdradza").replace("5000", "9000")
TEMPLATE_A_OTHER_CLASSES = TEMPLATE_A.replace("hero big", "banner").replace("cta", "btn")
TEMPLATE_B = "<html><body><table><tr><td>Zupełnie inna strona</td></tr></table><ul><li>a<li>b</ul></body></html>"


def test_same_template_different_text_gives_identical_fingerprint():
    a = WebInfrastructureExtractor.dom_fingerprint(TEMPLATE_A)
    b = WebInfrastructureExtractor.dom_fingerprint(TEMPLATE_A_OTHER_TEXT)
    assert a == b
    assert WebInfrastructureExtractor.structure_similarity(a, b) == 1.0


def test_changed_classes_keep_tag_hash_but_change_class_hash():
    a = WebInfrastructureExtractor.dom_fingerprint(TEMPLATE_A)
    b = WebInfrastructureExtractor.dom_fingerprint(TEMPLATE_A_OTHER_CLASSES)
    assert a["tag_hash"] == b["tag_hash"]
    assert a["tag_class_hash"] != b["tag_class_hash"]
    # Podobieństwo częściowe: część ścieżek (np. html/body/form) pozostaje wspólna
    assert 0.0 < WebInfrastructureExtractor.structure_similarity(a, b) < 1.0


def test_different_structure_is_dissimilar():
    a = WebInfrastructureExtractor.dom_fingerprint(TEMPLATE_A)
    b = WebInfrastructureExtractor.dom_fingerprint(TEMPLATE_B)
    assert a["tag_hash"] != b["tag_hash"]
    assert WebInfrastructureExtractor.structure_similarity(a, b) < 0.4


def test_injected_tracker_keeps_high_similarity():
    with_tracker = TEMPLATE_A.replace("<body>", "<body><script src='x.js'></script><img src='p.gif'>")
    a = WebInfrastructureExtractor.dom_fingerprint(TEMPLATE_A)
    b = WebInfrastructureExtractor.dom_fingerprint(with_tracker)
    assert a["tag_hash"] != b["tag_hash"]  # hash dokładny jest kruchy...
    assert WebInfrastructureExtractor.structure_similarity(a, b) > 0.7  # ...ale Jaccard nie


def test_malformed_and_empty_html_do_not_crash():
    fp = WebInfrastructureExtractor.dom_fingerprint("<div><p>niedomknięte<span></div></b>")
    assert fp["tag_count"] == 3

    empty = WebInfrastructureExtractor.dom_fingerprint("")
    assert empty["tag_count"] == 0 and empty["path_hashes"] == []
    assert WebInfrastructureExtractor.structure_similarity(empty, empty) == 0.0


# RDAP

def rdap_response(status=200, payload=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload or {}
    resp.raise_for_status.side_effect = (
        requests.HTTPError(f"{status}") if status >= 400 and status != 404 else None
    )
    return resp


RDAP_PAYLOAD = {
    "ldhName": "SZYBKI-ZAROBEK.PL",
    "status": ["active"],
    "events": [
        {"eventAction": "last changed", "eventDate": "2026-11-20T00:00:00Z"},
        {"eventAction": "registration", "eventDate": "2026-11-18T10:00:00Z"},
    ],
    "nameservers": [{"ldhName": "NS2.Example.net"}, {"ldhName": "ns1.example.net"}],
    "entities": [
        {"roles": ["registrant"], "vcardArray": ["vcard", [["fn", {}, "text", "REDACTED"]]]},
        {"roles": ["registrar"], "vcardArray": ["vcard", [["version", {}, "text", "4.0"], ["fn", {}, "text", "Tani Rejestrator sp. z o.o."]]]},
    ],
}


def test_rdap_parses_fields_and_age(extractor):
    with patch("feature_extractor.web_infrastructure_extractor.requests.get", return_value=rdap_response(200, RDAP_PAYLOAD)):
        info = extractor.query_rdap("szybki-zarobek.pl", now=NOW)

    assert info == {
        "domain": "szybki-zarobek.pl",
        "registered": True,
        "registrar": "Tani Rejestrator sp. z o.o.",
        "created": "2026-11-18T10:00:00+00:00",
        "age_days": 3,
        "nameservers": ["ns1.example.net", "ns2.example.net"],
        "status": ["active"],
    }


@pytest.mark.parametrize(
    "domain, expected_url",
    [
        ("szybki-zarobek.pl", "https://rdap.dns.pl/domain/szybki-zarobek.pl"),
        ("example.com", "https://rdap.org/domain/example.com"),
    ],
)
def test_rdap_endpoint_selection(extractor, domain, expected_url):
    with patch("feature_extractor.web_infrastructure_extractor.requests.get", return_value=rdap_response(200, RDAP_PAYLOAD)) as get:
        extractor.query_rdap(domain)
    assert get.call_args[0][0] == expected_url
    assert get.call_args[1]["timeout"] == 1.0


def test_rdap_404_means_unregistered(extractor):
    with patch("feature_extractor.web_infrastructure_extractor.requests.get", return_value=rdap_response(404)):
        info = extractor.query_rdap("nie-istnieje-xyz.pl")
    assert info["registered"] is False and info["created"] is None and info["nameservers"] == []


def test_rdap_server_error_propagates(extractor):
    with patch("feature_extractor.web_infrastructure_extractor.requests.get", return_value=rdap_response(500)):
        with pytest.raises(requests.HTTPError):
            extractor.query_rdap("example.com")


def test_rdap_missing_optional_fields(extractor):
    with patch("feature_extractor.web_infrastructure_extractor.requests.get", return_value=rdap_response(200, {"ldhName": "a.pl"})):
        info = extractor.query_rdap("a.pl", now=NOW)
    assert info["registrar"] is None and info["created"] is None and info["age_days"] is None


# DNS / ASN

class RR:
    def __init__(self, text):
        self._text = text

    def to_text(self):
        return self._text


def make_resolver(table):
    """table: {(nazwa, typ): [teksty] lub wyjątek}; brak klucza -> NXDOMAIN."""
    queried = []

    def fake_resolve(name, rtype, **kwargs):
        queried.append((name, rtype))
        value = table.get((name, rtype), dns.resolver.NXDOMAIN())
        if isinstance(value, Exception):
            raise value
        return [RR(v) for v in value]

    fake_resolve.queried = queried
    return fake_resolve


def test_resolve_dns_collects_records_and_asn(extractor):
    table = {
        ("login.szybki.pl", "A"): ["203.0.113.7"],
        ("szybki.pl", "NS"): ["NS2.cheap.net.", "ns1.cheap.net."],
        ("szybki.pl", "MX"): ["10 Mail.szybki.pl."],
        ("7.113.0.203.origin.asn.cymru.com", "TXT"): ['"64500 | 203.0.113.0/24 | PL | ripencc | 2020-01-01"'],
        ("AS64500.asn.cymru.com", "TXT"): ['"64500 | PL | ripencc | 2010-01-01 | EXAMPLE-AS, PL"'],
    }
    with patch("feature_extractor.web_infrastructure_extractor.dns.resolver.resolve", side_effect=make_resolver(table)):
        info = extractor.resolve_dns("login.szybki.pl", "szybki.pl")

    assert info["a"] == ["203.0.113.7"]
    assert info["aaaa"] == []  # NXDOMAIN -> pusta lista, bez wyjątku
    assert info["ns"] == ["ns1.cheap.net", "ns2.cheap.net"]
    assert info["mx"] == ["mail.szybki.pl"]
    assert info["asns"] == [
        {"ip": "203.0.113.7", "asn": 64500, "prefix": "203.0.113.0/24", "country": "PL", "as_name": "EXAMPLE-AS, PL"}
    ]


@pytest.mark.parametrize(
    "error", [dns.resolver.NXDOMAIN(), dns.resolver.NoAnswer(), dns.exception.Timeout(), dns.resolver.NoNameservers()]
)
def test_resolve_dns_survives_all_dns_errors(extractor, error):
    resolver = MagicMock(side_effect=error)
    with patch("feature_extractor.web_infrastructure_extractor.dns.resolver.resolve", resolver):
        info = extractor.resolve_dns("example.pl")
    assert info["a"] == info["aaaa"] == info["ns"] == info["mx"] == info["asns"] == []


def test_lookup_asn_query_names_for_ipv4_and_ipv6(extractor):
    resolver = make_resolver({})
    with patch("feature_extractor.web_infrastructure_extractor.dns.resolver.resolve", side_effect=resolver):
        assert extractor.lookup_asn("1.2.3.4") is None
        assert extractor.lookup_asn("2001:db8::1") is None

    assert resolver.queried[0] == ("4.3.2.1.origin.asn.cymru.com", "TXT")
    expected_v6 = ".".join("2001:0db8:0000:0000:0000:0000:0000:0001".replace(":", "")[::-1]) + ".origin6.asn.cymru.com"
    assert resolver.queried[1] == (expected_v6, "TXT")


def test_lookup_asn_without_as_name_still_returns_asn(extractor):
    table = {("1.0.0.10.origin.asn.cymru.com", "TXT"): ['"64501 | 10.0.0.0/8 | ZZ | x | 2000-01-01"']}
    with patch("feature_extractor.web_infrastructure_extractor.dns.resolver.resolve", side_effect=make_resolver(table)):
        info = extractor.lookup_asn("10.0.0.1")
    assert info["asn"] == 64501 and info["as_name"] is None


def test_lookup_asn_malformed_answer_returns_none(extractor):
    table = {("1.0.0.10.origin.asn.cymru.com", "TXT"): ['"nie-liczba | x"']}
    with patch("feature_extractor.web_infrastructure_extractor.dns.resolver.resolve", side_effect=make_resolver(table)):
        assert extractor.lookup_asn("10.0.0.1") is None


# TLS

def make_cert_der(not_before: datetime, sans=("szybki.pl", "www.szybki.pl")) -> bytes:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "szybki.pl")])
    issuer = x509.Name(
        [x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Let's Encrypt"), x509.NameAttribute(NameOID.COMMON_NAME, "R11")]
    )
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(0xABC123)
        .not_valid_before(not_before)
        .not_valid_after(not_before + timedelta(days=90))
    )
    if sans:
        builder = builder.add_extension(x509.SubjectAlternativeName([x509.DNSName(s) for s in sans]), critical=False)
    return builder.sign(key, hashes.SHA256()).public_bytes(serialization.Encoding.DER)


def test_parse_certificate_fields():
    der = make_cert_der(NOW - timedelta(days=2))
    info = WebInfrastructureExtractor.parse_certificate(der, now=NOW)

    assert info["issuer_org"] == "Let's Encrypt"
    assert "R11" in info["issuer"]
    assert info["subject_cn"] == "szybki.pl"
    assert info["sans"] == ["szybki.pl", "www.szybki.pl"]
    assert info["age_days"] == 2
    assert info["serial_hex"] == "abc123"
    assert len(info["sha256"]) == 64
    assert info["not_after"] > info["not_before"]


def test_parse_certificate_without_san():
    info = WebInfrastructureExtractor.parse_certificate(make_cert_der(NOW, sans=()), now=NOW)
    assert info["sans"] == []


def test_get_tls_info_uses_fetched_der(extractor):
    der = make_cert_der(NOW - timedelta(days=1))
    with patch.object(extractor, "fetch_certificate_der", return_value=der) as fetch:
        info = extractor.get_tls_info("szybki.pl", port=8443)
    fetch.assert_called_once_with("szybki.pl", 8443)
    assert info["subject_cn"] == "szybki.pl"


def test_fetch_certificate_der_does_not_verify_chain(extractor):
    """Cert wygasły/samopodpisany ma być czytelny - sprawdzamy, że kontekst SSL nie weryfikuje."""
    sock, tls = MagicMock(), MagicMock()
    tls.getpeercert.return_value = b"DER"
    context = MagicMock()
    context.wrap_socket.return_value.__enter__.return_value = tls

    with patch("feature_extractor.web_infrastructure_extractor.ssl.create_default_context", return_value=context), \
         patch("feature_extractor.web_infrastructure_extractor.socket.create_connection") as conn:
        conn.return_value.__enter__.return_value = sock
        assert extractor.fetch_certificate_der("szybki.pl") == b"DER"

    assert context.check_hostname is False
    assert context.verify_mode == __import__("ssl").CERT_NONE
    context.wrap_socket.assert_called_once_with(sock, server_hostname="szybki.pl")


# Przeglądarka (Playwright) - atrapa

def make_playwright_mock(page_steps):
    """Buduje atrapę sync_playwright; page_steps(handler, page) symuluje zdarzenia w trakcie goto."""
    page = MagicMock()
    page.main_frame = object()
    page.url = "https://landing.example/ok"
    page.content.return_value = "<html><body><p>x</p></body></html>"

    handlers = {}
    page.on.side_effect = lambda event, fn: handlers.__setitem__(event, fn)
    page.goto.side_effect = lambda *a, **k: page_steps(handlers["response"], page)

    browser = MagicMock()
    browser.new_context.return_value.new_page.return_value = page
    playwright = MagicMock()
    playwright.chromium.launch.return_value = browser

    factory = MagicMock()
    factory.return_value.__enter__.return_value = playwright
    return factory, browser, page


def nav_response(page, url, status, location=None, navigation=True, frame=None, resource_type="document", content_type=None):
    headers = {"location": location} if location else {}
    if content_type:
        headers["content-type"] = content_type
    return SimpleNamespace(
        url=url,
        status=status,
        headers=headers,
        request=SimpleNamespace(
            is_navigation_request=lambda: navigation,
            frame=frame or page.main_frame,
            resource_type=resource_type,
        ),
    )


def test_fetch_page_collects_image_and_media_resources(extractor):
    def steps(handler, page):
        handler(nav_response(page, "https://landing.example/", 200))
        handler(nav_response(page, "https://cdn.example/a.jpg", 200, navigation=False, resource_type="image", content_type="image/jpeg"))
        handler(nav_response(page, "https://cdn.example/v.mp4", 200, navigation=False, resource_type="media", content_type="video/mp4"))
        handler(nav_response(page, "https://cdn.example/app.js", 200, navigation=False, resource_type="script"))

    factory, browser, page = make_playwright_mock(steps)
    with patch("feature_extractor.web_infrastructure_extractor.sync_playwright", factory):
        result = extractor.fetch_page("https://landing.example/")

    assert result["resources"] == [
        {"url": "https://cdn.example/a.jpg", "type": "image", "content_type": "image/jpeg"},
        {"url": "https://cdn.example/v.mp4", "type": "media", "content_type": "video/mp4"},
    ]


def test_fetch_page_records_redirect_chain_only_for_main_frame_navigations(extractor):
    def steps(handler, page):
        handler(nav_response(page, "https://ads.tracker/click?id=1", 302, location="https://cloaker.example/r"))
        handler(nav_response(page, "https://cloaker.example/r", 301, location="https://landing.example/ok"))
        handler(nav_response(page, "https://landing.example/ok", 200))
        handler(nav_response(page, "https://cdn.example/logo.png", 200, navigation=False))       # podzasób
        handler(nav_response(page, "https://ads.example/frame", 200, frame=object()))             # iframe

    factory, browser, page = make_playwright_mock(steps)
    with patch("feature_extractor.web_infrastructure_extractor.sync_playwright", factory):
        result = extractor.fetch_page("https://ads.tracker/click?id=1")

    assert [(h["url"], h["status"], h["location"]) for h in result["hops"]] == [
        ("https://ads.tracker/click?id=1", 302, "https://cloaker.example/r"),
        ("https://cloaker.example/r", 301, "https://landing.example/ok"),
        ("https://landing.example/ok", 200, None),
    ]
    assert result["final_url"] == "https://landing.example/ok"
    assert "<p>x</p>" in result["html"]
    browser.new_context.assert_called_once()
    assert browser.new_context.call_args[1]["ignore_https_errors"] is True
    browser.close.assert_called_once()


def test_fetch_page_closes_browser_on_navigation_error(extractor):
    def steps(handler, page):
        raise TimeoutError("goto timeout")

    factory, browser, _ = make_playwright_mock(steps)
    with patch("feature_extractor.web_infrastructure_extractor.sync_playwright", factory):
        with pytest.raises(TimeoutError):
            extractor.fetch_page("https://slow.example")
    browser.close.assert_called_once()


def test_fetch_page_rejects_bad_scheme_before_launching_browser(extractor):
    factory = MagicMock()
    with patch("feature_extractor.web_infrastructure_extractor.sync_playwright", factory):
        with pytest.raises(ValueError):
            extractor.fetch_page("file:///etc/passwd")
    factory.assert_not_called()


# analyze() - całość z atrapami kroków

def patch_steps(extractor, **overrides):
    defaults = {
        "fetch_page": {"hops": [{"url": "https://t.example", "status": 302, "location": "https://szybki.pl/"}],
                       "final_url": "https://www.szybki.pl/", "html": TEMPLATE_A},
        "get_tls_info": {"issuer": "CN=R11"},
        "resolve_dns": {"host": "www.szybki.pl"},
        "query_rdap": {"domain": "szybki.pl"},
    }
    defaults.update(overrides)
    mocks = {}
    for name, value in defaults.items():
        mock = MagicMock(side_effect=value) if isinstance(value, Exception) else MagicMock(return_value=value)
        mocks[name] = mock
        setattr(extractor, name, mock)
    return mocks


def test_analyze_full_report_uses_landing_host(extractor):
    mocks = patch_steps(extractor)
    report = extractor.analyze("https://t.example/c?x=1")

    assert isinstance(report, InfrastructureReport)
    assert report.final_url == "https://www.szybki.pl/"
    assert report.host == "www.szybki.pl"            # host docelowy, nie tracker
    assert report.registered_domain == "szybki.pl"
    assert report.redirect_chain[0]["status"] == 302
    assert report.dom_fingerprint["tag_count"] > 0
    assert report.tls == {"issuer": "CN=R11"} and report.dns and report.rdap
    assert report.errors == []

    mocks["get_tls_info"].assert_called_once_with("www.szybki.pl")
    mocks["resolve_dns"].assert_called_once_with("www.szybki.pl", "szybki.pl")
    mocks["query_rdap"].assert_called_once_with("szybki.pl")
    assert report.to_dict()["registered_domain"] == "szybki.pl"


def test_analyze_one_failing_step_does_not_lose_the_rest(extractor):
    patch_steps(extractor, get_tls_info=ConnectionRefusedError("443 zamknięty"))
    report = extractor.analyze("https://szybki.pl")

    assert report.tls is None
    assert report.dns is not None and report.rdap is not None
    assert len(report.errors) == 1 and report.errors[0].startswith("tls: ConnectionRefusedError")


def test_analyze_unreachable_page_falls_back_to_input_host(extractor):
    mocks = patch_steps(
        extractor,
        fetch_page=TimeoutError("strona nie odpowiada"),
        fetch_page_http=TimeoutError("http też nie"),
    )
    report = extractor.analyze("https://martwa.example.pl/reklama")

    assert report.final_url is None and report.dom_fingerprint is None and report.redirect_chain == []
    assert report.host == "martwa.example.pl"
    assert any(e.startswith("browser: TimeoutError") for e in report.errors)
    assert any(e.startswith("page: TimeoutError") for e in report.errors)
    mocks["get_tls_info"].assert_called_once_with("martwa.example.pl")  # infrastruktura nadal zebrana


def test_analyze_skip_browser(extractor):
    mocks = patch_steps(extractor)
    report = extractor.analyze("https://szybki.pl", fetch_page=False)
    mocks["fetch_page"].assert_not_called()
    assert report.dom_fingerprint is None and report.rdap is not None


def test_analyze_invalid_url_returns_error_report_without_network(extractor):
    mocks = patch_steps(extractor)
    report = extractor.analyze("to nie jest url")

    assert report.host is None and report.errors and report.errors[0].startswith("url:")
    for mock in mocks.values():
        mock.assert_not_called()


def test_analyze_ip_host_skips_rdap(extractor):
    mocks = patch_steps(extractor, fetch_page=TimeoutError("x"))
    report = extractor.analyze("http://203.0.113.9/promo")

    assert report.registered_domain is None and report.rdap is None
    mocks["query_rdap"].assert_not_called()
    mocks["resolve_dns"].assert_called_once_with("203.0.113.9", None)


def test_visible_text_drops_scripts_styles_and_markup():
    html = (
        "<html><head><title>Wiadomości</title><style>.a{color:red}</style></head>"
        "<body><script>var x = 'ukryte';</script><h1>Tusk   poleca</h1>"
        "<noscript>włącz JS</noscript><p>Zarób 5000 zł</p><svg><text>ikona</text></svg></body></html>"
    )
    text = WebInfrastructureExtractor.visible_text(html)

    assert text.splitlines() == ["Wiadomości", "Tusk poleca", "Zarób 5000 zł"]


def test_visible_text_of_empty_html_is_empty():
    assert WebInfrastructureExtractor.visible_text("") == ""


def test_analyze_stores_visible_page_text(extractor):
    patch_steps(extractor)
    report = extractor.analyze("https://t.example/c")

    assert "Szok! Tusk poleca" in report.page_text
    assert "class=" not in report.page_text


def test_analyze_without_browser_has_no_page_text(extractor):
    patch_steps(extractor)
    assert extractor.analyze("https://szybki.pl", fetch_page=False).page_text is None


def test_analyze_keeps_page_html(extractor):
    patch_steps(extractor)

    assert extractor.analyze("https://t.example/c").page_html == TEMPLATE_A
    assert extractor.analyze("https://szybki.pl", fetch_page=False).page_html is None


def test_browser_cache_points_to_models_dir(tmp_path):
    extractor = WebInfrastructureExtractor(models_dir=str(tmp_path / "m"))
    extractor._use_browser_cache()

    assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == str(tmp_path / "m" / "playwright")
    assert (tmp_path / "m" / "playwright").is_dir()


def test_existing_browser_path_is_respected(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "/opt/przegladarki")
    WebInfrastructureExtractor(models_dir=str(tmp_path / "m"))._use_browser_cache()

    assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == "/opt/przegladarki"
    assert not (tmp_path / "m" / "playwright").exists()


def test_missing_chromium_is_installed_and_launch_retried(extractor):
    browser = MagicMock()
    playwright = MagicMock()
    playwright.chromium.launch.side_effect = [
        PlaywrightError("BrowserType.launch: Executable doesn't exist at /x/chrome"),
        browser,
    ]
    with patch("feature_extractor.web_infrastructure_extractor.install_chromium") as install:
        assert extractor._launch_browser(playwright) is browser

    install.assert_called_once_with()
    assert playwright.chromium.launch.call_count == 2


def test_other_launch_errors_do_not_trigger_install(extractor):
    playwright = MagicMock()
    playwright.chromium.launch.side_effect = PlaywrightError("Target closed")
    with patch("feature_extractor.web_infrastructure_extractor.install_chromium") as install:
        with pytest.raises(PlaywrightError, match="Target closed"):
            extractor._launch_browser(playwright)
    install.assert_not_called()


def test_installed_chromium_is_launched_once_without_install(extractor):
    playwright = MagicMock()
    with patch("feature_extractor.web_infrastructure_extractor.install_chromium") as install:
        extractor._launch_browser(playwright)
    install.assert_not_called()
    playwright.chromium.launch.assert_called_once_with(headless=True, args=["--disable-dev-shm-usage", "--disable-gpu"])


def test_install_chromium_runs_playwright_cli():
    with patch("feature_extractor.web_infrastructure_extractor.subprocess.run") as run:
        install_chromium()

    command = run.call_args[0][0]
    assert command[1:] == ["-m", "playwright", "install", "chromium"]
    assert run.call_args[1] == {"check": True}


def test_ensure_browser_launches_and_closes(extractor):
    factory, browser, _ = make_playwright_mock(lambda handler, page: None)
    with patch("feature_extractor.web_infrastructure_extractor.sync_playwright", factory):
        extractor.ensure_browser()

    browser.close.assert_called_once()
    assert "PLAYWRIGHT_BROWSERS_PATH" in os.environ


def fake_http(routes):
    class Raw:
        def __init__(self, body):
            self.body = body

        def read(self, size, decode_content=True):
            return self.body[:size]

    def get(url, **kwargs):
        assert kwargs["allow_redirects"] is False
        status, headers, body = routes[url]
        return SimpleNamespace(
            status_code=status, headers=headers, raw=Raw(body), encoding="utf-8", close=lambda: None
        )

    return get


HTML = b"<html><body><h1>Wygrana</h1><p>Odbierz nagrode</p></body></html>"


def test_http_fallback_follows_redirects_to_the_last_page(extractor):
    routes = {
        "https://t.example/c": (302, {"location": "https://mid.example/r"}, b""),
        "https://mid.example/r": (301, {"location": "/final"}, b""),
        "https://mid.example/final": (200, {"content-type": "text/html; charset=utf-8"}, HTML),
    }
    with patch("feature_extractor.web_infrastructure_extractor.requests.get", fake_http(routes)), patch(
        "feature_extractor.web_infrastructure_extractor.check_public_url"
    ):
        page = extractor.fetch_page_http("https://t.example/c")
    assert page["final_url"] == "https://mid.example/final"
    assert [(h["url"], h["status"]) for h in page["hops"]] == [
        ("https://t.example/c", 302),
        ("https://mid.example/r", 301),
        ("https://mid.example/final", 200),
    ]
    assert b"Wygrana" in page["html"].encode()


def test_http_fallback_follows_meta_refresh(extractor):
    refresh = b'<html><head><meta http-equiv="refresh" content="0; url=https://landing.example/ok"></head></html>'
    routes = {
        "https://a.example/": (200, {"content-type": "text/html"}, refresh),
        "https://landing.example/ok": (200, {"content-type": "text/html"}, HTML),
    }
    with patch("feature_extractor.web_infrastructure_extractor.requests.get", fake_http(routes)), patch(
        "feature_extractor.web_infrastructure_extractor.check_public_url"
    ):
        page = extractor.fetch_page_http("https://a.example/")
    assert page["final_url"] == "https://landing.example/ok"


def test_http_fallback_blocks_redirect_to_private_address(extractor):
    routes = {"https://a.example/": (302, {"location": "http://127.0.0.1/admin"}, b"")}
    resolver = lambda host, port: [(None, None, None, None, ("127.0.0.1" if host == "127.0.0.1" else "93.184.216.34", 0))]
    with patch("feature_extractor.web_infrastructure_extractor.requests.get", fake_http(routes)), patch(
        "feature_extractor.web_infrastructure_extractor.check_public_url",
        lambda url: check_public_url(url, resolver),
    ):
        with pytest.raises(ValueError, match="niepubliczny"):
            extractor.fetch_page_http("https://a.example/")


def test_http_fallback_stops_on_redirect_loop(extractor):
    routes = {"https://a.example/": (302, {"location": "https://a.example/"}, b"")}
    with patch("feature_extractor.web_infrastructure_extractor.requests.get", fake_http(routes)), patch(
        "feature_extractor.web_infrastructure_extractor.check_public_url"
    ):
        with pytest.raises(ValueError, match="przekierowań"):
            extractor.fetch_page_http("https://a.example/")


def test_non_html_final_response_gives_empty_html(extractor):
    routes = {"https://a.example/f.pdf": (200, {"content-type": "application/pdf"}, b"%PDF")}
    with patch("feature_extractor.web_infrastructure_extractor.requests.get", fake_http(routes)), patch(
        "feature_extractor.web_infrastructure_extractor.check_public_url"
    ):
        assert extractor.fetch_page_http("https://a.example/f.pdf")["html"] == ""


def test_analyze_uses_http_fallback_when_browser_fails(extractor):
    fallback = {
        "hops": [{"url": "https://t.example/c", "status": 302, "location": "https://szybki.pl/"},
                 {"url": "https://szybki.pl/", "status": 200, "location": None}],
        "final_url": "https://szybki.pl/",
        "html": "<html><body><p>Treść strony</p></body></html>",
        "resources": [],
    }
    patch_steps(extractor, fetch_page=RuntimeError("TargetClosedError: Browser closed\nCall log: ..."), fetch_page_http=fallback)
    extractor.fetch_page_http = MagicMock(return_value=fallback)
    report = extractor.analyze("https://t.example/c")
    assert report.final_url == "https://szybki.pl/"
    assert report.host == "szybki.pl"
    assert "Treść strony" in report.page_text and report.page_html
    assert [h["status"] for h in report.redirect_chain] == [302, 200]
    assert report.errors[0] == "browser: RuntimeError: TargetClosedError: Browser closed"
    assert any("zapasowego" in e for e in report.errors)


def test_fetch_page_keeps_partial_result_when_load_fails_after_redirects(extractor):
    from playwright.sync_api import Error as PlaywrightError

    def steps(handler, page):
        handler(nav_response(page, "https://t.example/c", 302, location="https://landing.example/"))
        handler(nav_response(page, "https://landing.example/", 200))
        raise PlaywrightError("Timeout 1000ms exceeded.\nCall log: waiting for load")

    factory, browser, page = make_playwright_mock(steps)
    page.url = "https://landing.example/"
    page.content.return_value = "<html><body>czesciowa</body></html>"
    with patch("feature_extractor.web_infrastructure_extractor.sync_playwright", factory):
        result = extractor.fetch_page("https://t.example/c")
    assert result["final_url"] == "https://landing.example/"
    assert "czesciowa" in result["html"]
    assert "częściowo" in result["warning"] and "Call log" not in result["warning"]
    browser.close.assert_called_once()
