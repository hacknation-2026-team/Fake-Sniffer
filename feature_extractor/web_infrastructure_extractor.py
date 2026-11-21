import dns.exception
import dns.resolver
import hashlib
import ipaddress
import requests
import socket
import ssl
import tldextract
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.x509.oid import NameOID
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from playwright.sync_api import sync_playwright
from typing import Any, Dict, List, Optional, Sequence, TypedDict
from urllib.parse import urlparse

_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())


RDAP_URL_TEMPLATES: Dict[str, str] = {"pl": "https://rdap.dns.pl/domain/{domain}"}
RDAP_DEFAULT_TEMPLATE = "https://rdap.org/domain/{domain}"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


class RedirectHop(TypedDict):
    url: str
    status: int
    location: Optional[str]


class DomFingerprint(TypedDict):
    tag_hash: str            # sha256 sekwencji (głębokość, tag)
    tag_class_hash: str      # to samo z klasami CSS
    path_hashes: List[str]   # skróty unikalnych ścieżek tag+klasy, do podobieństwa Jaccarda
    tag_count: int


class RdapInfo(TypedDict):
    domain: str
    registered: bool
    registrar: Optional[str]
    created: Optional[str]    # ISO 8601
    age_days: Optional[int]
    nameservers: List[str]
    status: List[str]


class DnsInfo(TypedDict):
    host: str
    a: List[str]
    aaaa: List[str]
    ns: List[str]
    mx: List[str]
    asns: List[Dict[str, Any]]  # ip, asn, prefix, country, as_name


class CertInfo(TypedDict):
    issuer: str
    issuer_org: Optional[str]
    subject_cn: Optional[str]
    sans: List[str]
    not_before: str
    not_after: str
    age_days: int
    serial_hex: str
    sha256: str


@dataclass
class InfrastructureReport:
    input_url: str
    host: Optional[str] = None
    registered_domain: Optional[str] = None
    final_url: Optional[str] = None
    redirect_chain: List[RedirectHop] = field(default_factory=list)
    dom_fingerprint: Optional[DomFingerprint] = None
    rdap: Optional[RdapInfo] = None
    dns: Optional[DnsInfo] = None
    tls: Optional[CertInfo] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _StructureParser(HTMLParser):
    """Zbiera tagi z głębokością. Znosi niedomknięte tagi i bałagan w HTML-u."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: List[str] = []
        self.tag_tokens: List[str] = []
        self.class_tokens: List[str] = []
        self.paths: List[str] = []

    def handle_starttag(self, tag: str, attrs: Sequence[tuple]) -> None:
        classes = ""
        for name, value in attrs:
            if name == "class" and value:
                classes = ".".join(sorted(value.split()))
        depth = len(self.stack)
        label = f"{tag}.{classes}" if classes else tag

        self.tag_tokens.append(f"{depth}:{tag}")
        self.class_tokens.append(f"{depth}:{label}")
        self.paths.append("/".join([*self.stack, label]))

        if tag not in _VOID_TAGS:
            
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: Sequence[tuple]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS and self.stack and self.stack[-1] == tag:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        if tag in self.stack:

            while self.stack:
                if self.stack.pop() == tag:
                    break


def _short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


class WebInfrastructureExtractor:
    def __init__(
        self,
        timeout: float = 10.0,
        user_agent: str = DEFAULT_USER_AGENT,
        settle_ms: int = 1500,
    ):
        """
        Ślady infrastrukturalne strony: przekierowania, odcisk DOM, RDAP, DNS/ASN i certyfikat TLS.
        Kroki są niezależne - jak któryś padnie, błąd ląduje w `errors`, a reszta raportu powstaje.

        timeout to limit pojedynczego zapytania (s), user_agent ma wyglądać jak zwykła przeglądarka
        (część cloakerów serwuje botom inną treść), a settle_ms to dodatkowe czekanie na
        przekierowania z JavaScriptu po załadowaniu strony.

        Strony oszustów odpalają się w prawdziwej przeglądarce, więc rób to w kontenerze lub VM.
        """
        self.timeout = timeout
        self.user_agent = user_agent
        self.settle_ms = settle_ms

    # --- URL i domena ---

    @staticmethod
    def validate_url(url: str) -> str:
        """Przepuszcza tylko http(s) z nazwą hosta i zwraca tego hosta."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"Nieprawidłowy adres URL (wymagany http/https): {url}")
        return parsed.hostname

    @staticmethod
    def registered_domain(host: str) -> Optional[str]:
        """Domena rejestrowalna wg Public Suffix List (login.x.com.pl -> x.com.pl), None dla IP."""
        try:
            ipaddress.ip_address(host)
            return None
        except ValueError:
            pass
        extracted = _TLD_EXTRACT(host)

        name = getattr(extracted, "top_domain_under_public_suffix", None)
        if name is None:
            name = extracted.registered_domain
        return name or None

    # --- przeglądarka ---

    def fetch_page(self, url: str) -> Dict[str, Any]:
        """
        Otwiera URL w headless Chromium i zwraca {"hops": [...], "final_url": ..., "html": ...}.
        W hops lądują wszystkie nawigacje głównej ramki: 30x, meta refresh i przekierowania z JS.
        """
        self.validate_url(url)
        hops: List[RedirectHop] = []

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
  
                context = browser.new_context(user_agent=self.user_agent, ignore_https_errors=True)
                page = context.new_page()

                def on_response(response: Any) -> None:
                    request = response.request
                    if request.is_navigation_request() and request.frame == page.main_frame:
                        hops.append(
                            {
                                "url": response.url,
                                "status": int(response.status),
                                "location": response.headers.get("location"),
                            }
                        )

                page.on("response", on_response)
                page.goto(url, wait_until="load", timeout=int(self.timeout * 1000))
                page.wait_for_timeout(self.settle_ms)
                return {"hops": hops, "final_url": page.url, "html": page.content()}
            finally:
                browser.close()

    # --- odcisk DOM ---

    @staticmethod
    def dom_fingerprint(html: str) -> DomFingerprint:
        """
        Odcisk szablonu strony, niezależny od treści. Oszuści stawiają ten sam szablon na wielu
        domenach, zmieniając tylko teksty. Hashe dokładne są kruche (wystarczy dopisany tracker),
        dlatego `path_hashes` pozwala policzyć też podobieństwo częściowe.
        """
        parser = _StructureParser()
        parser.feed(html)
        parser.close()

        return {
            "tag_hash": hashlib.sha256("\n".join(parser.tag_tokens).encode()).hexdigest(),
            "tag_class_hash": hashlib.sha256("\n".join(parser.class_tokens).encode()).hexdigest(),
            "path_hashes": sorted({_short_hash(p) for p in parser.paths}),
            "tag_count": len(parser.tag_tokens),
        }

    @staticmethod
    def structure_similarity(fp_a: DomFingerprint, fp_b: DomFingerprint) -> float:
        """Jaccard zbiorów ścieżek DOM, 1.0 dla identycznych szablonów."""
        a, b = set(fp_a["path_hashes"]), set(fp_b["path_hashes"])
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)

    # --- RDAP ---

    def query_rdap(self, domain: str, now: Optional[datetime] = None) -> RdapInfo:
        """
        Dane rejestracyjne domeny z RDAP: rejestrator, data utworzenia, serwery NS. Domena
        młodsza niż kilka dni to mocny sygnał. RDAP NASK (.pl) publikuje dane z opóźnieniem
        i ukrywa rejestranta, więc wiek bywa pusty dla świeżo zarejestrowanych domen.
        """
        tld = domain.rsplit(".", 1)[-1].lower()
        url = RDAP_URL_TEMPLATES.get(tld, RDAP_DEFAULT_TEMPLATE).format(domain=domain)

        response = requests.get(
            url,
            headers={"Accept": "application/rdap+json"},
            timeout=self.timeout,
            allow_redirects=True,
        )
        if response.status_code == 404:
            return {
                "domain": domain, "registered": False, "registrar": None, "created": None,
                "age_days": None, "nameservers": [], "status": [],
            }
        response.raise_for_status()
        data = response.json()

        created: Optional[datetime] = None
        for event in data.get("events", []):
            if event.get("eventAction") == "registration" and event.get("eventDate"):
                created = datetime.fromisoformat(event["eventDate"].replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                break

        current = now or datetime.now(timezone.utc)
        return {
            "domain": data.get("ldhName", domain).lower(),
            "registered": True,
            "registrar": self._rdap_registrar(data),
            "created": created.isoformat() if created else None,
            "age_days": (current - created).days if created else None,
            "nameservers": sorted(
                ns["ldhName"].lower() for ns in data.get("nameservers", []) if ns.get("ldhName")
            ),
            "status": list(data.get("status", [])),
        }

    @staticmethod
    def _rdap_registrar(data: Dict[str, Any]) -> Optional[str]:
        for entity in data.get("entities", []):
            if "registrar" not in entity.get("roles", []):
                continue
            vcard = entity.get("vcardArray")
            if vcard and len(vcard) > 1:
                for item in vcard[1]:
                    if item and item[0] == "fn" and len(item) > 3:
                        return str(item[3])
            if entity.get("handle"):
                return str(entity["handle"])
        return None

    # --- DNS i ASN ---

    def resolve_dns(self, host: str, domain: Optional[str] = None) -> DnsInfo:
        """
        A/AAAA dla hosta, NS/MX dla domeny rejestrowalnej i ASN każdego IP (Team Cymru przez DNS).
        Brak rekordu daje pustą listę, nie wyjątek.
        """
        domain = domain or host

        def query(name: str, rtype: str) -> List[str]:
            try:
                answers = dns.resolver.resolve(name, rtype, lifetime=self.timeout)
            except (
                dns.resolver.NXDOMAIN,
                dns.resolver.NoAnswer,
                dns.resolver.NoNameservers,
                dns.exception.Timeout,
            ):
                return []
            return [a.to_text().rstrip(".") for a in answers]

        ips_v4, ips_v6 = query(host, "A"), query(host, "AAAA")
        mx_raw = query(domain, "MX")  # format "10 mail.example.com"
        return {
            "host": host,
            "a": sorted(ips_v4),
            "aaaa": sorted(ips_v6),
            "ns": sorted(ns.lower() for ns in query(domain, "NS")),
            "mx": sorted(m.split()[-1].lower() for m in mx_raw),
            "asns": [info for ip in [*ips_v4, *ips_v6] if (info := self.lookup_asn(ip))],
        }

    def lookup_asn(self, ip: str) -> Optional[Dict[str, Any]]:
        """ASN, prefiks, kraj i nazwa AS dla IP przez DNS Team Cymru, None gdy brak danych."""
        address = ipaddress.ip_address(ip)
        if address.version == 4:
            query_name = ".".join(reversed(ip.split("."))) + ".origin.asn.cymru.com"
        else:
            nibbles = address.exploded.replace(":", "")[::-1]
            query_name = ".".join(nibbles) + ".origin6.asn.cymru.com"

        try:
            origin = dns.resolver.resolve(query_name, "TXT", lifetime=self.timeout)
            # np. "13335 | 1.1.1.0/24 | AU | apnic | 2011-08-11"
            fields = [f.strip() for f in origin[0].to_text().strip('"').split("|")]
            asn = fields[0].split()[0]  # przy kilku ASN-ach bierzemy pierwszy
            info: Dict[str, Any] = {
                "ip": ip, "asn": int(asn), "prefix": fields[1], "country": fields[2], "as_name": None,
            }
        except (
            dns.resolver.NXDOMAIN,
            dns.resolver.NoAnswer,
            dns.resolver.NoNameservers,
            dns.exception.Timeout,
            IndexError,
            ValueError,
        ):
            return None

        try:
            name_answer = dns.resolver.resolve(f"AS{info['asn']}.asn.cymru.com", "TXT", lifetime=self.timeout)
            # np. "13335 | US | arin | 2010-07-14 | CLOUDFLARENET, US"
            info["as_name"] = name_answer[0].to_text().strip('"').split("|")[-1].strip()
        except (
            dns.resolver.NXDOMAIN,
            dns.resolver.NoAnswer,
            dns.resolver.NoNameservers,
            dns.exception.Timeout,
        ):
            pass
        return info

    # --- TLS ---

    def fetch_certificate_der(self, host: str, port: int = 443) -> bytes:
        """
        Pobiera certyfikat serwera bez weryfikacji łańcucha. Interesuje nas to, co serwer
        pokazuje, także gdy cert jest wygasły albo samopodpisany (typowe u oszustów).
        """
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        with socket.create_connection((host, port), timeout=self.timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as tls:
                der = tls.getpeercert(binary_form=True)
        if not der:
            raise ssl.SSLError("Serwer nie przedstawił certyfikatu.")
        return der

    @staticmethod
    def parse_certificate(der: bytes, now: Optional[datetime] = None) -> CertInfo:
        cert = x509.load_der_x509_certificate(der)

        def first_attr(name: x509.Name, oid: Any) -> Optional[str]:
            attrs = name.get_attributes_for_oid(oid)
            return str(attrs[0].value) if attrs else None

        try:
            san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            sans = sorted(san_ext.value.get_values_for_type(x509.DNSName))
        except x509.ExtensionNotFound:
            sans = []

        current = now or datetime.now(timezone.utc)
        return {
            "issuer": cert.issuer.rfc4514_string(),
            "issuer_org": first_attr(cert.issuer, NameOID.ORGANIZATION_NAME),
            "subject_cn": first_attr(cert.subject, NameOID.COMMON_NAME),
            "sans": sans,
            "not_before": cert.not_valid_before_utc.isoformat(),
            "not_after": cert.not_valid_after_utc.isoformat(),
            "age_days": (current - cert.not_valid_before_utc).days,
            "serial_hex": format(cert.serial_number, "x"),
            "sha256": cert.fingerprint(hashes.SHA256()).hex(),
        }

    def get_tls_info(self, host: str, port: int = 443) -> CertInfo:
        return self.parse_certificate(self.fetch_certificate_der(host, port))

    # --- całość ---

    def analyze(self, url: str, fetch_page: bool = True) -> InfrastructureReport:
        """
        Pełna analiza URL-a, bez wyjątków z kroków sieciowych (błędy idą do `errors`).
        Domena, DNS i TLS dotyczą hosta docelowego po przekierowaniach, a gdy strona się
        nie załadowała - hosta z podanego URL-a. Pośrednie domeny trackerów zostają w
        `redirect_chain`. fetch_page=False pomija przeglądarkę (bez przekierowań i odcisku DOM).
        """
        report = InfrastructureReport(input_url=url)
        try:
            report.host = self.validate_url(url)
        except ValueError as exc:
            report.errors.append(f"url: {exc}")
            return report

        if fetch_page:
            try:
                page = self.fetch_page(url)
                report.redirect_chain = page["hops"]
                report.final_url = page["final_url"]
                report.dom_fingerprint = self.dom_fingerprint(page["html"])
                report.host = urlparse(page["final_url"]).hostname or report.host
            except Exception as exc: 
                report.errors.append(f"browser: {type(exc).__name__}: {exc}")

        host = report.host
        report.registered_domain = self.registered_domain(host)

        steps = [
            ("tls", lambda: self.get_tls_info(host), "tls"),
            ("dns", lambda: self.resolve_dns(host, report.registered_domain), "dns"),
        ]
        if report.registered_domain:
            steps.append(("rdap", lambda: self.query_rdap(report.registered_domain), "rdap"))

        for name, step, attribute in steps:
            try:
                setattr(report, attribute, step())
            except Exception as exc:
                report.errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return report
