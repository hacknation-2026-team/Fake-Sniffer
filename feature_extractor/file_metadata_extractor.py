import av
import json
import os
import re
import struct
import tldextract
from html.parser import HTMLParser
from PIL import ExifTags, Image
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict
from urllib.parse import parse_qsl, urljoin, urlparse

_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic"}
HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
MAX_TEXT = 500
MAX_MOOV_BYTES = 5_000_000

AI_TOOL_MARKERS = (
    "stable diffusion", "stablediffusion", "midjourney", "dall-e", "dall·e", "dalle", "firefly",
    "imagen", "novelai", "comfyui", "automatic1111", "invokeai", "leonardo.ai", "ideogram",
    "flux", "gpt-4o", "chatgpt", "openai", "sora", "runway", "pika", "kling", "hailuo", "heygen",
    "synthesia", "d-id", "elevenlabs",
)
AI_TEXT_KEYS = ("parameters", "prompt", "workflow", "dream", "sd-metadata", "invokeai_metadata", "negative_prompt")
EXPORT_APP_MARKERS = (
    ("capcut", "CapCut"), ("te_is_reencode", "CapCut"), ("inshot", "InShot"), ("kinemaster", "KineMaster"),
    ("vn video editor", "VN"), ("premiere", "Adobe Premiere"), ("davinci", "DaVinci Resolve"),
    ("final cut", "Final Cut"), ("imovie", "iMovie"), ("canva", "Canva"), ("tiktok", "TikTok"),
    ("photoshop", "Adobe Photoshop"), ("lightroom", "Adobe Lightroom"), ("gimp", "GIMP"),
    ("snapseed", "Snapseed"), ("picsart", "PicsArt"), ("whatsapp", "WhatsApp"),
)
SYNTHETIC_SOURCE_TYPES = ("trainedalgorithmicmedia", "compositesynthetic", "algorithmicmedia")

TRACKER_PATTERNS: Sequence[Tuple[str, re.Pattern]] = (
    ("google_tag", re.compile(r"googletagmanager\.com/gtag/js\?id=((?:G|UA|AW|DC)-[A-Z0-9-]{4,20})", re.I)),
    ("google_tag", re.compile(r"gtag\(\s*['\"]config['\"]\s*,\s*['\"]((?:G|UA|AW|DC)-[A-Z0-9-]{4,20})", re.I)),
    ("google_tag", re.compile(r"[?&]tid=((?:G|UA)-[A-Z0-9-]{4,20})", re.I)),
    ("google_tag", re.compile(r"\b(UA-\d{4,10}-\d{1,4})\b")),
    ("google_tag_manager", re.compile(r"\b(GTM-[A-Z0-9]{4,9})\b")),
    ("facebook_pixel", re.compile(r"fbq\(\s*['\"]init['\"]\s*,\s*['\"](\d{10,20})")),
    ("facebook_pixel", re.compile(r"facebook\.com/tr/?\?[^\"'\s<>]*?\bid=(\d{10,20})", re.I)),
    ("tiktok_pixel", re.compile(r"ttq\.load\(\s*['\"]([A-Z0-9]{15,25})['\"]")),
    ("tiktok_pixel", re.compile(r"analytics\.tiktok\.com/[^\"'\s<>]*?sdkid=([A-Z0-9]{15,25})", re.I)),
    ("yandex_metrica", re.compile(r"\bym\(\s*(\d{6,10})\s*,\s*['\"]init")),
    ("yandex_metrica", re.compile(r"mc\.yandex\.(?:ru|com)/watch/(\d{6,10})", re.I)),
    ("hotjar", re.compile(r"\bhjid\s*:\s*(\d{5,10})")),
    ("microsoft_clarity", re.compile(r"clarity\.ms/tag/([a-z0-9]{8,12})", re.I)),
    ("microsoft_clarity", re.compile(r"\(window,\s*document,\s*['\"]clarity['\"]\s*,\s*['\"]script['\"]\s*,\s*['\"]([a-z0-9]{8,12})['\"]")),
)
C2PA_GENERATORS = (
    (rb"ChatGPT", "ChatGPT"), (rb"OpenAI", "OpenAI"), (rb"DALL.{0,3}E", "DALL-E"), (rb"Gemini", "Google Gemini"),
    (rb"Imagen", "Google Imagen"), (rb"Google", "Google"), (rb"Firefly", "Adobe Firefly"),
    (rb"Midjourney", "Midjourney"), (rb"Microsoft Designer", "Microsoft Designer"), (rb"Bing Image", "Bing Image Creator"),
    (rb"Meta AI", "Meta AI"), (rb"Stability", "Stability AI"),
)
PIXEL_PARAMS = ("pxl", "fbpx", "fb_pixel", "pixel", "pixel_id", "fbpixel", "px", "pixelid")
AD_ID_PARAMS = ("ad_id", "adset_id", "campaign_id", "utm_campaign", "utm_id", "utm_content", "utm_term", "fb_ad_id")
AD_LABEL_PARAMS = ("ad_name", "adset_name", "campaign_name", "acc", "account")
SECRET_PARAMS = ("access_token", "token", "fbclid", "gclid", "ttclid", "msclkid", "_token", "key", "lpkey")
COMMON_PARAMS = frozenset({"utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term", "utm_id", "fbclid", "gclid"})
_SAVED_FROM_RE = re.compile(r"<!--\s*saved from url=\(\d+\)(\S+?)\s*-->", re.I)
_SINGLEFILE_RE = re.compile(r"Page saved with SingleFile.*?url:\s*(\S+)", re.I | re.S)
_GOOGLE_KINDS = {"G": "google_analytics", "UA": "google_analytics", "AW": "google_ads", "DC": "google_ads"}


class TrackerDict(TypedDict):
    kind: str
    id: str


def registered_domain(url_or_host: Optional[str]) -> Optional[str]:
    if not url_or_host:
        return None
    parts = _TLD_EXTRACT(url_or_host)
    if parts.domain and parts.suffix:
        return f"{parts.domain}.{parts.suffix}".lower()
    return parts.domain.lower() or None


def find_trackers(text: str) -> List[TrackerDict]:
    found: Dict[Tuple[str, str], TrackerDict] = {}
    for kind, pattern in TRACKER_PATTERNS:
        for match in pattern.finditer(text or ""):
            value = match.group(1)
            if kind == "google_tag":
                value = value.upper()
                kind_name = _GOOGLE_KINDS.get(value.split("-", 1)[0], "google_analytics")
            else:
                kind_name = kind
            found.setdefault((kind_name, value), {"kind": kind_name, "id": value})
    return sorted(found.values(), key=lambda t: (t["kind"], t["id"]))


def url_signals(urls: Sequence[Optional[str]]) -> Dict[str, Any]:
    trackers: Dict[Tuple[str, str], TrackerDict] = {}
    ad_ids, ad_labels, param_names = set(), set(), set()
    for url in urls:
        if not url:
            continue
        for key, value in parse_qsl(urlparse(url).query, keep_blank_values=True):
            name = key.lower()
            param_names.add(name)
            if name in SECRET_PARAMS:
                continue
            if name in PIXEL_PARAMS and value.isdigit() and 10 <= len(value) <= 20:
                trackers[("facebook_pixel", value)] = {"kind": "facebook_pixel", "id": value}
            elif name in AD_ID_PARAMS and value.isdigit() and len(value) >= 10:
                ad_ids.add(value)
            elif name in AD_LABEL_PARAMS and value:
                ad_labels.add(f"{name}={_short(value)}")
    custom = sorted(param_names - COMMON_PARAMS)
    return {
        "trackers": sorted(trackers.values(), key=lambda t: (t["kind"], t["id"])),
        "ad_ids": sorted(ad_ids),
        "ad_labels": sorted(ad_labels),
        "param_names": sorted(param_names),
        "param_signature": ",".join(custom) if len(custom) >= 3 else None,
    }


def _short(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        value = value.replace("\x00", "").strip()
        return value[:MAX_TEXT] + "…" if len(value) > MAX_TEXT else value
    return value


def _ai_tool(text: Optional[str]) -> Optional[str]:
    lowered = (text or "").lower()
    return next((marker for marker in AI_TOOL_MARKERS if marker in lowered), None)


def _export_apps(texts: Sequence[str]) -> List[str]:
    joined = " ".join(texts).lower()
    return sorted({name for marker, name in EXPORT_APP_MARKERS if marker in joined})


def _gps_coordinate(values: Any, ref: Any) -> Optional[float]:
    try:
        degrees, minutes, seconds = (float(v) for v in values)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    result = degrees + minutes / 60 + seconds / 3600
    return round(-result if str(ref).upper() in ("S", "W") else result, 5)


def _json_blobs(data: bytes) -> List[Any]:
    text = data.decode("latin-1")
    decoder = json.JSONDecoder()
    blobs, position = [], 0
    while True:
        start = text.find('{"', position)
        if start < 0:
            return blobs
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            position = start + 2
            continue
        blobs.append(value)
        position = end


def _mp4_moov(path: str) -> bytes:
    with open(path, "rb") as file:
        file_size = os.fstat(file.fileno()).st_size
        position = 0
        while position + 8 <= file_size:
            file.seek(position)
            header = file.read(16)
            size, kind = struct.unpack(">I4s", header[:8])
            if size == 1:
                size = struct.unpack(">Q", header[8:16])[0]
            elif size == 0:
                size = file_size - position
            if size < 8:
                return b""
            if kind == b"moov":
                if size > MAX_MOOV_BYTES:
                    return b""
                file.seek(position)
                return file.read(size)
            position += size
    return b""


def _flags(blob: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(blob, dict):
        for key, value in blob.items():
            flat.update(_flags(value, f"{prefix}{key}."))
    elif not isinstance(blob, list):
        flat[prefix.rstrip(".")] = blob
    return flat


class _HeadParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: Dict[str, str] = {}
        self.title: Optional[str] = None
        self.lang: Optional[str] = None
        self.canonical: Optional[str] = None
        self.base: Optional[str] = None
        self.resources: List[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Sequence[tuple]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "html" and a.get("lang"):
            self.lang = a["lang"].strip()
        elif tag == "title" and self.title is None:
            self._in_title = True
            self.title = ""
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").strip().lower()
            if key and a.get("content") and key not in self.meta:
                self.meta[key] = a["content"].strip()
        elif tag == "link" and "canonical" in a.get("rel", "").lower() and a.get("href"):
            self.canonical = self.canonical or a["href"].strip()
        elif tag == "base" and a.get("href"):
            self.base = self.base or a["href"].strip()
        if tag in ("script", "iframe", "img") and a.get("src"):
            self.resources.append(a["src"].strip())
        elif tag == "link" and a.get("href"):
            self.resources.append(a["href"].strip())

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title = (self.title or "") + data


class FileMetadataExtractor:
    def extract(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Nie znaleziono pliku pod ścieżką: {path}")
        extension = os.path.splitext(path)[1].lower()
        if extension in IMAGE_EXTENSIONS:
            return self.extract_image(path)
        if extension in HTML_EXTENSIONS:
            with open(path, "rb") as file:
                return self.extract_html(file.read().decode("utf-8", "replace"))
        return self.extract_media(path)

    def extract_image(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Nie znaleziono pliku pod ścieżką: {path}")
        with open(path, "rb") as file:
            raw = file.read()

        with Image.open(path) as image:
            exif = image.getexif()
            base_tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
            exif_tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.get_ifd(ExifTags.IFD.Exif).items()}
            gps_tags = exif.get_ifd(ExifTags.IFD.GPSInfo)
            text_chunks = {
                str(k): _short(v) for k, v in image.info.items() if isinstance(v, (str, bytes)) and k not in ("icc_profile", "exif", "xmp")
            }
            result: Dict[str, Any] = {
                "kind": "image",
                "file": os.path.basename(path),
                "size_bytes": len(raw),
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "frames": getattr(image, "n_frames", 1),
            }

        gps = None
        if gps_tags:
            latitude = _gps_coordinate(gps_tags.get(2), gps_tags.get(1))
            longitude = _gps_coordinate(gps_tags.get(4), gps_tags.get(3))
            if latitude is not None and longitude is not None:
                gps = {"latitude": latitude, "longitude": longitude}

        xmp_match = re.search(rb"<x:xmpmeta.*?</x:xmpmeta>", raw, re.S)
        xmp = xmp_match.group(0).decode("utf-8", "replace") if xmp_match else ""
        creator_tool = re.search(r"CreatorTool(?:=\"|>)([^\"<]+)", xmp)
        source_type = re.search(r"DigitalSourceType(?:=\"|>)([^\"<]+)", xmp)

        result.update(
            {
                "camera": " ".join(str(base_tags[k]).strip() for k in ("Make", "Model") if base_tags.get(k)) or None,
                "software": _short(base_tags.get("Software")),
                "created": _short(exif_tags.get("DateTimeOriginal") or base_tags.get("DateTime")),
                "artist": _short(base_tags.get("Artist")),
                "copyright": _short(base_tags.get("Copyright")),
                "description": _short(base_tags.get("ImageDescription")),
                "gps": gps,
                "creator_tool": _short(creator_tool.group(1)) if creator_tool else None,
                "digital_source_type": source_type.group(1).rsplit("/", 1)[-1] if source_type else None,
                "content_credentials": self._content_credentials(raw),
                "text_chunks": text_chunks,
            }
        )
        result["tools"] = sorted({t for t in (result["software"], result["creator_tool"]) if t})
        result["export_apps"] = _export_apps(result["tools"] + list(text_chunks.values()))
        result["ai_hints"] = self._image_ai_hints(result)
        return result

    @staticmethod
    def _content_credentials(raw: bytes) -> Optional[Dict[str, Any]]:
        if b"c2pa" not in raw:
            return None
        generators: List[str] = []
        for pattern, name in C2PA_GENERATORS:
            if re.search(pattern, raw) and name not in generators:
                generators.append(name)
        if "Google" in generators and any(g.startswith("Google ") for g in generators):
            generators.remove("Google")
        source_types = sorted(
            {m.decode().lower() for m in re.findall(rb"(trainedAlgorithmicMedia|compositeSynthetic|algorithmicMedia)", raw)}
        )
        actions = sorted({m.decode() for m in re.findall(rb"c2pa\.(created|edited|converted|opened|placed)", raw)})
        return {
            "generators": generators,
            "digital_source_types": source_types,
            "actions": actions,
            "synthetic": any(t in SYNTHETIC_SOURCE_TYPES for t in source_types),
        }

    @staticmethod
    def _image_ai_hints(result: Dict[str, Any]) -> List[str]:
        hints: List[str] = []
        credentials = result.get("content_credentials") or {}
        if credentials.get("synthetic"):
            who = ", ".join(credentials["generators"]) or "nieznany generator"
            hints.append(f"manifest C2PA: obraz wygenerowany przez AI ({who})")
        for key in result["text_chunks"]:
            if key.lower() in AI_TEXT_KEYS:
                hints.append(f"parametry generowania obrazu w polu '{key}'")
        for field in ("software", "creator_tool", "description"):
            marker = _ai_tool(result.get(field))
            if marker:
                hints.append(f"narzędzie AI w polu {field}: {marker}")
        for value in result["text_chunks"].values():
            marker = _ai_tool(value)
            if marker:
                hints.append(f"narzędzie AI w metadanych tekstowych: {marker}")
                break
        if (result.get("digital_source_type") or "").lower() in SYNTHETIC_SOURCE_TYPES:
            hints.append(f"oznaczenie IPTC: {result['digital_source_type']}")
        return hints

    def extract_media(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Nie znaleziono pliku pod ścieżką: {path}")

        with av.open(str(path)) as container:
            tags = {str(k): _short(v) for k, v in container.metadata.items()}
            streams = []
            for stream in container.streams:
                ctx = stream.codec_context
                info: Dict[str, Any] = {"type": stream.type, "codec": ctx.name}
                if stream.type == "video":
                    info.update(width=ctx.width, height=ctx.height, fps=round(float(stream.average_rate or 0), 2))
                elif stream.type == "audio":
                    info.update(sample_rate=ctx.sample_rate, channels=ctx.channels)
                info["metadata"] = {str(k): _short(v) for k, v in stream.metadata.items()}
                streams.append(info)
            result: Dict[str, Any] = {
                "kind": "video" if any(s["type"] == "video" for s in streams) else "audio",
                "file": os.path.basename(path),
                "size_bytes": os.path.getsize(path),
                "container": container.format.name,
                "duration_s": round(container.duration / av.time_base, 2) if container.duration else None,
                "bit_rate": container.bit_rate,
                "tags": tags,
                "streams": streams,
            }

        editor: Dict[str, Any] = {}
        if os.path.splitext(path)[1].lower() in (".mp4", ".mov", ".m4v", ".m4a", ".3gp"):
            for blob in _json_blobs(_mp4_moov(path)):
                editor.update(_flags(blob))

        location = next((v for k, v in tags.items() if "location" in k.lower()), None)
        tools = sorted(
            {
                str(v)
                for k, v in tags.items()
                if any(word in k.lower() for word in ("encoder", "software", "creation_tool", "make", "model", "comment"))
                and v
            }
        )
        texts = [f"{k} {v}" for k, v in tags.items()] + [f"{k} {v}" for k, v in editor.items()]
        result.update(
            {
                "created": tags.get("creation_time") or next((s["metadata"].get("creation_time") for s in streams if s["metadata"].get("creation_time")), None),
                "location": location,
                "tools": tools,
                "export_apps": _export_apps(texts + tools),
                "editor_flags": {k: v for k, v in editor.items() if isinstance(v, (str, int, float, bool))},
            }
        )
        result["ai_hints"] = self._media_ai_hints(result)
        return result

    @staticmethod
    def _media_ai_hints(result: Dict[str, Any]) -> List[str]:
        hints = [
            f"flaga edytora {key.rsplit('.', 1)[-1]}={value}"
            for key, value in result["editor_flags"].items()
            if "ai" in key.lower().split(".")[-1].split("_") and str(value) not in ("0", "", "False", "false", "None")
        ]
        for value in list(result["tags"].values()) + result["tools"]:
            marker = _ai_tool(str(value))
            if marker:
                hints.append(f"narzędzie AI w tagach pliku: {marker}")
                break
        return hints

    def extract_html(
        self, html: str, url: Optional[str] = None, extra_urls: Sequence[Optional[str]] = ()
    ) -> Dict[str, Any]:
        parser = _HeadParser()
        parser.feed(html or "")
        parser.close()
        meta = parser.meta

        saved = _SAVED_FROM_RE.search(html or "") or _SINGLEFILE_RE.search((html or "")[:5000])
        saved_from = saved.group(1) if saved and saved.group(1).lower().startswith(("http://", "https://")) else None
        page_url = url or saved_from
        page_domain = registered_domain(urlparse(page_url).hostname) if page_url else None

        third_party = set()
        for resource in parser.resources:
            absolute = urljoin(page_url or parser.base or "", resource)
            host = urlparse(absolute).hostname
            domain = registered_domain(host) if host else None
            if domain and domain != page_domain:
                third_party.add(domain)

        signals = url_signals([url, saved_from, *extra_urls])
        trackers = {(t["kind"], t["id"]): t for t in find_trackers(html or "") + signals["trackers"]}
        canonical = parser.canonical
        claimed = {
            registered_domain(urlparse(value).hostname)
            for value in (canonical, meta.get("og:url"))
            if value and urlparse(value).hostname
        }
        return {
            "kind": "html",
            "url": url,
            "saved_from": saved_from,
            "page_domain": page_domain,
            "title": _short(" ".join((parser.title or "").split())) or None,
            "description": _short(meta.get("description") or meta.get("og:description")),
            "keywords": _short(meta.get("keywords")),
            "lang": parser.lang or meta.get("og:locale"),
            "generator": _short(meta.get("generator")),
            "author": _short(meta.get("author")),
            "canonical": canonical,
            "claimed_domains": sorted(d for d in claimed if d and d != page_domain),
            "open_graph": {k: _short(v) for k, v in meta.items() if k.startswith("og:")},
            "twitter": {k: _short(v) for k, v in meta.items() if k.startswith("twitter:")},
            "verification": {k: v for k, v in meta.items() if k.endswith("-site-verification") or k in ("facebook-domain-verification", "p:domain_verify")},
            "trackers": sorted(trackers.values(), key=lambda t: (t["kind"], t["id"])),
            "ad_ids": signals["ad_ids"],
            "ad_labels": signals["ad_labels"],
            "url_param_names": signals["param_names"],
            "param_signature": signals["param_signature"],
            "third_party_domains": sorted(third_party),
        }
