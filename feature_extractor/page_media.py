import hashlib
import io
import ipaddress
import re
import socket
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from PIL import Image

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".m4v", ".m3u8")
SKIPPED_IMAGES = re.compile(r"\.(svg|ico)(\?|$)|sprite|pixel|spacer|1x1|favicon", re.I)
EMBED_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com", "vimeo.com", "dailymotion.com", "tiktok.com")
STYLE_URL = re.compile(r"url\(\s*['\"]?([^'\")]+)", re.I)
MAX_REDIRECTS = 5
DECLARED_MIN_SIDE = 100


def check_public_url(url: str, resolver: Callable[..., Any] = socket.getaddrinfo) -> None:
    """Przepuszcza tylko http(s) wskazujące wyłącznie na adresy publiczne (ochrona przed SSRF)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("dozwolone tylko adresy http(s)")
    try:
        addresses = {info[4][0] for info in resolver(parsed.hostname, None)}
    except OSError as exc:
        raise ValueError(f"nie udało się rozwiązać hosta: {exc}")
    if not addresses or not all(ipaddress.ip_address(a).is_global for a in addresses):
        raise ValueError("adres niepubliczny, pomijam")

EMOJI = re.compile(
    "(?:[\U0001F1E6-\U0001F1FF]{2}"
    "|[\U0001F300-\U0001FAFF☀-➿⭐⭕⬆⬇⌚⌛⏩-⏺]"
    "️?(?:‍[\U0001F300-\U0001FAFF☀-➿]️?)*)"
)


def find_emojis(text: str, top: int = 10) -> Dict[str, Any]:
    """Emotikony w tekście: ile ich jest, jakie i jak gęsto (na 1000 znaków)."""
    found = EMOJI.findall(text or "")
    counts = Counter(found)
    density = round(1000 * len(found) / len(text), 2) if text else 0.0
    return {
        "total": len(found),
        "unique": len(counts),
        "density": density,
        "top": [[emoji, count] for emoji, count in counts.most_common(top)],
    }


def _largest_from_srcset(srcset: str) -> Optional[str]:
    best, best_width = None, -1.0
    for candidate in srcset.split(","):
        parts = candidate.strip().split()
        if not parts:
            continue
        descriptor = parts[1] if len(parts) > 1 else "1x"
        try:
            width = float(descriptor[:-1])
        except ValueError:
            width = 0.0
        if width > best_width:
            best, best_width = parts[0], width
    return best


def _size(value: Optional[str]) -> Optional[int]:
    return int(value) if value and value.isdigit() else None


class _MediaParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.images: List[Dict[str, Any]] = []
        self.videos: List[Dict[str, Any]] = []
        self.inline_images = 0
        self._video_depth = 0

    def _url(self, value: Optional[str]) -> Optional[str]:
        if not value or not value.strip():
            return None
        value = value.strip()
        if value.startswith("data:"):
            self.inline_images += 1
            return None
        return urljoin(self.base_url, value)

    def _image(self, value: Optional[str], source: str, alt: str = "", attrs: Dict[str, str] = {}) -> None:
        url = self._url(value)
        if url:
            self.images.append(
                {
                    "url": url,
                    "alt": alt,
                    "source": source,
                    "width": _size(attrs.get("width")),
                    "height": _size(attrs.get("height")),
                }
            )

    def _video(self, value: Optional[str], poster: Optional[str] = None) -> None:
        url = self._url(value)
        if url:
            self.videos.append({"url": url, "kind": video_kind(url), "poster": self._url(poster)})

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag == "img":
            largest = _largest_from_srcset(a.get("srcset", "") or a.get("data-srcset", ""))
            if largest:
                self._image(largest, "srcset", a.get("alt", ""), a)
            else:
                for name in ("src", "data-src", "data-lazy-src", "data-original"):
                    self._image(a.get(name), "img", a.get("alt", ""), a)
        elif tag == "video":
            self._video_depth += 1
            self._video(a.get("src"), a.get("poster"))
            self._image(a.get("poster"), "poster")
        elif tag == "source":
            if self._video_depth:
                self._video(a.get("src"))
            else:
                self._image(_largest_from_srcset(a.get("srcset", "")) or a.get("src"), "srcset")
        elif tag == "iframe":
            src = self._url(a.get("src"))
            if src and any(host in (urlparse(src).hostname or "") for host in EMBED_HOSTS):
                self.videos.append({"url": src, "kind": "embed", "poster": None})
        elif tag == "meta" and a.get("property", a.get("name", "")) in ("og:image", "twitter:image"):
            self._image(a.get("content"), "meta")
        elif tag == "meta" and a.get("property") == "og:video":
            self._video(a.get("content"))
        for match in STYLE_URL.findall(a.get("style", "")):
            self._image(match, "style")

    def handle_endtag(self, tag: str) -> None:
        if tag == "video" and self._video_depth:
            self._video_depth -= 1


class _CanonicalParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.url: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if self.url is None and tag == "link" and a.get("rel", "").lower() == "canonical":
            self.url = a.get("href") or None
        elif self.url is None and tag == "meta" and a.get("property") == "og:url":
            self.url = a.get("content") or None


def find_page_url(html: str) -> Optional[str]:
    """Adres strony zapisanego HTML-a z `link rel=canonical` albo `og:url`. Pozwala rozwinąć względne linki."""
    parser = _CanonicalParser()
    parser.feed(html or "")
    parser.close()
    url = (parser.url or "").strip()
    return url if urlparse(url).scheme in ("http", "https") else None


def declared_side(item: Dict[str, Any]) -> Optional[int]:
    """Krótszy bok z atrybutów width i height albo z parametrów w, h w adresie (sklepy podają je w URL-u)."""
    width, height = item.get("width"), item.get("height")
    if not (width and height):
        query = parse_qs(urlparse(item["url"]).query)
        try:
            width = width or int(query.get("w", query.get("width", [""]))[0])
            height = height or int(query.get("h", query.get("height", [""]))[0])
        except ValueError:
            return None
    return min(width, height) if width and height else None


def video_kind(url: str) -> str:
    path = urlparse(url).path.lower()
    return "hls" if path.endswith(".m3u8") else "file"


def _unique(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, result = set(), []
    for item in items:
        if item["url"] not in seen:
            seen.add(item["url"])
            result.append(item)
    return result


def extract_media(
    html: str,
    base_url: str,
    text: str = "",
    resources: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Obrazy, wideo i emotikony strony. `resources` to pliki, które przeglądarka faktycznie
    pobrała (łapie to media doładowane skryptami, np. strumienie HLS odtwarzacza).
    """
    parser = _MediaParser(base_url)
    parser.feed(html or "")
    parser.close()
    images, videos = parser.images, parser.videos

    for resource in resources or []:
        url, kind = resource.get("url"), resource.get("type")
        content_type = (resource.get("content_type") or "").lower()
        if not url or url.startswith("data:"):
            continue
        if kind == "image" or content_type.startswith("image/"):
            images.append({"url": url, "alt": "", "source": "network", "width": None, "height": None})
        elif kind == "media" or content_type.startswith("video/") or "mpegurl" in content_type:
            videos.append({"url": url, "kind": video_kind(url), "poster": None})

    def worth_fetching(item: Dict[str, Any]) -> bool:
        side = declared_side(item)
        return not SKIPPED_IMAGES.search(item["url"]) and (side is None or side >= DECLARED_MIN_SIDE)

    return {
        "images": [i for i in _unique(images) if worth_fetching(i)],
        "videos": _unique(videos),
        "inline_images": parser.inline_images,
        "emojis": find_emojis(text),
    }


@dataclass
class MediaLimits:
    max_images: int = 8
    max_videos: int = 2
    candidates: int = 40
    image_bytes: int = 15 * 1024 * 1024
    video_bytes: int = 60 * 1024 * 1024
    min_side: int = 120
    timeout: float = 15.0
    hls_seconds: float = 30.0


class MediaDownloader:
    def __init__(
        self,
        limits: Optional[MediaLimits] = None,
        session: Optional[requests.Session] = None,
        resolver: Callable[..., Any] = socket.getaddrinfo,
    ):
        """
        Pobiera media znalezione na stronie do analizy. Strona oszusta decyduje, jakie adresy
        trafią na listę, więc każde zapytanie (także po przekierowaniu) musi iść na adres publiczny.
        """
        self.limits = limits or MediaLimits()
        self.session = session or requests.Session()
        self.resolver = resolver

    def _check_public(self, url: str) -> None:
        check_public_url(url, self.resolver)

    def _stream(self, url: str, limit: int, sink: Callable[[bytes], None]) -> None:
        for _ in range(MAX_REDIRECTS + 1):
            self._check_public(url)
            response = self.session.get(
                url,
                stream=True,
                timeout=self.limits.timeout,
                allow_redirects=False,
                headers={"User-Agent": USER_AGENT},
            )
            try:
                if response.is_redirect:
                    url = urljoin(url, response.headers["Location"])
                    continue
                response.raise_for_status()
                total = 0
                for chunk in response.iter_content(65536):
                    total += len(chunk)
                    if total > limit:
                        raise ValueError(f"plik większy niż limit {limit // (1024 * 1024)} MB")
                    sink(chunk)
                return
            finally:
                response.close()
        raise ValueError("zbyt wiele przekierowań")

    def _fetch(self, url: str, limit: int) -> bytes:
        data = bytearray()
        self._stream(url, limit, data.extend)
        return bytes(data)

    def _fetch_file(self, url: str, path: Path, limit: int) -> None:
        with open(path, "wb") as out:
            self._stream(url, limit, out.write)

    def _download_image(self, url: str, folder: Path, seen: set) -> Dict[str, Any]:
        data = self._fetch(url, self.limits.image_bytes)
        with Image.open(io.BytesIO(data)) as image:
            width, height, fmt = image.width, image.height, (image.format or "").lower()
        if min(width, height) < self.limits.min_side:
            raise ValueError(f"za mały obraz ({width}x{height})")
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            raise ValueError("duplikat innego obrazu")
        seen.add(digest)
        suffix = {"jpeg": ".jpg", "mpo": ".jpg"}.get(fmt, f".{fmt or 'img'}")
        path = folder / f"image_{len(seen)}{suffix}"
        path.write_bytes(data)
        return {"file_path": str(path), "width": width, "height": height, "bytes": len(data)}

    def _hls_segments(self, url: str, depth: int = 0) -> List[Tuple[str, float]]:
        playlist = self._fetch(url, 1024 * 1024).decode("utf-8", errors="replace")
        if "#EXT-X-KEY" in playlist and "METHOD=NONE" not in playlist:
            raise ValueError("strumień jest szyfrowany")
        if "#EXT-X-MAP" in playlist:
            raise ValueError("strumień fMP4 nie jest obsługiwany")
        lines = [line.strip() for line in playlist.splitlines() if line.strip()]

        if any(line.startswith("#EXT-X-STREAM-INF") for line in lines):
            if depth:
                raise ValueError("zagnieżdżona lista odtwarzania")
            variants = [urljoin(url, lines[i + 1]) for i, line in enumerate(lines[:-1]) if line.startswith("#EXT-X-STREAM-INF")]
            return self._hls_segments(variants[0], depth + 1)

        segments, duration = [], 0.0
        for line in lines:
            if line.startswith("#EXTINF:"):
                duration = float(line[8:].split(",")[0] or 0)
            elif not line.startswith("#"):
                segments.append((urljoin(url, line), duration))
        return segments

    def _download_hls(self, url: str, path: Path) -> None:
        collected = 0.0
        with open(path, "wb") as out:
            for segment, duration in self._hls_segments(url):
                self._stream(segment, self.limits.video_bytes, out.write)
                collected += duration
                if collected >= self.limits.hls_seconds or out.tell() >= self.limits.video_bytes:
                    break
        if collected == 0 and path.stat().st_size == 0:
            raise ValueError("pusta lista odtwarzania")

    def download(self, media: Dict[str, Any], folder: Path) -> List[Dict[str, Any]]:
        """Wynik na każdy kandydat: pobrany plik albo powód pominięcia. Obrazy od największych."""
        folder.mkdir(parents=True, exist_ok=True)
        results: List[Dict[str, Any]] = []

        seen: set = set()
        images: List[Dict[str, Any]] = []
        for candidate in media.get("images", [])[: self.limits.candidates]:
            entry = {"kind": "image", "url": candidate["url"]}
            try:
                entry.update(self._download_image(candidate["url"], folder, seen), status="ok")
            except Exception as exc:
                entry.update(status="skipped", error=f"{type(exc).__name__}: {exc}")
            images.append(entry)

        good = sorted((e for e in images if e["status"] == "ok"), key=lambda e: -e["width"] * e["height"])
        for extra in good[self.limits.max_images :]:
            Path(extra["file_path"]).unlink(missing_ok=True)
            extra.update(status="skipped", error="zbyt wiele obrazów, zostały największe")
            extra.pop("file_path")
        results.extend(images)

        downloaded = 0
        for index, candidate in enumerate(media.get("videos", []), 1):
            entry = {"kind": "video", "url": candidate["url"], "video_kind": candidate["kind"]}
            if candidate["kind"] == "embed":
                entry.update(status="skipped", error="osadzone wideo nie jest pobierane")
            elif downloaded >= self.limits.max_videos:
                entry.update(status="skipped", error="zbyt wiele filmów")
            else:
                suffix = ".ts" if candidate["kind"] == "hls" else (Path(urlparse(candidate["url"]).path).suffix or ".mp4")
                path = folder / f"video_{index}{suffix}"
                try:
                    if candidate["kind"] == "hls":
                        self._download_hls(candidate["url"], path)
                    else:
                        self._fetch_file(candidate["url"], path, self.limits.video_bytes)
                    entry.update(status="ok", file_path=str(path), bytes=path.stat().st_size)
                    downloaded += 1
                except Exception as exc:
                    path.unlink(missing_ok=True)
                    entry.update(status="skipped", error=f"{type(exc).__name__}: {exc}")
            results.append(entry)
        return results
