from __future__ import annotations

import ipaddress
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from feature_extractor.page_media import extract_media, find_page_url

from .profile import registered_domain

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
HTML_EXTENSIONS = {".html", ".htm"}
MAX_REDIRECTS = 20


@dataclass
class Case:
    """Jeden incydent z zebranych materiałów: pliki z folderu i jego podfolderów."""

    name: str
    path: Path
    html: List[Path] = field(default_factory=list)
    har: List[Path] = field(default_factory=list)
    images: List[Path] = field(default_factory=list)
    videos: List[Path] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.html or self.har or self.images or self.videos)


def _natural(path: Path) -> List[Any]:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]


def _is_resource_dir(name: str) -> bool:
    return name.endswith("_files") or name == "resources"


def collect_case(path: Path) -> Case:
    """Pomija katalogi z zasobami zapisanych stron (`*_files`, `resources`)."""
    case = Case(name=path.name, path=path)
    for folder, subdirs, files in os.walk(path):
        subdirs[:] = sorted((d for d in subdirs if not _is_resource_dir(d)), key=str.lower)
        for name in sorted(files, key=str.lower):
            file = Path(folder, name)
            suffix = file.suffix.lower()
            if suffix in HTML_EXTENSIONS:
                case.html.append(file)
            elif suffix == ".har":
                case.har.append(file)
            elif suffix in IMAGE_EXTENSIONS:
                case.images.append(file)
            elif suffix in VIDEO_EXTENSIONS:
                case.videos.append(file)
    return case


def find_cases(root: str | Path) -> List[Case]:
    folders = sorted((p for p in Path(root).iterdir() if p.is_dir()), key=_natural)
    return [collect_case(folder) for folder in folders]


def _is_document(entry: Dict[str, Any]) -> bool:
    kind = entry.get("_resourceType")
    if kind:
        return kind == "document"
    response = entry.get("response", {})
    mime = (response.get("content") or {}).get("mimeType") or ""
    return "html" in mime.lower() or bool(response.get("redirectURL"))


def har_infrastructure(path: str | Path) -> Optional[Dict[str, Any]]:
    """
    Ślad infrastruktury z zapisu HAR: łańcuch przekierowań głównego dokumentu,
    domena końcowa i adresy IP serwerów. Zasoby zewnętrzne (trackery, CDN) są pomijane.
    """
    entries = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))["log"]["entries"]
    start = next((e for e in entries if _is_document(e)), entries[0] if entries else None)
    if start is None:
        return None

    by_url = {}
    for entry in entries:
        by_url.setdefault(entry["request"]["url"], entry)

    chain: List[Dict[str, Any]] = []
    ips: List[str] = []
    current: Optional[Dict[str, Any]] = start
    final_url = start["request"]["url"]
    while current is not None and len(chain) < MAX_REDIRECTS:
        url = current["request"]["url"]
        status = current.get("response", {}).get("status", 0)
        target = current.get("response", {}).get("redirectURL") or None
        chain.append({"url": url, "status": status, "location": target})
        ip = (current.get("serverIPAddress") or "").strip("[]")
        if ip:
            ips.append(ip)
        final_url = urljoin(url, target) if target else url
        current = by_url.get(final_url) if target else None

    return {
        "registered_domain": registered_domain(start["request"]["url"]),
        "final_url": final_url,
        "redirect_chain": chain,
        "dns": {"a": sorted(set(ips)), "aaaa": [], "asns": []},
    }


def _is_public_url(url: Optional[str]) -> bool:
    host = urlparse(url or "").hostname
    if not host or host == "localhost" or host.endswith(".local"):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return address.is_global


def har_metadata(path: str | Path) -> Dict[str, Any]:
    """Trackery ze wszystkich zapytań HAR i sygnały z adresów głównego dokumentu."""
    from feature_extractor.file_metadata_extractor import find_trackers, url_signals

    entries = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))["log"]["entries"]
    urls = [e["request"]["url"] for e in entries]
    documents = [e["request"]["url"] for e in entries if _is_document(e)]
    signals = url_signals(documents)
    trackers = {(t["kind"], t["id"]): t for t in find_trackers("\n".join(urls)) + signals["trackers"]}
    return {**signals, "trackers": sorted(trackers.values(), key=lambda t: (t["kind"], t["id"]))}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _step(name: str, source: str, data: Any, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "name": name,
        "source": source,
        "status": "error" if error else "ok",
        "duration_sec": 0.0,
        "data": data,
        "error": error,
    }


def _safe(steps: List[Dict[str, Any]], name: str, source: str, action: Callable[[], Any]) -> Any:
    try:
        data = action()
    except Exception as exc:
        steps.append(_step(name, source, None, f"{type(exc).__name__}: {exc}"))
        return None
    steps.append(_step(name, source, data))
    return data


def run_input(orchestrator: Any, source: str, **inputs: str) -> List[Dict[str, Any]]:
    steps = orchestrator.run(**inputs).to_dict()["steps"]
    return [step if step["source"].startswith("page_") else {**step, "source": source} for step in steps]


def _har_final_url(steps: List[Dict[str, Any]]) -> Optional[str]:
    for step in steps:
        url = (step.get("data") or {}).get("final_url")
        if step["name"] == "web_infrastructure" and url:
            return url
    return None


def build_case_report(
    case: Case,
    orchestrator: Any,
    keyframes_root: Optional[Path] = None,
    images: bool = True,
    videos: bool = True,
) -> Dict[str, Any]:
    """
    Raport w tym samym kształcie co z main.py, zbudowany z plików incydentu: HAR daje
    infrastrukturę, zapisany HTML szablon i tekst strony, a obrazy i filmy idą przez ekstraktory.
    """
    from feature_extractor.file_metadata_extractor import FileMetadataExtractor
    from feature_extractor.web_infrastructure_extractor import WebInfrastructureExtractor

    steps: List[Dict[str, Any]] = []
    for har in case.har:
        _safe(steps, "web_infrastructure", "har", lambda har=har: har_infrastructure(har))
        _safe(steps, "page_metadata", "har", lambda har=har: har_metadata(har))

    metadata = FileMetadataExtractor()
    htmls = [_read(page) for page in case.html]
    texts = [WebInfrastructureExtractor.visible_text(html) for html in htmls]
    for html in htmls:
        meta = _safe(steps, "page_metadata", "html", lambda html=html: metadata.extract_html(html))
        saved_from = (meta or {}).get("saved_from")
        if _is_public_url(saved_from):
            _safe(
                steps,
                "web_infrastructure",
                "html",
                lambda saved_from=saved_from: {
                    "registered_domain": registered_domain(saved_from),
                    "final_url": saved_from,
                },
            )
    text = "\n\n".join(t for t in texts if t.strip())
    page_url = (_har_final_url(steps) or next((u for u in map(find_page_url, htmls) if u), None)) if htmls else None
    media = extract_media("\n".join(htmls), page_url or "", text) if htmls else None
    if htmls:
        _safe(
            steps,
            "web_infrastructure",
            "html",
            lambda: {"dom_fingerprint": WebInfrastructureExtractor.dom_fingerprint(htmls[0]), "media": media},
        )

    if text:
        steps += run_input(orchestrator, "html", text=text)

    found_media = media and (media["images"] or media["videos"])
    if found_media and getattr(getattr(orchestrator, "config", None), "page_media", False):
        steps += orchestrator.run_page_media(media, page_url or f"case:{case.name}")

    if images:
        for path in case.images:
            steps += run_input(orchestrator, "image", image=str(path))

    if videos:
        for path in case.videos:
            if keyframes_root is not None:
                orchestrator.config.keyframes_dir = str(keyframes_root / case.name / path.stem)
            steps += run_input(orchestrator, "video", video=str(path))

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "case": case.name,
            "html": [str(p.relative_to(case.path)) for p in case.html],
            "har": [str(p.relative_to(case.path)) for p in case.har],
            "images": [str(p.relative_to(case.path)) for p in case.images],
            "videos": [str(p.relative_to(case.path)) for p in case.videos],
        },
        "steps": steps,
    }
