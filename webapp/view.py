from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from correlation.profile import Profile
from feature_extractor.orchestrator import VECTOR_KEYS

MAX_TEXT = 300
MAX_ITEMS = 20


def trim(value: Any) -> Any:
    """Skraca dane kroku do wielkości sensownej w przeglądarce i usuwa wektory."""
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "…"
    if isinstance(value, dict):
        return {
            str(k): None if k in VECTOR_KEYS else trim(v) for k, v in list(value.items())[:40]
        }
    if isinstance(value, (list, tuple)):
        items = [trim(v) for v in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"… (jeszcze {len(value) - MAX_ITEMS})")
        return items
    if hasattr(value, "tolist"):
        return trim(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _names(identities: List[Optional[Dict[str, Any]]]) -> str:
    names = sorted({i["name"] for i in identities if i})
    return ", ".join(names)


def _web(d: Dict[str, Any]) -> str:
    parts = [d.get("final_url") or d.get("registered_domain") or "strona"]
    hops = len(d.get("redirect_chain") or [])
    if hops > 1:
        parts.append(f"przekierowania: {hops - 1}")
    ips = len((d.get("dns") or {}).get("a") or [])
    if ips:
        parts.append(f"IP: {ips}")
    return ", ".join(parts)


def _entities(d: List[Dict[str, Any]]) -> str:
    shown = ", ".join(e["text"] for e in d[:5])
    return f"{len(d)} encji" + (f": {shown}" if shown else "")


def _faces(d: List[Dict[str, Any]]) -> str:
    names = _names([f.get("identity") for f in d])
    return f"{len(d)} twarzy" + (f" ({names})" if names else "")


def _fake(value: Optional[float], kind: str = "") -> str:
    if value is None:
        return "nie rozpoznano etykiety fałszerstwa"
    return f"prawdopodobieństwo fałszerstwa {kind + ' ' if kind else ''}{value:.0%}"


SUMMARIES: Dict[str, Callable[[Any], str]] = {
    "web_infrastructure": _web,
    "named_entities": _entities,
    "text_embeddings": lambda d: f"{len(d)} fragmentów tekstu",
    "visual_hashes": lambda d: f"pHash {d['phash']}",
    "visual_embedding": lambda d: f"wektor {len(d)}-wymiarowy",
    "faces": _faces,
    "face_identities": lambda d: _names(d) or "nikogo nie rozpoznano",
    "ocr": lambda d: f"{len(d.get('text', ''))} znaków, {len(d.get('lines', []))} linii",
    "video_features": lambda d: f"{len(d['keyframes'])} klatek, {len(d['face_embeddings'])} twarzy",
    "deepfake_image": lambda d: _fake(d.get("fake_probability")),
    "deepfake_frames": lambda d: f"{len(d['frames'])} klatek, " + _fake(d.get("max_fake_probability"), "max"),
    "deepfake_audio": lambda d: _fake(d.get("max_fake_probability"), "max"),
    "audio_features": lambda d: f"transkrypcja: {len(d.get('text') or '')} znaków",
}


def summarize_step(step: Dict[str, Any]) -> str:
    data, status = step.get("data"), step.get("status")
    if data is None:
        return step.get("error") or ("brak danych" if status == "ok" else status) or "brak danych"
    try:
        text = SUMMARIES[step["name"]](data) if step["name"] in SUMMARIES else "ok"
    except (KeyError, TypeError, ValueError):
        text = "ok"
    if status == "partial" and step.get("error"):
        text += f" ({step['error']})"
    return text


def step_view(step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": step["name"],
        "source": step.get("source"),
        "status": step.get("status"),
        "duration_sec": step.get("duration_sec"),
        "error": step.get("error"),
        "summary": summarize_step(step),
        "data": trim(step.get("data")),
    }


def short(value: Optional[str], length: int = 16) -> Optional[str]:
    return value[:length] if value else None


def profile_view(profile: Profile) -> Dict[str, Any]:
    return {
        "domains": profile.domains,
        "ips": profile.ips,
        "asns": profile.asns,
        "nameservers": profile.nameservers,
        "registrar": profile.registrar,
        "dom_hash": short(profile.dom_hash),
        "cert_sha256": short(profile.cert_sha256),
        "entities": profile.entities,
        "identities": profile.identities,
        "phashes": profile.phashes,
        "dhashes": len(profile.dhashes),
        "faces": len(profile.face_embeddings),
        "has_text_embedding": profile.text_embedding is not None,
        "has_visual_embedding": profile.visual_embedding is not None,
    }


HIGH_RISK = 0.7
MEDIUM_RISK = 0.4
DEEPFAKE_KINDS = {"deepfake_image": "obraz", "deepfake_frames": "klatki wideo", "deepfake_audio": "głos"}
LEVELS = {
    "high": "podejrzane",
    "mid": "niejednoznaczne",
    "low": "raczej autentyczne",
    "unknown": "brak oceny",
}


def risk_level(probability: Optional[float]) -> str:
    if probability is None:
        return "unknown"
    if probability >= HIGH_RISK:
        return "high"
    return "mid" if probability >= MEDIUM_RISK else "low"


def deepfake_rows(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Wiersze tabeli deepfake: jeden na obraz, film lub nagranie. Oceniamy po wartości maksymalnej."""
    rows: List[Dict[str, Any]] = []
    for step in steps:
        kind = DEEPFAKE_KINDS.get(step["name"])
        data = step.get("data")
        if kind is None or (data is None and step.get("status") != "error"):
            continue
        data = data or {}
        mean = data.get("mean_fake_probability", data.get("fake_probability"))
        peak = data.get("max_fake_probability", data.get("fake_probability"))
        level = risk_level(peak)
        rows.append(
            {
                "item": data.get("file") or step.get("source"),
                "kind": kind,
                "mean": mean,
                "max": peak,
                "level": level,
                "verdict": LEVELS[level],
                "model": data.get("model"),
                "error": step.get("error") if step.get("status") == "error" else None,
            }
        )
    return rows
