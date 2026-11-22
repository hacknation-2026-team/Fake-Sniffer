from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from .profile import USABLE_STATUSES, profile_from_report, registered_domain
from .similarity import SHORTENERS

HIGH = 0.5
MEDIUM = 0.25
LABELS = {"high": "Prawdopodobnie fake", "mid": "Podejrzane", "low": "Raczej nie fake", "unknown": "Za mało danych"}

SUSPICIOUS_TLDS = frozenset(
    {"top", "xyz", "click", "shop", "online", "site", "icu", "cfd", "sbs", "buzz", "rest", "vip", "work", "live", "store"}
)

PHRASES: Dict[str, List[str]] = {
    "presja czasu": [
        "tylko dzis", "ostatnie sztuki", "ostatnia szansa", "ograniczona liczba", "zostalo tylko",
        "oferta wygasa", "spiesz sie", "tylko do konca", "tylko przez",
    ],
    "obietnica zysku": [
        "gwarantowany zysk", "gwarantowany dochod", "bez ryzyka", "pasywny dochod", "zarob ", "zarabiaj",
        "podwoj", "inwestycja", "kryptowalut", "bitcoin", "forex", "zl dziennie", "zl dziennie",
    ],
    "wyłudzenie danych": [
        "podaj dane", "podaj numer", "pesel", "dane karty", "numer karty", "zweryfikuj konto",
        "potwierdz tozsamosc", "kod sms", "dopłata", "doplata",
    ],
    "sensacyjny język": [
        "szok", "nie uwierzysz", "ujawnil", "sekret", "lekarze nienawidza", "polecaja", "afera", "ukrywaja",
    ],
    "nagroda": ["wygrales", "wygrana", "gratulacje", "darmowy", "za darmo", "nagroda", "prezent od"],
}
MONEY = "obietnica zysku"


@dataclass
class Signal:
    text: str
    weight: float

    @property
    def kind(self) -> str:
        return "risk" if self.weight > 0 else "safe"


@dataclass
class Assessment:
    score: float
    level: str
    signals: List[Signal] = field(default_factory=list)
    coverage: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return LABELS[self.level]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "level": self.level,
            "label": self.label,
            "coverage": self.coverage,
            "signals": [{"text": s.text, "weight": s.weight, "kind": s.kind} for s in self.signals],
        }


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower().replace("ł", "l"))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _steps(report: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    return [
        s["data"]
        for s in report.get("steps", [])
        if s.get("name") == name and s.get("status") in USABLE_STATUSES and s.get("data") is not None
    ]


def _texts(report: Dict[str, Any]) -> str:
    parts: List[str] = []
    for data in _steps(report, "web_infrastructure"):
        parts.append(data.get("page_text") or "")
    for data in _steps(report, "ocr"):
        parts.append(data.get("text") or "")
    for data in _steps(report, "audio_features"):
        parts.append(data.get("text") or "")
    for data in _steps(report, "video_features"):
        parts.append(data.get("ocr_text") or "")
    return _plain("\n".join(parts))


def _infrastructure(report: Dict[str, Any]) -> List[Signal]:
    signals: List[Signal] = []
    for web in _steps(report, "web_infrastructure"):
        age = (web.get("rdap") or {}).get("age_days")
        if age is not None:
            if age < 30:
                signals.append(Signal(f"domena zarejestrowana zaledwie {age} dni temu", 0.35))
            elif age < 180:
                signals.append(Signal(f"domena młoda: {age} dni", 0.15))
            elif age >= 365 * 3:
                signals.append(Signal(f"domena istnieje od {age // 365} lat", -0.2))

        chain = web.get("redirect_chain") or []
        domains = []
        for hop in chain:
            domain = registered_domain(hop.get("url"))
            if domain and domain not in domains:
                domains.append(domain)
        if len(domains) >= 3:
            signals.append(Signal(f"przekierowania przez {len(domains)} różnych domen", 0.2))
        elif len(domains) == 2:
            signals.append(Signal("przekierowanie na inną domenę", 0.05))
        if any(d in SHORTENERS for d in domains):
            signals.append(Signal("w łańcuchu jest skracacz linków", 0.1))

        final = web.get("final_url") or web.get("input_url") or ""
        domain = web.get("registered_domain") or registered_domain(final)
        if domain:
            name, _, tld = domain.rpartition(".")
            if tld in SUSPICIOUS_TLDS:
                signals.append(Signal(f"domena w często nadużywanej strefie .{tld}", 0.1))
            if name.count("-") >= 2 or len(name) > 25:
                signals.append(Signal("nazwa domeny długa lub z wieloma myślnikami", 0.08))
        if final.startswith("http://"):
            signals.append(Signal("strona bez szyfrowania HTTPS", 0.08))
        break
    return signals


def _content(report: Dict[str, Any], identities: Iterable[str]) -> List[Signal]:
    text = _texts(report)
    signals: List[Signal] = []
    matched: Dict[str, List[str]] = {}
    for category, phrases in PHRASES.items():
        hits = [p.strip() for p in phrases if p in text]
        if hits:
            matched[category] = hits
    for category, hits in list(matched.items())[:3]:
        signals.append(Signal(f"{category}: „{hits[0]}”", 0.12))

    names = sorted(set(identities))
    if names and MONEY in matched:
        signals.append(Signal(f"osoba publiczna ({', '.join(names[:2])}) obok obietnicy zysku", 0.25))

    for web in _steps(report, "web_infrastructure"):
        emojis = (web.get("media") or {}).get("emojis") or {}
        if emojis.get("total", 0) >= 5 and emojis.get("density", 0) >= 6:
            signals.append(Signal(f"dużo emotikon: {emojis['total']} ({emojis['density']} na 1000 znaków)", 0.08))
        break
    return signals


def _peak(report: Dict[str, Any], names: Iterable[str]) -> Optional[float]:
    values: List[float] = []
    for name in names:
        for data in _steps(report, name):
            value = data.get("max_fake_probability", data.get("fake_probability"))
            if value is not None:
                values.append(value)
    return max(values) if values else None


def _media(report: Dict[str, Any]) -> List[Signal]:
    signals: List[Signal] = []
    ai = _peak(report, ("ai_image", "ai_frames"))
    if ai is not None and ai >= 0.8:
        signals.append(Signal(f"obraz prawdopodobnie wygenerowany przez AI ({ai:.0%})", 0.2 if ai >= 0.95 else 0.1))
    deepfake = _peak(report, ("deepfake_image", "deepfake_frames"))
    if deepfake is not None and deepfake >= 0.7:
        signals.append(Signal(f"twarz wygląda na deepfake ({deepfake:.0%})", 0.2 if deepfake >= 0.9 else 0.1))
    voice = _peak(report, ("deepfake_audio",))
    if voice is not None and voice >= 0.7:
        signals.append(Signal(f"głos wygląda na syntetyczny ({voice:.0%})", 0.2))
    return signals


def assess(report: Dict[str, Any]) -> Assessment:
    """
    Reguły z wagami, a nie wyuczony model: każdy sygnał ma powód po polsku, więc wynik da się
    sprawdzić i poprawić. Wagi są dobrane ręcznie i wymagają kalibracji na prawdziwych danych.
    """
    identities = profile_from_report(report).identities
    coverage = [
        label
        for label, present in (
            ("infrastruktura", bool(_steps(report, "web_infrastructure"))),
            ("treść", bool(_texts(report).strip())),
            ("obrazy i dźwięk", any(_steps(report, n) for n in ("ai_image", "ai_frames", "deepfake_image", "deepfake_frames", "deepfake_audio"))),
        )
        if present
    ]
    signals = _infrastructure(report) + _content(report, identities) + _media(report)
    score = round(min(max(sum(s.weight for s in signals), 0.0), 1.0), 2)

    if not coverage:
        level = "unknown"
    elif score >= HIGH:
        level = "high"
    elif score >= MEDIUM:
        level = "mid"
    else:
        level = "low"
    return Assessment(score, level, sorted(signals, key=lambda s: -abs(s.weight)), coverage)


def describe_risk(assessment: Assessment) -> List[str]:
    lines = [f"  ryzyko: {assessment.label} (wynik {assessment.score:.2f}, dane: {', '.join(assessment.coverage) or 'brak'})"]
    lines += [f"  {'+' if s.weight > 0 else '-'} {s.text}" for s in assessment.signals]
    return lines
