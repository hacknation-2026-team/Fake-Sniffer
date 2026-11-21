from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .profile import Profile

WEIGHTS: Dict[str, float] = {"infra": 0.35, "face": 0.25, "visual": 0.20, "text": 0.20}
THRESHOLD = 0.65
WEAK_THRESHOLD = 0.45
TIGHT_THRESHOLD = 0.85

TEXT_FLOOR = 0.3
CLIP_FLOOR = 0.5
FACE_FLOOR = 0.25
FACE_CEILING = 0.6
HASH_RANGE = 32

DUPLICATE_PHASH_DISTANCE = 4
DUPLICATE_TEXT_COSINE = 0.95

SHORTENERS = frozenset({"bit.ly", "t.co", "tinyurl.com", "goo.gl", "is.gd", "cutt.ly", "ow.ly", "rebrand.ly"})
FEATURE_NAMES = {"infra": "infrastruktura", "face": "twarz", "visual": "obraz", "text": "tekst"}


@dataclass
class Comparison:
    """Wynik porównania dwóch profili: ocena, wkład modalności i powody."""

    score: float
    modalities: Dict[str, float] = field(default_factory=dict)
    contributions: Dict[str, float] = field(default_factory=dict)
    hard: List[str] = field(default_factory=list)
    duplicate: bool = False
    reasons: List[str] = field(default_factory=list)

    def linked(self, threshold: float = THRESHOLD) -> bool:
        return bool(self.hard) or self.score >= threshold

    @property
    def rank(self) -> Tuple[bool, float]:
        return (bool(self.hard), self.score)


def _clip(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _rescale(value: float, floor: float, ceiling: float = 1.0) -> float:
    return _clip((value - floor) / (ceiling - floor))


def cosine(a: List[float], b: List[float]) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / denom) if denom else 0.0


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def hamming(a: str, b: str) -> Optional[int]:
    if len(a) != len(b):
        return None
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def min_hash_distance(hashes_a: List[str], hashes_b: List[str]) -> Optional[int]:
    distances = [d for x in hashes_a for y in hashes_b if (d := hamming(x, y)) is not None]
    return min(distances) if distances else None


def text_cosine(a: Profile, b: Profile) -> Optional[float]:
    if a.text_embedding and b.text_embedding:
        return cosine(a.text_embedding, b.text_embedding)
    return None


def _infra(a: Profile, b: Profile) -> Optional[Tuple[float, List[str], List[str]]]:
    def present(p: Profile) -> bool:
        return bool(p.domains or p.ips or p.dom_hash or p.cert_sha256)

    if not (present(a) and present(b)):
        return None

    hard: List[str] = []
    for domain in sorted((set(a.domains) & set(b.domains)) - SHORTENERS):
        hard.append(f"wspólna domena {domain}")
    if a.dom_hash and a.dom_hash == b.dom_hash:
        hard.append("identyczny szablon strony (DOM)")
    if a.cert_sha256 and a.cert_sha256 == b.cert_sha256:
        hard.append("ten sam certyfikat TLS")

    soft: List[Tuple[float, str]] = []
    if a.dom_paths and b.dom_paths:
        overlap = jaccard(a.dom_paths, b.dom_paths)
        if overlap >= 0.3:
            soft.append((overlap, f"podobna struktura strony ({overlap:.0%} wspólnych ścieżek DOM)"))
    if set(a.ips) & set(b.ips):
        soft.append((0.5, "ten sam adres IP"))
    if set(a.asns) & set(b.asns):
        soft.append((0.4, "ta sama sieć hostingowa (ASN)"))
    if set(a.nameservers) & set(b.nameservers):
        soft.append((0.4, "te same serwery nazw"))

    score = 1.0 if hard else max((s for s, _ in soft), default=0.0)
    return score, hard, hard + [text for _, text in soft]


def _face(a: Profile, b: Profile) -> Optional[Tuple[float, List[str]]]:
    if not ((a.face_embeddings or a.identities) and (b.face_embeddings or b.identities)):
        return None

    shared = sorted(set(a.identities) & set(b.identities))
    if shared:
        return 1.0, [f"ta sama osoba: {', '.join(shared)}"]

    best = 0.0
    if a.face_embeddings and b.face_embeddings:
        best = max(cosine(x, y) for x in a.face_embeddings for y in b.face_embeddings)
    score = _rescale(best, FACE_FLOOR, FACE_CEILING)
    reasons = [f"podobna twarz (cosinus {best:.2f})"] if score >= 0.5 else []
    return score, reasons


def _visual(a: Profile, b: Profile) -> Optional[Tuple[float, List[str]]]:
    parts: List[float] = []
    reasons: List[str] = []

    pairs = ((a.phashes, b.phashes, "pHash"), (a.dhashes, b.dhashes, "dHash"))
    for left, right, name in pairs:
        distance = min_hash_distance(left, right)
        if distance is not None:
            parts.append(_clip(1 - distance / HASH_RANGE))
            if distance <= 8:
                reasons.append(f"zbliżony obraz ({name}, odległość {distance})")
            break

    if a.visual_embedding and b.visual_embedding:
        similarity = cosine(a.visual_embedding, b.visual_embedding)
        parts.append(_rescale(similarity, CLIP_FLOOR))
        if similarity >= 0.85:
            reasons.append(f"podobna zawartość obrazu (CLIP {similarity:.2f})")

    return (float(np.mean(parts)), reasons) if parts else None


def _text(a: Profile, b: Profile) -> Optional[Tuple[float, List[str]]]:
    parts: List[Tuple[float, float]] = []
    reasons: List[str] = []

    similarity = text_cosine(a, b)
    if similarity is not None:
        parts.append((0.7, _rescale(similarity, TEXT_FLOOR)))
        if similarity >= 0.8:
            reasons.append(f"podobny tekst (embedding {similarity:.2f})")

    if a.entities and b.entities:
        overlap = jaccard(a.entities, b.entities)
        parts.append((0.3, overlap))
        shared = sorted(set(a.entities) & set(b.entities))
        if shared:
            names = ", ".join(key.split(":", 1)[1] for key in shared[:5])
            reasons.append(f"wspólne nazwy własne: {names}")

    if not parts:
        return None
    total = sum(w for w, _ in parts)
    return sum(w * s for w, s in parts) / total, reasons


def compare(a: Profile, b: Profile) -> Comparison:
    """
    Średnia ważona po modalnościach dostępnych w obu profilach (brakujące
    nie obniżają wyniku). Twardy ślad infrastruktury jest zgłaszany osobno.
    """
    scores: Dict[str, float] = {}
    reasons: List[str] = []
    hard: List[str] = []

    infra = _infra(a, b)
    if infra:
        scores["infra"], hard, infra_reasons = infra
        reasons += infra_reasons
    for name, result in (("face", _face(a, b)), ("visual", _visual(a, b)), ("text", _text(a, b))):
        if result:
            scores[name], found = result
            reasons += found

    if not scores:
        return Comparison(score=0.0)

    total = sum(WEIGHTS[m] for m in scores)
    contributions = {m: WEIGHTS[m] * s / total for m, s in scores.items()}

    distance = min_hash_distance(a.phashes, b.phashes)
    cosine_text = text_cosine(a, b)
    duplicate = (
        distance is not None
        and cosine_text is not None
        and distance <= DUPLICATE_PHASH_DISTANCE
        and cosine_text > DUPLICATE_TEXT_COSINE
    )
    if duplicate:
        reasons.insert(0, "duplikat lub niemal duplikat (EXACT_OR_NEAR_DUPLICATE)")
    return Comparison(
        score=round(sum(contributions.values()), 4),
        modalities={m: round(s, 4) for m, s in scores.items()},
        contributions={m: round(c, 4) for m, c in contributions.items()},
        hard=hard,
        duplicate=duplicate,
        reasons=reasons,
    )
