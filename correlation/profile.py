from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import tldextract

_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())
USABLE_STATUSES = ("ok", "partial")


def registered_domain(url_or_host: Optional[str]) -> Optional[str]:
    if not url_or_host:
        return None
    parts = _TLD_EXTRACT(url_or_host)
    if parts.domain and parts.suffix:
        return f"{parts.domain}.{parts.suffix}".lower()
    return parts.domain.lower() or None


def _vector(values: Optional[Iterable[float]]) -> Optional[List[float]]:
    if values is None:
        return None
    array = np.asarray(list(values), dtype=float)
    return [round(float(v), 5) for v in array] if array.size else None


def _mean_vector(vectors: List[List[float]], weights: List[float]) -> Optional[List[float]]:
    if not vectors:
        return None
    mean = np.average(np.asarray(vectors, dtype=float), axis=0, weights=weights)
    norm = np.linalg.norm(mean)
    return _vector(mean / norm) if norm else None


def _unique(values: Iterable[Optional[str]]) -> List[str]:
    return sorted({v for v in values if v})


@dataclass
class Profile:
    """Cechy jednego incydentu wyciągnięte z raportu, gotowe do porównań."""

    text_embedding: Optional[List[float]] = None
    entities: List[str] = field(default_factory=list)
    phashes: List[str] = field(default_factory=list)
    dhashes: List[str] = field(default_factory=list)
    visual_embedding: Optional[List[float]] = None
    face_embeddings: List[List[float]] = field(default_factory=list)
    identities: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    ips: List[str] = field(default_factory=list)
    asns: List[str] = field(default_factory=list)
    nameservers: List[str] = field(default_factory=list)
    registrar: Optional[str] = None
    dom_hash: Optional[str] = None
    dom_paths: List[str] = field(default_factory=list)
    cert_sha256: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not any(getattr(self, f.name) for f in fields(self))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Profile":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _steps(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        s for s in report.get("steps", [])
        if s.get("status") in USABLE_STATUSES and s.get("data") is not None
    ]


def _add_infrastructure(profile: Profile, data: Dict[str, Any]) -> None:
    hosts = [data.get("registered_domain"), registered_domain(data.get("final_url"))]
    hosts += [registered_domain(hop.get("url")) for hop in data.get("redirect_chain") or []]
    profile.domains = _unique(profile.domains + hosts)

    dns = data.get("dns") or {}
    profile.ips = _unique(profile.ips + list(dns.get("a") or []) + list(dns.get("aaaa") or []))
    profile.asns = _unique(profile.asns + [str(a["asn"]) for a in dns.get("asns") or [] if a.get("asn")])

    rdap = data.get("rdap") or {}
    profile.nameservers = _unique(profile.nameservers + [n.lower() for n in rdap.get("nameservers") or []])
    profile.registrar = profile.registrar or rdap.get("registrar")

    dom = data.get("dom_fingerprint") or {}
    profile.dom_hash = profile.dom_hash or dom.get("tag_class_hash")
    profile.dom_paths = _unique(profile.dom_paths + list(dom.get("path_hashes") or []))

    profile.cert_sha256 = profile.cert_sha256 or (data.get("tls") or {}).get("sha256")


def _add_faces(profile: Profile, faces: List[Dict[str, Any]]) -> None:
    for face in faces:
        if face.get("embedding"):
            profile.face_embeddings.append(_vector(face["embedding"]))
        name = (face.get("identity") or {}).get("name")
        if name:
            profile.identities = _unique(profile.identities + [name])


def profile_from_report(report: Dict[str, Any]) -> Profile:
    """Zbiera cechy ze wszystkich udanych kroków raportu, niezależnie od źródła."""
    profile = Profile()
    chunk_vectors: List[List[float]] = []
    chunk_weights: List[float] = []
    entities: List[str] = []

    for step in _steps(report):
        name, data = step["name"], step["data"]
        if name == "web_infrastructure":
            _add_infrastructure(profile, data)
        elif name == "named_entities":
            entities += [f"{e['label']}:{(e.get('lemma') or e['text']).casefold()}" for e in data]
        elif name == "text_embeddings":
            for chunk in data:
                if chunk.get("embedding"):
                    chunk_vectors.append(chunk["embedding"])
                    chunk_weights.append(max(chunk.get("word_count") or 1, 1))
        elif name == "visual_hashes":
            profile.phashes.append(data["phash"])
            profile.dhashes.append(data["dhash"])
        elif name == "visual_embedding" and profile.visual_embedding is None:
            profile.visual_embedding = _vector(data)
        elif name == "faces":
            _add_faces(profile, data)
        elif name == "video_features":
            profile.dhashes += list(data.get("dhashes") or [])
            profile.visual_embedding = profile.visual_embedding or _vector(data.get("visual_embedding"))
            profile.face_embeddings += [_vector(e) for e in data.get("face_embeddings") or []]
        elif name == "face_identities":
            names = [m["name"] for m in data if m]
            profile.identities = _unique(profile.identities + names)

    profile.entities = _unique(entities)
    profile.text_embedding = _mean_vector(chunk_vectors, chunk_weights)
    return profile


def load_report(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
