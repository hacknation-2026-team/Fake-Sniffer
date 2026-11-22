from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Base, Incident
from .clustering import Clusters
from .profile import Profile
from .similarity import (
    FEATURE_NAMES,
    THRESHOLD,
    TIGHT_THRESHOLD,
    WEAK_THRESHOLD,
    Comparison,
    compare,
)

KNOWN = "KNOWN_CAMPAIGN"
WEAK = "WEAK_SIMILARITY"
NO_MATCH = "NO_MATCH"
NEIGHBOURS = 5


@dataclass
class Neighbour:
    incident: Incident
    comparison: Comparison
    campaign: Optional[str] = None
    family: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        c = self.comparison
        return {
            "id": self.incident.id,
            "label": self.incident.label,
            "campaign": self.campaign,
            "family": self.family,
            "score": c.score,
            "hard": c.hard,
            "duplicate": c.duplicate,
            "modalities": c.modalities,
            "reasons": c.reasons,
        }


@dataclass
class Verdict:
    verdict: str
    best_score: float
    campaign: Optional[str] = None
    family: Optional[str] = None
    top_features: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    neighbours: List[Neighbour] = field(default_factory=list)
    outdated_clusters: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "best_score": self.best_score,
            "campaign": self.campaign,
            "family": self.family,
            "top_features": self.top_features,
            "reasons": self.reasons,
            "outdated_clusters": self.outdated_clusters,
            "neighbours": [n.to_dict() for n in self.neighbours],
        }


def _dominant(neighbours: List[Neighbour], attribute: str, tight: Optional[float] = None) -> Optional[str]:
    votes: Dict[str, float] = defaultdict(float)
    for n in neighbours:
        value = getattr(n, attribute)
        c = n.comparison
        if value is None or (tight is not None and not (c.hard or c.score >= tight)):
            continue
        votes[value] += c.score + (1.0 if c.hard else 0.0)
    return max(votes, key=votes.get) if votes else None


def _top_features(best: Comparison, limit: int = 3) -> List[Dict[str, Any]]:
    ranked = sorted(best.contributions.items(), key=lambda item: item[1], reverse=True)
    return [
        {"feature": FEATURE_NAMES[name], "contribution": value, "similarity": best.modalities[name]}
        for name, value in ranked[:limit]
        if value > 0
    ]


def classify(
    profile: Profile,
    base: Base,
    clusters: Optional[Clusters] = None,
    threshold: float = THRESHOLD,
    tight: float = TIGHT_THRESHOLD,
    weak: float = WEAK_THRESHOLD,
) -> Verdict:
    """
    Porównuje profil z całą bazą i decyduje: znana kampania, słabe podobieństwo albo brak dopasowania.
    Brak dopasowania nie oznacza ani oszustwa, ani uczciwej strony: baza zna tylko swoje incydenty.
    Z `clusters` dodatkowo wskazuje kampanię i rodzinę najbliższych sąsiadów.
    """
    campaigns = clusters.campaigns if clusters else {}
    families = clusters.families if clusters else {}
    compared = sorted(
        (
            Neighbour(i, compare(profile, i.profile), campaigns.get(i.id), families.get(i.id))
            for i in base.incidents
        ),
        key=lambda n: n.comparison.rank,
        reverse=True,
    )
    if not compared or not compared[0].comparison.modalities:
        reason = "baza jest pusta" if not compared else "brak wspólnych cech do porównania"
        return Verdict(NO_MATCH, 0.0, reasons=[reason])

    best = compared[0].comparison
    linked = [n for n in compared[:NEIGHBOURS] if n.comparison.linked(threshold)]
    if linked:
        kind = KNOWN
    elif best.score >= weak:
        kind = WEAK
    else:
        kind = NO_MATCH

    if kind == NO_MATCH:
        reason = f"nic w bazie nie jest podobne: najlepszy wynik {best.score:.2f}, próg słabego podobieństwa {weak:.2f}"
        return Verdict(
            verdict=kind,
            best_score=best.score,
            reasons=[reason],
            neighbours=compared[:NEIGHBOURS],
            outdated_clusters=bool(clusters) and any(i.id not in campaigns for i in base.incidents),
        )

    return Verdict(
        verdict=kind,
        best_score=best.score,
        campaign=_dominant(linked, "campaign", tight),
        family=_dominant(linked, "family"),
        top_features=_top_features(best),
        reasons=best.reasons,
        neighbours=compared[:NEIGHBOURS],
        outdated_clusters=bool(clusters) and any(i.id not in campaigns for i in base.incidents),
    )
