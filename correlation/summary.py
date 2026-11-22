from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .clustering import Clusters
from .matcher import KNOWN, NO_MATCH, WEAK, Verdict

HEADLINES = {
    KNOWN: "Podobne do znanej kampanii",
    WEAK: "Słabe podobieństwo do znanych incydentów",
    NO_MATCH: "Brak dopasowania do znanych incydentów",
}


def _group_line(label: str, name: Optional[str], sizes: dict) -> str:
    return f"{label}: {name} ({sizes[name]} incydentów)" if name else f"{label}: brak"


def describe(path: Path, verdict: Verdict, clusters: Optional[Clusters] = None) -> str:
    lines: List[str] = [f"{path}: {HEADLINES[verdict.verdict]} (wynik {verdict.best_score:.2f})"]
    if clusters:
        campaigns = {k: len(v) for k, v in clusters.campaign_members().items()}
        families = {k: len(v) for k, v in clusters.family_members().items()}
        lines.append("  " + _group_line("kampania", verdict.campaign, campaigns))
        lines.append("  " + _group_line("rodzina", verdict.family, families))
        if verdict.outdated_clusters:
            lines.append("  uwaga: baza zmieniła się od klasteryzacji, uruchom cluster.py ponownie")
    for feature in verdict.top_features:
        lines.append(f"  cecha: {feature['feature']} (wkład {feature['contribution']:.2f})")
    for reason in verdict.reasons:
        lines.append(f"  - {reason}")
    if verdict.neighbours:
        nearest = verdict.neighbours[0]
        lines.append(f"  najbliższy: {nearest.incident.id} {nearest.incident.label or ''}".rstrip())
    return "\n".join(lines)
