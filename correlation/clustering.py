from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from .base import Incident
from .similarity import THRESHOLD, TIGHT_THRESHOLD, compare


def _groups(assignment: Dict[str, str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for incident_id, name in assignment.items():
        groups[name].append(incident_id)
    return {name: sorted(members) for name, members in sorted(groups.items())}


@dataclass
class Clusters:
    """Wynik klasteryzacji: kampanie i rodziny kampanii, osobno od bazy."""

    campaigns: Dict[str, str]
    families: Dict[str, str]
    threshold: float = THRESHOLD
    tight: float = TIGHT_THRESHOLD
    created_at: str = field(default="", compare=False)

    def campaign_members(self) -> Dict[str, List[str]]:
        return _groups(self.campaigns)

    def family_members(self) -> Dict[str, List[str]]:
        return _groups(self.families)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "threshold": self.threshold,
            "tight": self.tight,
            "campaigns": self.campaign_members(),
            "families": self.family_members(),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Clusters":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        def flatten(groups: Dict[str, List[str]]) -> Dict[str, str]:
            return {incident_id: name for name, members in groups.items() for incident_id in members}

        return cls(
            flatten(raw["campaigns"]),
            flatten(raw["families"]),
            raw.get("threshold", THRESHOLD),
            raw.get("tight", TIGHT_THRESHOLD),
            raw.get("created_at", ""),
        )


def build_graph(incidents: List[Incident], threshold: float = THRESHOLD) -> nx.Graph:
    """Węzły to incydenty, krawędź powstaje przy wyniku >= progu lub twardym śladzie infrastruktury."""
    graph = nx.Graph()
    graph.add_nodes_from(i.id for i in incidents)
    for a, b in combinations(incidents, 2):
        result = compare(a.profile, b.profile)
        if result.linked(threshold):
            graph.add_edge(
                a.id,
                b.id,
                weight=max(result.score, 0.01),
                hard=bool(result.hard),
                duplicate=result.duplicate,
            )
    return graph


def _name(groups: List[List[str]], prefix: str) -> Dict[str, str]:
    ordered = sorted(groups, key=lambda g: (-len(g), min(g)))
    return {node: f"{prefix}-{n}" for n, group in enumerate(ordered, 1) for node in group}


def cluster(
    incidents: List[Incident],
    threshold: float = THRESHOLD,
    tight: float = TIGHT_THRESHOLD,
    seed: int = 0,
) -> Clusters:
    """
    Kampanie to spójne składowe grafu z wysokim progiem, rodziny kampanii
    to społeczności Louvain na grafie z progiem zwykłym.
    """
    graph = build_graph(incidents, threshold)

    strict = nx.Graph()
    strict.add_nodes_from(graph.nodes)
    strict.add_edges_from(
        (u, v) for u, v, d in graph.edges(data=True) if d["hard"] or d["weight"] >= tight
    )
    campaigns = [list(c) for c in nx.connected_components(strict)]
    families = [list(c) for c in nx.community.louvain_communities(graph, weight="weight", seed=seed)]
    return Clusters(
        _name(campaigns, "campaign"),
        _name(families, "family"),
        threshold,
        tight,
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
