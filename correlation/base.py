from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .profile import Profile

BASE_VERSION = 1


@dataclass
class Incident:
    id: str
    added_at: str
    source: str
    profile: Profile
    label: Optional[str] = None
    preview: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Incident":
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        return cls(**{**values, "profile": Profile.from_dict(data["profile"])})


class Base:
    """Baza znanych incydentów w jednym pliku JSON."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.incidents: List[Incident] = []
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.incidents = [Incident.from_dict(item) for item in raw.get("incidents", [])]

    def __len__(self) -> int:
        return len(self.incidents)

    def get(self, incident_id: str) -> Optional[Incident]:
        return next((i for i in self.incidents if i.id == incident_id), None)

    def next_id(self) -> str:
        numbers = [int(i.id.rsplit("-", 1)[1]) for i in self.incidents]
        return f"inc-{max(numbers, default=0) + 1}"

    def add(
        self,
        profile: Profile,
        source: str,
        label: Optional[str] = None,
        preview: Optional[str] = None,
    ) -> Incident:
        incident = Incident(
            id=self.next_id(),
            added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=source,
            profile=profile,
            label=label,
            preview=preview,
        )
        self.incidents.append(incident)
        return incident

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": BASE_VERSION, "incidents": [i.to_dict() for i in self.incidents]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return self.path
