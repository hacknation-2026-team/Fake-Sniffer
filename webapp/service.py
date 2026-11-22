from __future__ import annotations

import re
import shutil
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple

from PIL import Image

from correlation.base import Base, Incident
from correlation.clustering import Clusters, build_graph, cluster
from correlation.dataset import build_case_report, collect_case, run_input
from correlation.matcher import classify
from correlation.profile import Profile, profile_from_report
from correlation.similarity import compare

from .view import deepfake_rows, profile_view, step_view

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
TEXT_EXTENSIONS = {".txt", ".md"}
KEPT_ANALYSES = 20
PREVIEW_SIZE = 720
UNASSIGNED = "unassigned"


@dataclass
class Settings:
    base_path: Path = Path("base.json")
    clusters_path: Path = Path("clusters.json")
    data_dir: Path = Path("gui_data")
    models_dir: Optional[str] = None
    faces_dir: Optional[str] = None
    use_gpu: bool = False
    deepfake: bool = True


@dataclass
class Analysis:
    id: str
    name: str
    report: Dict[str, Any]
    profile: Profile
    preview: Optional[Path] = None
    added_as: Optional[str] = None


def _default_orchestrator(settings: Settings) -> Any:
    from feature_extractor.orchestrator import Orchestrator, OrchestratorConfig

    return Orchestrator(
        OrchestratorConfig(
            use_gpu=settings.use_gpu,
            reference_faces_dir=settings.faces_dir,
            models_dir=settings.models_dir,
            deepfake=settings.deepfake,
        )
    )


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w.\- ()]", "_", Path(name).name).strip(" .")
    return cleaned or "plik"


def _unique(folder: Path, name: str) -> Path:
    target = folder / name
    counter = 1
    while target.exists():
        target = folder / f"{Path(name).stem}_{counter}{Path(name).suffix}"
        counter += 1
    return target


class Workspace:
    """Baza, klastry i analizy używane przez interfejs. Pliki są czytane przy każdym zapytaniu."""

    def __init__(self, settings: Settings, orchestrator_factory: Optional[Callable[[], Any]] = None):
        self.settings = settings
        self._factory = orchestrator_factory or (lambda: _default_orchestrator(settings))
        self._orchestrator: Any = None
        self._lock = threading.RLock()
        self._run_lock = threading.Lock()
        self._analyses: "OrderedDict[str, Analysis]" = OrderedDict()

    def _base(self) -> Base:
        return Base(self.settings.base_path)

    def _clusters(self, base: Base, create: bool = True) -> Optional[Clusters]:
        if not base.incidents:
            return None
        path = self.settings.clusters_path
        if path.exists():
            try:
                return Clusters.load(path)
            except (OSError, ValueError, KeyError):
                pass
        if not create:
            return None
        result = cluster(base.incidents)
        result.save(path)
        return result

    def _get_orchestrator(self) -> Any:
        if self._orchestrator is None:
            self._orchestrator = self._factory()
        return self._orchestrator

    def _save_uploads(self, folder: Path, uploads: List[Tuple[str, BinaryIO]]) -> List[str]:
        folder.mkdir(parents=True, exist_ok=True)
        names = []
        for name, source in uploads:
            target = _unique(folder, _safe_name(name))
            with open(target, "wb") as out:
                shutil.copyfileobj(source, out)
            names.append(target.name)
        return names

    def _collect_steps(self, url: str, folder: Path) -> List[Dict[str, Any]]:
        orchestrator = self._get_orchestrator()
        steps: List[Dict[str, Any]] = []
        if url:
            steps += run_input(orchestrator, "url", url=url)

        case = collect_case(folder)
        if not case.is_empty:
            keyframes = self.settings.data_dir / "keyframes"
            steps += build_case_report(case, orchestrator, keyframes_root=keyframes)["steps"]

        for path in sorted(folder.glob("*")):
            suffix = path.suffix.lower()
            if suffix in AUDIO_EXTENSIONS:
                steps += run_input(orchestrator, "audio", audio=str(path))
            elif suffix in TEXT_EXTENSIONS:
                text = path.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    steps += run_input(orchestrator, "text", text=text)
        return steps

    def _make_preview(self, folder: Path, analysis_id: str) -> Optional[Path]:
        images = collect_case(folder).images
        if not images:
            return None
        target = self.settings.data_dir / "previews" / f"{analysis_id}.jpg"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(images[0]) as image:
                image = image.convert("RGB")
                image.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
                image.save(target, "JPEG", quality=85)
        except OSError:
            return None
        return target

    def analyze(self, url: str, uploads: List[Tuple[str, BinaryIO]]) -> Dict[str, Any]:
        """Uruchamia ekstraktory na linku i plikach, potem porównuje wynik z bazą."""
        url = (url or "").strip()
        uploads = [(name, source) for name, source in uploads if name]
        if not url and not uploads:
            raise ValueError("Podaj link albo dodaj plik.")

        analysis_id = uuid.uuid4().hex[:10]
        folder = self.settings.data_dir / "uploads" / analysis_id
        names = self._save_uploads(folder, uploads)

        with self._run_lock:
            steps = self._collect_steps(url, folder)

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "inputs": {"url": url, "files": names},
            "steps": steps,
        }
        analysis = Analysis(
            id=analysis_id,
            name=url or ", ".join(names),
            report=report,
            profile=profile_from_report(report),
            preview=self._make_preview(folder, analysis_id),
        )
        with self._lock:
            self._analyses[analysis_id] = analysis
            while len(self._analyses) > KEPT_ANALYSES:
                self._analyses.popitem(last=False)
            return self._analysis_view(analysis)

    def _analysis_view(self, analysis: Analysis) -> Dict[str, Any]:
        base = self._base()
        clusters = self._clusters(base)
        verdict = classify(analysis.profile, base, clusters).to_dict() if not analysis.profile.is_empty else None

        view: Dict[str, Any] = {
            "id": analysis.id,
            "name": analysis.name,
            "inputs": analysis.report["inputs"],
            "steps": [step_view(s) for s in analysis.report["steps"]],
            "deepfake": deepfake_rows(analysis.report["steps"]),
            "deepfake_enabled": self.settings.deepfake,
            "has_features": not analysis.profile.is_empty,
            "added_as": analysis.added_as,
            "verdict": verdict,
        }
        if verdict and clusters:
            campaigns, families = clusters.campaign_members(), clusters.family_members()
            view["campaign_size"] = len(campaigns.get(verdict["campaign"], []))
            view["family_size"] = len(families.get(verdict["family"], []))
        return view

    def add_to_base(self, analysis_id: str, label: str = "") -> Dict[str, Any]:
        with self._lock:
            analysis = self._analyses.get(analysis_id)
            if analysis is None:
                raise LookupError("Nie ma takiej analizy (mogła wygasnąć), uruchom ją jeszcze raz.")
            if analysis.profile.is_empty:
                raise ValueError("Analiza nie dała cech do zapisania w bazie.")
            if analysis.added_as:
                raise ValueError(f"Ta analiza jest już w bazie jako {analysis.added_as}.")

            base = self._base()
            incident = base.add(analysis.profile, analysis.name, label.strip() or analysis.name)
            if analysis.preview and analysis.preview.exists():
                target = self.settings.data_dir / "previews" / f"{incident.id}.jpg"
                shutil.copyfile(analysis.preview, target)
                incident.preview = str(target)
            base.save()
            analysis.added_as = incident.id
            return {"id": incident.id}

    def recluster(self) -> Dict[str, int]:
        with self._lock:
            base = self._base()
            if not base.incidents:
                raise ValueError("Baza jest pusta.")
            result = cluster(base.incidents)
            result.save(self.settings.clusters_path)
            return {
                "campaigns": len(result.campaign_members()),
                "families": len(result.family_members()),
            }

    def campaigns_view(self) -> Dict[str, Any]:
        with self._lock:
            base = self._base()
            clusters = self._clusters(base)
            if clusters is None:
                return {"campaigns": [], "total": 0, "outdated": False}

            graph = build_graph(base.incidents, clusters.threshold)
            groups: Dict[str, List[Incident]] = {}
            for incident in base.incidents:
                groups.setdefault(clusters.campaigns.get(incident.id, UNASSIGNED), []).append(incident)

            campaigns = []
            for name, members in groups.items():
                ids = {i.id for i in members}
                edges = [
                    {"a": a, "b": b, "score": d["weight"], "hard": d["hard"], "duplicate": d["duplicate"]}
                    for a, b, d in graph.edges(data=True)
                    if a in ids and b in ids
                ]
                nodes = [
                    {
                        "id": i.id,
                        "label": i.label or i.id,
                        "family": clusters.families.get(i.id),
                        "preview": bool(i.preview),
                    }
                    for i in members
                ]
                campaigns.append({"id": name, "size": len(nodes), "nodes": nodes, "edges": edges})

            campaigns.sort(key=lambda c: (c["id"] == UNASSIGNED, -c["size"], c["id"]))
            return {
                "campaigns": campaigns,
                "total": len(base.incidents),
                "outdated": UNASSIGNED in groups,
            }

    def database_view(self) -> List[Dict[str, Any]]:
        with self._lock:
            base = self._base()
            clusters = self._clusters(base)
            campaigns = clusters.campaigns if clusters else {}
            families = clusters.families if clusters else {}
            return [
                {
                    "id": i.id,
                    "label": i.label or i.id,
                    "source": i.source,
                    "added_at": i.added_at,
                    "campaign": campaigns.get(i.id),
                    "family": families.get(i.id),
                    "preview": bool(i.preview),
                }
                for i in base.incidents
            ]

    def _find(self, base: Base, incident_id: str) -> Incident:
        incident = base.get(incident_id)
        if incident is None:
            raise LookupError(f"Nie ma incydentu {incident_id}.")
        return incident

    def incident_view(self, incident_id: str) -> Dict[str, Any]:
        with self._lock:
            base = self._base()
            incident = self._find(base, incident_id)
            clusters = self._clusters(base, create=False)
            return {
                "id": incident.id,
                "label": incident.label or incident.id,
                "source": incident.source,
                "added_at": incident.added_at,
                "campaign": clusters.campaigns.get(incident.id) if clusters else None,
                "family": clusters.families.get(incident.id) if clusters else None,
                "preview": bool(incident.preview),
                "profile": profile_view(incident.profile),
            }

    def preview_path(self, incident_id: str) -> Optional[Path]:
        with self._lock:
            incident = self._find(self._base(), incident_id)
        path = Path(incident.preview) if incident.preview else None
        return path if path and path.is_file() else None

    def compare(self, id_a: str, id_b: str) -> Dict[str, Any]:
        with self._lock:
            base = self._base()
            a, b = self._find(base, id_a), self._find(base, id_b)
        result = compare(a.profile, b.profile)
        return {
            "a": {"id": a.id, "label": a.label or a.id},
            "b": {"id": b.id, "label": b.label or b.id},
            "score": result.score,
            "modalities": result.modalities,
            "contributions": result.contributions,
            "hard": result.hard,
            "duplicate": result.duplicate,
            "reasons": result.reasons,
        }

    def delete(self, incident_id: str) -> None:
        with self._lock:
            base = self._base()
            incident = self._find(base, incident_id)
            base.incidents = [i for i in base.incidents if i.id != incident_id]
            base.save()
            previews = (self.settings.data_dir / "previews").resolve()
            if incident.preview:
                path = Path(incident.preview).resolve()
                if path.is_file() and path.parent == previews:
                    path.unlink()
