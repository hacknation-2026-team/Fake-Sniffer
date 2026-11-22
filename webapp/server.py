from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .service import Settings, Workspace

STATIC = Path(__file__).parent / "static"


class AddRequest(BaseModel):
    label: str = ""


def _call(action: Callable[[], Any]) -> Any:
    try:
        return action()
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0]))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def create_app(
    settings: Optional[Settings] = None,
    orchestrator_factory: Optional[Callable[[], Any]] = None,
) -> FastAPI:
    workspace = Workspace(settings or Settings(), orchestrator_factory)
    app = FastAPI(title="Fake Sniffer")
    app.state.workspace = workspace

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.post("/api/analyze")
    def analyze(url: str = Form(""), files: List[UploadFile] = File(default=[])):
        uploads = [(f.filename, f.file) for f in files if f.filename]
        return _call(lambda: workspace.analyze(url, uploads))

    @app.post("/api/analyses/{analysis_id}/add")
    def add(analysis_id: str, body: AddRequest):
        return _call(lambda: workspace.add_to_base(analysis_id, body.label))

    @app.get("/api/campaigns")
    def campaigns():
        return _call(workspace.campaigns_view)

    @app.post("/api/cluster")
    def recluster():
        return _call(workspace.recluster)

    @app.get("/api/database")
    def database():
        return _call(workspace.database_view)

    @app.get("/api/incidents/{incident_id}")
    def incident(incident_id: str):
        return _call(lambda: workspace.incident_view(incident_id))

    @app.get("/api/incidents/{incident_id}/preview")
    def preview(incident_id: str):
        path = _call(lambda: workspace.preview_path(incident_id))
        if path is None:
            raise HTTPException(status_code=404, detail="Brak podglądu.")
        return FileResponse(path)

    @app.delete("/api/incidents/{incident_id}")
    def delete(incident_id: str):
        _call(lambda: workspace.delete(incident_id))
        return {"deleted": incident_id}

    @app.get("/api/compare")
    def compare(a: str, b: str):
        return _call(lambda: workspace.compare(a, b))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
