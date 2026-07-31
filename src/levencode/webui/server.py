"""Training dashboard server: serves the static UI + JSON APIs over runs/."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..util.io import read_json, read_jsonl

_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
STATIC_DIR = Path(__file__).parent / "static"


def _safe(name: str) -> str:
    if not _NAME_RE.match(name):
        raise HTTPException(400, f"bad name: {name!r}")
    return name


def create_app(runs_dir: str | Path) -> FastAPI:
    runs_root = Path(runs_dir).resolve()
    app = FastAPI(title="levencode dashboard")

    @app.get("/api/experiments")
    def experiments() -> JSONResponse:
        out = []
        if runs_root.exists():
            for exp_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
                stages = []
                for stage_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
                    state = read_json(stage_dir / "state.json", default=None)
                    if state is None:
                        continue
                    bench_dir = stage_dir / "bench"
                    stages.append(
                        {
                            "stage": stage_dir.name,
                            "state": state,
                            "bench_names": sorted(p.stem for p in bench_dir.glob("*.json"))
                            if bench_dir.exists()
                            else [],
                            "has_samples": (stage_dir / "samples").exists(),
                        }
                    )
                if stages:
                    out.append({"experiment": exp_dir.name, "stages": stages})
        return JSONResponse({"experiments": out})

    def _stage_dir(exp: str, stage: str) -> Path:
        d = runs_root / _safe(exp) / _safe(stage)
        if not d.exists():
            raise HTTPException(404, f"no run {exp}/{stage}")
        return d

    @app.get("/api/run/{exp}/{stage}/state")
    def state(exp: str, stage: str) -> JSONResponse:
        return JSONResponse(read_json(_stage_dir(exp, stage) / "state.json", default={}))

    @app.get("/api/run/{exp}/{stage}/metrics")
    def metrics(exp: str, stage: str, max_points: int = 1500) -> JSONResponse:
        rows = read_jsonl(_stage_dir(exp, stage) / "metrics.jsonl", max_points=max_points)
        return JSONResponse({"rows": rows})

    @app.get("/api/run/{exp}/{stage}/bench")
    def bench(exp: str, stage: str) -> JSONResponse:
        d = _stage_dir(exp, stage) / "bench"
        out = {}
        if d.exists():
            for p in sorted(d.glob("*.json")):
                out[p.stem] = read_json(p, default={})
        return JSONResponse(out)

    @app.get("/api/run/{exp}/{stage}/samples")
    def samples(exp: str, stage: str) -> JSONResponse:
        d = _stage_dir(exp, stage) / "samples"
        out = {}
        if d.exists():
            for p in sorted(d.glob("*.json")):
                out[p.stem] = read_json(p, default=[])
        return JSONResponse(out)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app
