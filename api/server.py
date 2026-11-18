"""Local GAZERA API (FastAPI).

Run locally so research data never leaves the machine:

    uvicorn api.server:app --reload --port 8000

Endpoints
---------
GET  /api/health    service metadata and licence notice
GET  /api/sample    synthetic demo dataset (TSV) for quick tests
POST /api/analyze   multipart upload + optional mapping/config JSON -> results
POST /api/export    same input, returns xlsx / csv / json attachment

This server is a thin wrapper around ``gazera_core``; the analysis rules live in
the package so the web workbench and the API cannot drift apart in definition.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gazera_core as gz  # noqa: E402
from gazera_core import AnalysisConfig, IVTConfig  # noqa: E402

APP_VERSION = gz.__version__

NOTICE = {
    "license": "MIT License · Copyright (c) 2026 FEI · HEL Lab, Gyeongsang National University",
    "affiliation": (
        "GAZERA is an independent open-source project developed by FEI · HEL Lab "
        "(Gyeongsang National University) under the MIT License. It is not affiliated with, "
        "endorsed by, or sponsored by Tobii AB. All product names and trademarks belong to "
        "their respective owners."
    ),
    "privacy": "Files are processed in memory on this machine and are not uploaded anywhere else.",
}

app = FastAPI(
    title="GAZERA Core API",
    version=APP_VERSION,
    description="Clean-room, vendor-neutral eye-tracking analytics. MIT licensed.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:8000", "null"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _config_from_payload(config_payload: str | None) -> AnalysisConfig:
    if not config_payload:
        return AnalysisConfig()
    try:
        raw = json.loads(config_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"config must be JSON: {exc}") from exc
    ivt_raw = raw.get("ivt") or {}
    ivt_fields = {k: v for k, v in ivt_raw.items() if k in IVTConfig.__dataclass_fields__}
    ivt = IVTConfig(**ivt_fields)
    return AnalysisConfig(
        ivt=ivt,
        aois=raw.get("aois"),
        dvs=raw.get("dvs"),
        group=raw.get("group", "condition"),
        paired=raw.get("paired", "condition"),
        clean=bool(raw.get("clean", True)),
        interpolate_gap_ms=float(raw.get("interpolate_gap_ms", 40.0)),
        screen=tuple(raw["screen"]) if raw.get("screen") else None,
        decimal_comma=bool(raw.get("decimal_comma", False)),
        include_segments=bool(raw.get("include_segments", False)),
        stimulus=raw.get("stimulus_path") or raw.get("stimulus"),
        exclude_rgb=raw.get("exclude_rgb"),
        rgb_tolerance=float(raw.get("rgb_tolerance", 15.0)),
        stimulus_offset_ms=float(raw.get("stimulus_offset_ms", 0.0)),
        rgb_frame_bucket_ms=float(raw.get("rgb_frame_bucket_ms", 200.0)),
    )


async def _run(file: UploadFile, mapping: str | None, config: str | None, stimulus: UploadFile | None = None):
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file")
    mapping_dict: dict[str, str] | None = None
    if mapping:
        try:
            mapping_dict = json.loads(mapping)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"mapping must be JSON: {exc}") from exc
    cfg = _config_from_payload(config)
    try:
        data = gz.load_gaze_bytes(payload, mapping=mapping_dict, filename=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if stimulus is None or not stimulus.filename:
        return gz.analyze(data, cfg)

    # the stimulus is written to a temporary file only for this request and
    # removed again before the response is returned
    blob = await stimulus.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Empty stimulus file")
    suffix = Path(stimulus.filename).suffix.lower() or ".png"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"stimulus{suffix}"
        path.write_bytes(blob)
        cfg.stimulus = path
        try:
            result = gz.analyze(data, cfg)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if result.rgb_filter is not None:
            result.rgb_filter.stimulus = stimulus.filename
        return result


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "name": "GAZERA Core",
        "version": APP_VERSION,
        "license": "MIT",
        "features": ["ivt", "aoi", "statistics", "rgb_exclusion"],
        "notices": NOTICE,
    }


@app.get("/api/sample")
def sample(subjects: int = Query(6, ge=1, le=40), duration: float = Query(6.0, gt=0, le=120)) -> Response:
    text = gz.generate_dataset(gz.SyntheticConfig(n_subjects=subjects, duration_s=duration))
    return Response(
        content=text.encode("utf-8"),
        media_type="text/tab-separated-values; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="gazera_synthetic_demo.tsv"'},
    )


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(..., description="CSV/TSV gaze export"),
    mapping: str | None = Form(None, description="JSON object: canonical field -> column"),
    config: str | None = Form(None, description="JSON object with ivt/aois/dvs/rgb settings"),
    stimulus: UploadFile | None = File(
        None, description="optional stimulus image or video for the RGB exclusion filter"
    ),
) -> JSONResponse:
    result = await _run(file, mapping, config, stimulus)
    return JSONResponse(result.to_dict())


@app.post("/api/export")
async def export(
    file: UploadFile = File(...),
    fmt: str = Query("xlsx", pattern="^(xlsx|csv|json)$"),
    mapping: str | None = Form(None),
    config: str | None = Form(None),
    stimulus: UploadFile | None = File(None, description="optional stimulus image or video"),
) -> Response:
    result = await _run(file, mapping, config, stimulus)
    stem = Path(file.filename or "gazera").stem
    if fmt == "json":
        body = json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str)
        return Response(
            body.encode("utf-8"),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{stem}_gazera.json"'},
        )
    if fmt == "csv":
        buffer = io.StringIO()
        result.subject_table.to_csv(buffer, index=False)
        return Response(
            ("\ufeff" + buffer.getvalue()).encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{stem}_participants.csv"'},
        )
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / f"{stem}_gazera.xlsx"
        gz.export_excel(
            target,
            result.sheets(),
            metadata={
                "GAZERA version": APP_VERSION,
                "Source file": file.filename,
                "License": NOTICE["license"],
                "Notice": NOTICE["affiliation"],
            },
        )
        body = target.read_bytes()
    return Response(
        body,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{stem}_gazera.xlsx"'},
    )


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "GAZERA Core API",
        "version": APP_VERSION,
        "endpoints": ["/api/health", "/api/sample", "/api/analyze", "/api/export"],
        "rgb_exclusion": {
            "form_field": "stimulus",
            "config": ["exclude_rgb", "rgb_tolerance", "stimulus_offset_ms", "rgb_frame_bucket_ms"],
            "stimuli": ".png .jpg .jpeg .bmp .gif .webp | .mp4 .mov .avi .webm .mkv",
        },
        "notices": NOTICE,
    }
