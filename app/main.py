from __future__ import annotations

import csv
import hmac
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response

from app.schemas import (
    AttendanceResponse,
    AttendanceSummaryResponse,
    HealthResponse,
    ModelResponse,
    RecognitionBatchResponse,
    RecognitionResponse,
    RetentionResponse,
)
from veripresence.config import AppConfig, load_config
from veripresence.inference.engine import RecognitionEngine
from veripresence.storage.repository import AttendanceRepository
from veripresence.utils.logging import configure_logging

configure_logging()


@asynccontextmanager
async def lifespan(application: FastAPI):
    config_path = Path(os.getenv("VERIPRESENCE_CONFIG", "configs/default.yaml"))
    config = load_config(config_path)
    application.state.config = config
    application.state.repository = AttendanceRepository(config.paths.database_path)
    try:
        application.state.engine = _load_engine(config)
        application.state.model_error = None
    except (FileNotFoundError, TypeError, ValueError) as error:
        application.state.engine = None
        application.state.model_error = str(error)
    yield


app = FastAPI(
    title="VeriPresence API",
    version="1.0.0",
    description="Face attendance and unknown-person recognition API.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    config: AppConfig = app.state.config
    loaded = app.state.engine is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        project=config.project.name,
        version=config.project.version,
    )


@app.get("/v1/model", response_model=ModelResponse)
def model_metadata(request: Request) -> ModelResponse:
    _check_api_key(request)
    engine = _require_engine()
    return ModelResponse(metadata=engine.metadata)


@app.get("/v1/identities")
def identities(request: Request) -> dict[str, object]:
    _check_api_key(request)
    metadata = _require_engine().metadata
    return {
        "identities": metadata.get("classes", []),
        "image_counts": metadata.get("class_counts", {}),
    }


@app.post("/v1/model/reload", response_model=ModelResponse)
def reload_model(request: Request) -> ModelResponse:
    _check_api_key(request, required=True)
    config: AppConfig = app.state.config
    try:
        app.state.engine = _load_engine(config)
        app.state.model_error = None
    except (FileNotFoundError, TypeError, ValueError) as error:
        app.state.engine = None
        app.state.model_error = str(error)
        raise HTTPException(status_code=503, detail=str(error)) from error
    return ModelResponse(metadata=app.state.engine.metadata)


@app.post("/v1/recognitions", response_model=RecognitionBatchResponse)
async def recognize(
    request: Request,
    source: str = Query(default="api", max_length=120),
    fallback_to_full_image: bool = Query(default=False),
) -> RecognitionBatchResponse:
    _check_api_key(request)
    config: AppConfig = app.state.config
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Send raw image bytes with an image/* Content-Type header.",
        )
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > config.api.max_upload_bytes:
        raise HTTPException(status_code=413, detail="The uploaded image is too large.")
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(payload) > config.api.max_upload_bytes:
        raise HTTPException(status_code=413, detail="The uploaded image is too large.")

    engine = _require_engine()
    repository: AttendanceRepository = app.state.repository
    try:
        recognitions = engine.recognize_bytes(payload, fallback_to_full_image)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    responses = []
    for recognition in recognitions:
        recorded = repository.record(
            identity=recognition.identity if recognition.accepted else None,
            accepted=recognition.accepted,
            confidence=recognition.confidence,
            margin=recognition.margin,
            source=source,
            cooldown_seconds=config.inference.attendance_cooldown_seconds,
        )
        responses.append(
            RecognitionResponse(**recognition.to_dict(), event_recorded=recorded)
        )
    return RecognitionBatchResponse(faces=responses)


@app.get("/v1/attendance", response_model=AttendanceResponse)
def attendance(
    request: Request,
    event_date: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    event_type: str | None = Query(default=None, pattern=r"^(attendance|unknown)$"),
    identity: str | None = Query(default=None, max_length=120),
    source: str | None = Query(default=None, max_length=120),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> AttendanceResponse:
    _check_api_key(request)
    repository: AttendanceRepository = app.state.repository
    return AttendanceResponse(
        events=repository.list_events(
            event_date=event_date,
            event_type=event_type,
            identity=identity,
            source=source,
            limit=limit,
            offset=offset,
        )
    )


@app.get("/v1/attendance/summary", response_model=AttendanceSummaryResponse)
def attendance_summary(
    request: Request,
    event_date: str = Query(pattern=r"^\d{4}-\d{2}-\d{2}$"),
    source: str | None = Query(default=None, max_length=120),
) -> AttendanceSummaryResponse:
    _check_api_key(request)
    repository: AttendanceRepository = app.state.repository
    return AttendanceSummaryResponse(**repository.daily_summary(event_date, source))


@app.get("/v1/attendance/export")
def export_attendance(
    request: Request,
    event_date: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    event_type: str | None = Query(default=None, pattern=r"^(attendance|unknown)$"),
    identity: str | None = Query(default=None, max_length=120),
    source: str | None = Query(default=None, max_length=120),
) -> Response:
    _check_api_key(request)
    repository: AttendanceRepository = app.state.repository
    events = repository.list_events(
        event_date=event_date,
        event_type=event_type,
        identity=identity,
        source=source,
        limit=100_000,
    )
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "id",
            "identity",
            "event_type",
            "confidence",
            "margin",
            "source",
            "captured_at",
        ],
    )
    writer.writeheader()
    writer.writerows(events)
    filename = f"attendance-{event_date or 'all'}.csv"
    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/v1/attendance", response_model=RetentionResponse)
def delete_old_events(
    request: Request,
    before: str = Query(pattern=r"^\d{4}-\d{2}-\d{2}$"),
    confirm: bool = Query(default=False),
) -> RetentionResponse:
    _check_api_key(request, required=True)
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to delete historical events.",
        )
    repository: AttendanceRepository = app.state.repository
    deleted = repository.delete_before(before)
    return RetentionResponse(deleted_events=deleted, cutoff_date=before)


def _require_engine() -> RecognitionEngine:
    engine = app.state.engine
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not available: {app.state.model_error}",
        )
    return engine


def _load_engine(config: AppConfig) -> RecognitionEngine:
    return RecognitionEngine(
        model_path=config.paths.model_path,
        preprocessing_config=config.preprocessing,
        quality_config=config.quality,
        unknown_label=config.inference.unknown_label,
    )


def _check_api_key(request: Request, required: bool = False) -> None:
    expected = os.getenv("VERIPRESENCE_API_KEY")
    if not expected:
        if required:
            raise HTTPException(
                status_code=503,
                detail="Set VERIPRESENCE_API_KEY before using administrative endpoints.",
            )
        return
    supplied = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")
