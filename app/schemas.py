from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    project: str
    version: str


class QualityResponse(BaseModel):
    acceptable: bool
    score: float = Field(ge=0.0, le=1.0)
    brightness: float
    contrast: float
    sharpness: float
    issues: list[str]


class RecognitionResponse(BaseModel):
    identity: str
    accepted: bool
    confidence: float = Field(ge=0.0, le=1.0)
    margin: float = Field(ge=0.0, le=1.0)
    bbox: list[int]
    quality: QualityResponse
    reason: str | None
    event_recorded: bool


class RecognitionBatchResponse(BaseModel):
    faces: list[RecognitionResponse]


class ModelResponse(BaseModel):
    metadata: dict[str, Any]


class AttendanceResponse(BaseModel):
    events: list[dict[str, Any]]


class AttendanceSummaryResponse(BaseModel):
    date: str
    source: str | None
    present_count: int
    unknown_events: int
    identities: list[dict[str, Any]]


class RetentionResponse(BaseModel):
    deleted_events: int
    cutoff_date: str
