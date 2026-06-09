from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


@dataclass
class ModelArtifact:
    estimator: Pipeline
    face_size: int
    confidence_threshold: float
    margin_threshold: float
    metadata: dict[str, Any]
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        estimator: Pipeline,
        face_size: int,
        confidence_threshold: float,
        margin_threshold: float,
        metadata: dict[str, Any],
    ) -> ModelArtifact:
        enriched_metadata = {
            **metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return cls(
            estimator=estimator,
            face_size=face_size,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            metadata=enriched_metadata,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> ModelArtifact:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Model artifact not found: {source}")
        artifact = joblib.load(source)
        if not isinstance(artifact, cls):
            raise TypeError(f"Unexpected artifact type: {type(artifact)!r}")
        if artifact.schema_version != 1:
            raise ValueError(f"Unsupported artifact schema: {artifact.schema_version}")
        return artifact
