from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str


@dataclass(frozen=True)
class PathsConfig:
    enrollment_dir: Path
    model_path: Path
    runs_dir: Path
    database_path: Path


@dataclass(frozen=True)
class PreprocessingConfig:
    face_size: int = 128
    min_face_size: int = 48
    detection_scale_factor: float = 1.1
    detection_min_neighbors: int = 5
    crop_margin: float = 0.18
    try_rotations: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    validation_fraction: float = 0.25
    cv_folds: int = 3
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0)
    augmentation_copies: int = 1
    minimum_samples_per_identity: int = 3
    confidence_threshold_bounds: tuple[float, float] = (0.55, 0.90)
    margin_threshold_bounds: tuple[float, float] = (0.10, 0.60)


@dataclass(frozen=True)
class InferenceConfig:
    unknown_label: str = "unknown"
    attendance_cooldown_seconds: int = 300
    process_every_n_frames: int = 3
    camera_index: int = 0


@dataclass(frozen=True)
class QualityConfig:
    enabled: bool = True
    min_brightness: float = 30.0
    max_brightness: float = 225.0
    min_contrast: float = 15.0
    min_sharpness: float = 15.0


@dataclass(frozen=True)
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_bytes: int = 8_000_000


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    paths: PathsConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    inference: InferenceConfig
    quality: QualityConfig
    api: ApiConfig
    root_dir: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return _stringify_paths(payload)


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    root_dir = config_path.parent.parent
    path_values = raw.get("paths", {})

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (root_dir / candidate).resolve()

    training_values = raw.get("training", {})
    training_values["c_values"] = tuple(training_values.get("c_values", (0.1, 1.0, 10.0)))
    training_values["confidence_threshold_bounds"] = tuple(
        training_values.get("confidence_threshold_bounds", (0.55, 0.90))
    )
    training_values["margin_threshold_bounds"] = tuple(
        training_values.get("margin_threshold_bounds", (0.10, 0.60))
    )

    return AppConfig(
        project=ProjectConfig(**raw.get("project", {"name": "VeriPresence", "version": "1.0.0"})),
        paths=PathsConfig(
            enrollment_dir=resolve(path_values.get("enrollment_dir", "data/raw/enrollment")),
            model_path=resolve(path_values.get("model_path", "artifacts/models/model.joblib")),
            runs_dir=resolve(path_values.get("runs_dir", "artifacts/runs")),
            database_path=resolve(path_values.get("database_path", "runtime/veripresence.db")),
        ),
        preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
        training=TrainingConfig(**training_values),
        inference=InferenceConfig(**raw.get("inference", {})),
        quality=QualityConfig(**raw.get("quality", {})),
        api=ApiConfig(**raw.get("api", {})),
        root_dir=root_dir,
    )


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_paths(item) for item in value]
    return value
