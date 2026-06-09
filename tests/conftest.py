from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from veripresence.config import load_config
from veripresence.data.synthetic import generate_synthetic_enrollment


@pytest.fixture
def synthetic_project(tmp_path: Path):
    enrollment = tmp_path / "data" / "raw" / "enrollment"
    generate_synthetic_enrollment(
        enrollment,
        counts={"alex": 6, "blair": 6, "casey": 6},
        seed=123,
        image_size=96,
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        """
project:
  name: VeriPresence Test
  version: "1.0"
paths:
  enrollment_dir: data/raw/enrollment
  model_path: artifacts/model.joblib
  runs_dir: artifacts/runs
  database_path: runtime/test.db
preprocessing:
  face_size: 64
  min_face_size: 32
  detection_scale_factor: 1.1
  detection_min_neighbors: 5
  crop_margin: 0.15
  try_rotations: false
training:
  seed: 7
  validation_fraction: 0.33
  cv_folds: 2
  c_values: [1.0]
  augmentation_copies: 0
  minimum_samples_per_identity: 3
  confidence_threshold_bounds: [0.34, 0.80]
  margin_threshold_bounds: [0.0, 0.50]
inference:
  unknown_label: unknown
  attendance_cooldown_seconds: 60
  process_every_n_frames: 1
  camera_index: 0
quality:
  enabled: true
  min_brightness: 20.0
  max_brightness: 235.0
  min_contrast: 5.0
  min_sharpness: 2.0
api:
  host: 127.0.0.1
  port: 8000
  max_upload_bytes: 1000000
""".strip(),
        encoding="utf-8",
    )
    return load_config(config_path)
