# VeriPresence

> Recognize presence. Reject uncertainty.

[![CI](https://github.com/a6rahamjr/VeriPresence/actions/workflows/ci.yml/badge.svg)](https://github.com/a6rahamjr/VeriPresence/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

VeriPresence is a reproducible computer-vision platform for enrolled-person attendance and
unknown-person detection. It replaces a webcam proof of concept with a tested ML pipeline,
versioned artifacts, SQLite events, FastAPI inference, and camera/video operation.

## Highlights

- Multi-image identity learning instead of filename-by-filename matching
- Orientation-aware face preprocessing with deterministic augmentation
- HOG, LBP, and intensity features with a tuned class-balanced SVM
- Confidence plus class-margin rejection for uncertain identities
- Brightness, contrast, and sharpness checks for unusable face crops
- Reproducible metrics, dataset fingerprints, and Joblib model export
- Filtered attendance history, daily summaries, and CSV exports
- REST API, webcam/video runner, SQLite persistence, and optional webhooks
- Optional API-key protection, upload limits, retention cleanup, and model reload
- Privacy-safe synthetic data, Docker packaging, tests, and CI

## Repository Structure

```text
veripresence/
|-- app/
|   |-- main.py                 # FastAPI routes and lifecycle
|   `-- schemas.py              # HTTP response contracts
|-- configs/
|   `-- default.yaml            # Paths, preprocessing, training, runtime settings
|-- data/
|   `-- raw/enrollment/         # Private identity folders; ignored by Git
|-- docs/
|   |-- PRD.md
|   |-- COMPARISON.md
|   `-- MODEL_CARD.md
|-- src/veripresence/
|   |-- alerts/                 # Optional webhook integration
|   |-- data/                   # Loading, migration, preprocessing, demo generation
|   |-- evaluation/             # Metrics and feature ablation
|   |-- inference/              # Model engine and camera/video runner
|   |-- models/                 # Features, SVM training, artifact schema
|   |-- storage/                # SQLite attendance repository
|   |-- training/               # Reproducible training workflow and CLI
|   `-- utils/                  # JSON logging and deterministic seeds
|-- tests/                      # Dataset, training, and inference tests
|-- .github/workflows/ci.yml
|-- Dockerfile
|-- pyproject.toml
|-- requirements.txt
`-- requirements-dev.txt
```

## Quick Start

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e ".[dev]"
```

### Option A: Import the 2022 images

From the new repository:

```bash
veripresence-import ../ImagesAttendance \
  --destination data/raw/enrollment \
  --minimum-images 2
```

The importer converts names such as `David11.jpg` into
`data/raw/enrollment/david/david_001.jpg`. Groups below the minimum image count are excluded.

### Option B: Generate privacy-safe demo data

```bash
veripresence-demo-data --destination data/raw/enrollment --seed 42
```

The synthetic data mirrors the old three-class imbalance and includes grayscale/color and
capture variation. It validates the software workflow; it is not a face-recognition
benchmark.

## Training

```bash
veripresence-train --config configs/default.yaml
```

Training performs:

1. Dataset validation and SHA-256 fingerprinting.
2. Face detection with 90-degree orientation retries and enrollment-crop fallback.
3. CLAHE normalization and seeded augmentation.
4. HOG, spatial LBP, and intensity feature extraction.
5. Stratified SVM hyperparameter tuning.
6. Holdout evaluation and rejection-threshold calibration.
7. Final refit and artifact export.

Outputs:

```text
artifacts/models/veripresence.joblib
artifacts/runs/<UTC-run-id>/
  config.json
  dataset_manifest.json
  metadata.json
  metrics.json
```

The migrated legacy data produced 0.909 holdout accuracy, 0.944 balanced accuracy, and
0.903 macro F1 on an 11-image validation split. See
[`docs/COMPARISON.md`](docs/COMPARISON.md) for context and limitations.

An end-to-end replay of the original `webcam.avi` completed successfully and recorded
deduplicated attendance events for Jobin, George, and David.

## Inference

### Webcam

```bash
veripresence-camera --config configs/default.yaml --source 0
```

Press `q` or Escape to stop.

### Video file and recording

```bash
veripresence-camera \
  --config configs/default.yaml \
  --source input.mp4 \
  --record runtime/annotated.mp4
```

### Headless video processing

```bash
veripresence-camera \
  --config configs/default.yaml \
  --source input.mp4 \
  --no-display
```

## REST API

Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive documentation is available at `http://localhost:8000/docs`.

Recognize faces:

```bash
curl -X POST "http://localhost:8000/v1/recognitions?source=front-desk" \
  -H "X-API-Key: $VERIPRESENCE_API_KEY" \
  -H "Content-Type: image/jpeg" \
  --data-binary "@sample.jpg"
```

List filtered events:

```bash
curl \
  -H "X-API-Key: $VERIPRESENCE_API_KEY" \
  "http://localhost:8000/v1/attendance?event_type=attendance&source=front-desk"
```

Get a daily summary:

```bash
curl \
  -H "X-API-Key: $VERIPRESENCE_API_KEY" \
  "http://localhost:8000/v1/attendance/summary?event_date=2026-06-09"
```

Export CSV:

```bash
curl \
  -H "X-API-Key: $VERIPRESENCE_API_KEY" \
  "http://localhost:8000/v1/attendance/export?event_date=2026-06-09" \
  --output attendance-2026-06-09.csv
```

Reload a newly trained artifact without restarting Uvicorn:

```bash
curl -X POST \
  -H "X-API-Key: $VERIPRESENCE_API_KEY" \
  "http://localhost:8000/v1/model/reload"
```

The API returns HTTP 503 for recognition until a model artifact has been trained.

Set `VERIPRESENCE_API_KEY` to protect all `/v1` routes. Model reload and event deletion
always require a configured key. Historical cleanup also requires an explicit confirmation:

```bash
curl -X DELETE \
  -H "X-API-Key: $VERIPRESENCE_API_KEY" \
  "http://localhost:8000/v1/attendance?before=2026-01-01&confirm=true"
```

Image requests larger than `api.max_upload_bytes` are rejected with HTTP 413.

## Unknown-Person Webhook

Set a webhook URL before running the camera process:

```bash
export VERIPRESENCE_WEBHOOK_URL="https://internal.example/events"
```

The webhook receives JSON only when a new unknown event passes the configured cooldown.
Unlike the legacy project, no browser or keyboard automation is used.

## Testing

```bash
pytest
ruff check .
```

Tests generate temporary synthetic enrollment images and do not require private data.

## Verified Results

The repository was verified on June 9, 2026 with Python 3.11:

- 6 automated tests passed.
- The package built and imported as `veripresence==1.0.0`.
- A 239-frame legacy video completed through headless inference.
- Known samples for David, George, and Jobin were accepted.
- The non-enrolled guide sample was rejected as `unknown`.
- The migrated 44-image dataset measured 0.909 holdout accuracy and 0.903 macro F1.

These measurements demonstrate the current pipeline on a small legacy dataset. They are
not a production certification or a substitute for a larger session-separated evaluation.

## Docker

Train or mount a trusted artifact first, then:

```bash
docker build -t veripresence .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/runtime:/app/runtime" \
  veripresence
```

## Security and Privacy

Face images are biometric data. Use VeriPresence only with informed consent and appropriate
access, retention, encryption, and deletion controls. The current system does not implement
liveness detection and must not be treated as a high-security authentication boundary.

Never load a Joblib model from an untrusted source.

## Documentation

- [Product requirements](docs/PRD.md)
- [Original vs new comparison](docs/COMPARISON.md)
- [Model card](docs/MODEL_CARD.md)
- [Security policy](SECURITY.md)

## License

MIT
