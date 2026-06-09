# Product Requirements Document

## Product

**Name:** VeriPresence

**Description:** A reproducible computer-vision platform that recognizes enrolled people,
rejects uncertain identities, and records attendance events through camera and API workflows.

**Tagline:** Recognize presence. Reject uncertainty.

## Problem Statement

Small offices, labs, and classrooms often begin attendance automation as a single webcam
script. Those prototypes usually combine image loading, face recognition, UI rendering,
recording, notifications, and CSV mutation in one process. They are difficult to test,
unsafe to operate with uncertain predictions, and hard to deploy beyond one computer.

VeriPresence turns that prototype into an independently structured ML system with explicit
data contracts, reproducible training, open-set rejection, model artifacts, event storage,
and stable inference interfaces.

## Objective

Build an end-to-end face-attendance system that:

1. Learns one identity from multiple enrollment photos.
2. Rejects low-confidence or ambiguous predictions as unknown.
3. Records deduplicated recognition events.
4. Supports images, video files, webcams, and HTTP clients.
5. Produces reproducible metrics and versioned model artifacts.
6. Keeps private face images and runtime records out of source control.

## Target Users

- ML engineers modernizing a proof of concept.
- Small organizations running consent-based attendance workflows.
- Backend engineers integrating recognition into an internal application.
- Researchers evaluating lightweight CPU-based identity classification.

## ML Task Type

- Primary task: supervised multi-class image classification over enrolled identities.
- Detection task: frontal-face localization using OpenCV Haar features.
- Open-set behavior: confidence and probability-margin rejection for unknown people.
- Explainability: feature-family ablation for shape, texture, and intensity features.

This is not a liveness or anti-spoofing system.

## Input and Output Specification

### Training input

Folder-based enrollment data:

```text
data/raw/enrollment/
  person-a/
    image-001.jpg
    image-002.jpg
  person-b/
    image-001.jpg
```

Each identity must have at least three images by default. Supported formats are JPEG, PNG,
BMP, and WebP.

### Training output

- A versioned Joblib model artifact containing preprocessing metadata, the fitted pipeline,
  rejection thresholds, class names, and validation results.
- A timestamped run directory containing configuration, metrics, dataset fingerprint, and
  manifest files.

### Inference input

- Encoded image upload.
- BGR image array.
- Webcam index.
- Video file path.

### Inference output

For every detected face:

```json
{
  "identity": "person-a",
  "accepted": true,
  "confidence": 0.91,
  "margin": 0.72,
  "bbox": [120, 80, 160, 160]
}
```

Rejected predictions use the identity `unknown`.

## System Workflow

1. Import legacy images or generate privacy-safe demo data.
2. Validate identity folders and minimum image counts.
3. Detect the dominant face with orientation retries and safe fallback for enrollment crops.
4. Normalize faces with grayscale conversion, CLAHE, and fixed-size resizing.
5. Extract HOG shape, spatial LBP texture, and intensity features.
6. Tune a class-balanced linear SVM using stratified cross-validation.
7. Evaluate against a held-out stratified split.
8. Calibrate confidence and class-margin rejection thresholds.
9. Refit on all enrollment images and export the artifact.
10. Reject face crops that fail brightness, contrast, or sharpness checks.
11. Serve inference through FastAPI or the camera runner.
12. Deduplicate and persist attendance or unknown events in SQLite.
13. Produce filtered history, daily summaries, and CSV reports.
14. Optionally emit unknown-person events to a configured webhook.

## MVP Features

- Folder-based identity dataset loader and validation.
- Legacy filename migration utility.
- Deterministic preprocessing and augmentation.
- Multi-image identity classifier.
- Stratified validation and standard classification metrics.
- Explicit unknown rejection.
- Face-image quality checks with machine-readable rejection reasons.
- Joblib model export.
- Webcam and video inference.
- FastAPI recognition and attendance endpoints.
- SQLite event persistence with cooldown-based deduplication.
- Daily attendance summaries, filters, pagination, and CSV export.
- Source-aware cooldowns for multi-camera deployments.
- Optional API-key protection and bounded image uploads.
- Live model reload and confirmed retention cleanup.
- Dataset, training, and inference tests.

## Advanced Features

- Deep face embeddings using an audited ONNX model.
- Dedicated unknown-identity calibration and ROC analysis.
- Liveness and presentation-attack detection.
- Encrypted biometric templates and configurable retention policies.
- Multi-camera event streaming through a message broker.
- Role-based access control and audit logs.
- Human review queue for uncertain events.
- Monitoring for drift, camera health, and identity-level error rates.

## Success Metrics

### ML quality

- Macro F1 of at least 0.85 on a person-disjoint or session-disjoint validation set.
- Balanced accuracy of at least 0.85.
- Unknown false-accept rate below 5% on a dedicated unknown-person benchmark.
- Known-person rejection rate below 10% at the selected operating threshold.

### System quality

- P95 model inference below 100 ms per normalized face on a modern CPU.
- No duplicate attendance event within the configured cooldown.
- Reproducible artifact metadata and dataset fingerprint for every run.
- Automated dataset, training, and inference tests on every pull request.

## Constraints

- Face data is sensitive biometric information and requires consent, access control, and
  jurisdiction-appropriate retention policies.
- The bundled detector is optimized for frontal faces and can degrade under occlusion,
  extreme pose, or poor lighting.
- The legacy dataset is small and not sufficient for production certification.
- Confidence rejection reduces risk but does not make the system a security boundary.
- Camera display requires an environment with GUI-capable OpenCV.

## Tech Stack

- Python 3.10+
- OpenCV for face detection, normalization, HOG features, and video I/O
- NumPy for numerical transformations
- scikit-learn for scaling, SVM training, tuning, and metrics
- Joblib for model artifacts
- FastAPI and Uvicorn for HTTP inference
- SQLite for local event persistence
- PyYAML for configuration
- pytest and Ruff for quality gates
- Docker and GitHub Actions for delivery
