# Original vs New Project Comparison

## Scope

The original 2022 project contained two standalone webcam scripts, a flat image directory,
a CSV attendance file, a text alert, and a recorded AVI. VeriPresence is a separate
production-style repository. The original files remain untouched.

## Architecture Differences

| Area | Original project | VeriPresence |
|---|---|---|
| Structure | Two monolithic scripts | Installable `src/` package split by data, models, training, evaluation, inference, storage, alerts, and API |
| Configuration | Hard-coded paths, thresholds, camera, and codec | Versioned YAML configuration with resolved project paths |
| State | Global variables and direct file mutation | Explicit services, model artifact, and repository interfaces |
| Attendance | Flat CSV, one record per name for the file lifetime | SQLite event log with timestamps, event types, sources, and cooldown deduplication |
| Alerts | Opens WhatsApp and types text through desktop automation | Optional webhook with timeout and failure logging |
| Deployment | Local interactive script only | CLI, FastAPI, Docker, and CI |
| Verification | No tests | Dataset, training, and inference tests |

The redesign separates training-time and runtime dependencies, allows independent API and
camera processes, and gives each component one clear responsibility.

## ML Pipeline Improvements

### Original

- Encoded every file independently with `face_recognition`.
- Treated filenames such as `Jobin1` and `Jobin2` as separate classes during matching.
- Removed digits only after selecting the nearest image.
- Used a single hard-coded distance threshold.
- Had no training split, tuning, metrics, augmentation, or exported model metadata.
- Assumed every image contained a detectable face and indexed the first result.

### New

- Groups all images under one identity before training.
- Handles portrait, landscape, grayscale, rotated, and already-cropped enrollment images.
- Applies CLAHE normalization and deterministic augmentation.
- Combines HOG shape, spatial LBP texture, and intensity features.
- Tunes a class-balanced linear SVM with stratified cross-validation.
- Calibrates both confidence and top-two-class margin thresholds.
- Exports the fitted pipeline, thresholds, classes, metrics, and dataset fingerprint.
- Reports feature-family ablation to show which signal families affect validation accuracy.

## Dataset Handling

The original dataset is private, nonstandard biometric enrollment data. There is no direct
official public replacement that preserves the same identities and consent context.

VeriPresence therefore provides two explicit paths:

1. `veripresence-import` migrates legacy filename labels into folder-based identity labels.
2. `veripresence-demo-data` generates seeded, privacy-safe face-like images with the legacy
   class imbalance and grayscale/color variation for functional testing.

Private images, model artifacts, and runtime databases are ignored by Git.

The legacy migration found:

| Identity | Images |
|---|---:|
| david | 11 |
| george | 10 |
| jobin | 23 |

The one-image `guide` group was excluded because it did not meet the minimum enrollment
count and appeared to be an accidental class in the old flat directory.

## Performance Improvements

The original project had no evaluation dataset or recorded metrics, so a defensible
old-versus-new accuracy delta cannot be calculated.

On a seeded 75/25 stratified split of the 44 migrated images, the new system measured:

| Metric | Result |
|---|---:|
| Holdout accuracy | 0.909 |
| Balanced accuracy | 0.944 |
| Macro F1 | 0.903 |
| Cross-validation macro F1 | 0.835 |
| Mean classifier time per normalized face | 0.59 ms |
| Selected SVM C | 0.1 |
| Confidence threshold | 0.55 |
| Margin threshold | 0.196 |

The excluded `guide` image was tested as an unknown person and rejected with confidence
0.448 and margin 0.021.

The original `webcam.avi` was also replayed through the headless camera pipeline. It
completed successfully and wrote deduplicated attendance events for all three enrolled
identities: Jobin, George, and David.

These results demonstrate that the pipeline works, not that it is deployment-certified.
The holdout contains only 11 images and may share capture conditions with training data.
A production assessment needs new sessions, more people, and a dedicated unknown set.

## Feature Enhancements

- Hyperparameter tuning with stratified cross-validation.
- Open-set rejection using confidence and ambiguity thresholds.
- Multi-face HTTP inference through FastAPI.
- SQLite attendance and unknown-event history.
- Image-quality rejection with explicit reasons and measurements.
- Source-aware cooldowns for installations with more than one camera.
- Filtered event history, daily summaries, pagination, and CSV reports.
- Optional API-key protection, upload limits, model reload, and retention cleanup.
- Optional webhook alerts without GUI automation.
- Video or webcam runner with annotation and optional recording.
- Reproducible model export and timestamped experiment tracking.
- Feature-family ablation report.
- Privacy-safe synthetic demo-data generator.
- Docker image and GitHub Actions workflow.

## Code Quality Improvements

- Typed dataclasses define configuration and inference contracts.
- Paths are resolved from the configuration file rather than the current shell.
- Errors for missing data, invalid images, missing models, and unavailable cameras are
  explicit.
- Private data and generated files are excluded from version control.
- JSON logs replace scattered print statements.
- Deterministic seeds, manifests, and fingerprints make runs traceable.
- Tests cover the required dataset loading, training export, and inference behavior.
- The original duplicate alert block, unused imports, ignored camera read status, and
  filename-label bug are removed.
