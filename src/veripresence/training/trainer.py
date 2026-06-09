from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from veripresence.config import AppConfig
from veripresence.data.dataset import EnrollmentDataset
from veripresence.data.preprocessing import FacePreprocessor
from veripresence.evaluation.metrics import classification_metrics, feature_family_ablation
from veripresence.models.artifact import ModelArtifact
from veripresence.models.classifier import (
    calibrate_rejection_thresholds,
    fit_classifier,
    probability_summary,
)
from veripresence.models.features import FaceFeatureExtractor
from veripresence.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    run_dir: Path
    metrics: dict[str, Any]
    metadata: dict[str, Any]


def train(config: AppConfig) -> TrainingResult:
    started = time.perf_counter()
    seed_everything(config.training.seed)
    rng = np.random.default_rng(config.training.seed)

    dataset = EnrollmentDataset(
        config.paths.enrollment_dir,
        minimum_samples_per_identity=config.training.minimum_samples_per_identity,
    )
    samples = dataset.discover()
    preprocessor = FacePreprocessor(config.preprocessing)
    extractor = FaceFeatureExtractor(config.preprocessing.face_size)

    faces: list[np.ndarray] = []
    labels: list[str] = []
    valid_samples = []
    rejected_samples: list[dict[str, str]] = []
    for sample in samples:
        try:
            image = dataset.read_image(sample)
            face = preprocessor.extract_primary_face(
                image, allow_full_frame_fallback=True
            ).image
            faces.append(face)
            labels.append(sample.identity)
            valid_samples.append(sample)
        except (ValueError, cv2.error) as error:
            LOGGER.warning("Skipping %s: %s", sample.path, error)
            rejected_samples.append({"path": str(sample.path), "reason": str(error)})

    _validate_processed_labels(labels, config.training.minimum_samples_per_identity)
    label_array = np.asarray(labels)
    train_indices, validation_indices = _stratified_indices(
        label_array,
        config.training.validation_fraction,
        config.training.seed,
    )

    base_training_faces = [faces[index] for index in train_indices]
    base_training_labels = label_array[train_indices]
    tuning_features = extractor.transform(base_training_faces)
    tuning_fit = fit_classifier(
        tuning_features,
        base_training_labels,
        c_values=config.training.c_values,
        cv_folds=config.training.cv_folds,
        seed=config.training.seed,
    )

    training_faces, training_labels = _augment_faces(
        base_training_faces,
        base_training_labels,
        preprocessor,
        rng,
        config.training.augmentation_copies,
    )
    training_features = extractor.transform(training_faces)
    validation_features = extractor.transform([faces[index] for index in validation_indices])
    validation_labels = label_array[validation_indices]

    validation_fit = fit_classifier(
        training_features,
        np.asarray(training_labels),
        c_values=(tuning_fit.best_parameters["C"],),
        cv_folds=1,
        seed=config.training.seed,
    )
    validation_metrics = classification_metrics(
        validation_fit.estimator, validation_features, validation_labels
    )
    predicted, confidence, margin = probability_summary(
        validation_fit.estimator, validation_features
    )
    confidence_threshold, margin_threshold = calibrate_rejection_thresholds(
        predicted,
        confidence,
        margin,
        validation_labels,
        config.training.confidence_threshold_bounds,
        config.training.margin_threshold_bounds,
    )
    validation_metrics["feature_family_ablation"] = feature_family_ablation(
        validation_fit.estimator,
        validation_features,
        validation_labels,
        extractor.family_slices(),
    )
    validation_metrics["confidence_threshold"] = confidence_threshold
    validation_metrics["margin_threshold"] = margin_threshold
    validation_metrics["best_parameters"] = tuning_fit.best_parameters
    validation_metrics["cross_validation_macro_f1"] = tuning_fit.cross_validation_score

    all_faces, all_labels = _augment_faces(
        faces,
        label_array,
        preprocessor,
        np.random.default_rng(config.training.seed + 1),
        config.training.augmentation_copies,
    )
    final_features = extractor.transform(all_faces)
    final_fit = fit_classifier(
        final_features,
        np.asarray(all_labels),
        c_values=(tuning_fit.best_parameters["C"],),
        cv_folds=1,
        seed=config.training.seed,
    )

    class_counts = dict(sorted(Counter(labels).items()))
    metadata = {
        "project": config.project.name,
        "project_version": config.project.version,
        "classes": [str(label) for label in final_fit.estimator.classes_],
        "class_counts": class_counts,
        "dataset_fingerprint": EnrollmentDataset.fingerprint(valid_samples),
        "feature_dimensions": extractor.layout.total_dimensions,
        "best_parameters": tuning_fit.best_parameters,
        "validation_metrics": validation_metrics,
        "rejected_samples": rejected_samples,
    }
    artifact = ModelArtifact.create(
        estimator=final_fit.estimator,
        face_size=config.preprocessing.face_size,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        metadata=metadata,
    )
    artifact.save(config.paths.model_path)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    duration = time.perf_counter() - started
    validation_metrics["training_duration_seconds"] = float(duration)
    _write_json(run_dir / "metrics.json", validation_metrics)
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "config.json", config.to_dict())
    _write_json(
        run_dir / "dataset_manifest.json",
        {
            "fingerprint": metadata["dataset_fingerprint"],
            "class_counts": class_counts,
            "files": [
                {"identity": sample.identity, "filename": sample.path.name}
                for sample in valid_samples
            ],
        },
    )
    return TrainingResult(
        model_path=config.paths.model_path,
        run_dir=run_dir,
        metrics=validation_metrics,
        metadata=metadata,
    )


def _stratified_indices(
    labels: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    class_count = len(np.unique(labels))
    validation_size = max(class_count, int(round(len(labels) * validation_fraction)))
    validation_size = min(validation_size, len(labels) - class_count)
    if validation_size < class_count:
        raise ValueError("Not enough data to create a stratified validation split.")
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        stratify=labels,
    )
    return train_indices, validation_indices


def _augment_faces(
    faces: list[np.ndarray],
    labels: np.ndarray,
    preprocessor: FacePreprocessor,
    rng: np.random.Generator,
    copies: int,
) -> tuple[list[np.ndarray], list[str]]:
    augmented_faces: list[np.ndarray] = []
    augmented_labels: list[str] = []
    for face, label in zip(faces, labels, strict=True):
        augmented_faces.append(face)
        augmented_labels.append(str(label))
        for _ in range(copies):
            augmented_faces.append(preprocessor.augment(face, rng))
            augmented_labels.append(str(label))
    return augmented_faces, augmented_labels


def _validate_processed_labels(labels: list[str], minimum: int) -> None:
    counts = Counter(labels)
    underfilled = {name: count for name, count in counts.items() if count < minimum}
    if underfilled:
        raise ValueError(f"Too few decodable images after preprocessing: {underfilled}")
    if len(counts) < 2:
        raise ValueError("At least two valid identities are required.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
