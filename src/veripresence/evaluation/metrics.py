from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from veripresence.models.classifier import probability_summary


def classification_metrics(
    estimator: Pipeline,
    features: np.ndarray,
    true_labels: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    predicted_labels, confidence, margin = probability_summary(estimator, features)
    elapsed = time.perf_counter() - started
    labels = [str(label) for label in estimator.classes_]
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro")),
        "mean_confidence": float(np.mean(confidence)),
        "mean_margin": float(np.mean(margin)),
        "mean_inference_ms_per_face": float(elapsed * 1000 / max(1, len(features))),
        "labels": labels,
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def feature_family_ablation(
    estimator: Pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    family_slices: dict[str, slice],
) -> dict[str, dict[str, float]]:
    baseline = float(accuracy_score(labels, estimator.predict(features)))
    report: dict[str, dict[str, float]] = {}
    for name, feature_slice in family_slices.items():
        ablated = features.copy()
        ablated[:, feature_slice] = 0.0
        score = float(accuracy_score(labels, estimator.predict(ablated)))
        report[name] = {
            "accuracy_without_family": score,
            "accuracy_drop": float(baseline - score),
        }
    return report
