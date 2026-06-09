from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ClassifierFitResult:
    estimator: Pipeline
    best_parameters: dict[str, float]
    cross_validation_score: float | None


def fit_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    c_values: tuple[float, ...],
    cv_folds: int,
    seed: int,
) -> ClassifierFitResult:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )

    _, counts = np.unique(labels, return_counts=True)
    effective_folds = min(cv_folds, int(counts.min()))
    if effective_folds >= 2 and len(c_values) > 1:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid={"classifier__C": list(c_values)},
            scoring="f1_macro",
            cv=StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed),
            n_jobs=1,
            refit=True,
        )
        search.fit(features, labels)
        return ClassifierFitResult(
            estimator=search.best_estimator_,
            best_parameters={"C": float(search.best_params_["classifier__C"])},
            cross_validation_score=float(search.best_score_),
        )

    c_value = float(c_values[0])
    pipeline.set_params(classifier__C=c_value)
    pipeline.fit(features, labels)
    return ClassifierFitResult(
        estimator=pipeline,
        best_parameters={"C": c_value},
        cross_validation_score=None,
    )


def probability_summary(
    estimator: Pipeline, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = estimator.predict_proba(features)
    class_names = estimator.classes_
    sorted_indices = np.argsort(probabilities, axis=1)
    top_indices = sorted_indices[:, -1]
    second_indices = sorted_indices[:, -2]
    labels = class_names[top_indices]
    confidence = probabilities[np.arange(len(probabilities)), top_indices]
    margin = confidence - probabilities[np.arange(len(probabilities)), second_indices]
    return labels, confidence, margin


def calibrate_rejection_thresholds(
    predicted_labels: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
    true_labels: np.ndarray,
    confidence_bounds: tuple[float, float],
    margin_bounds: tuple[float, float],
) -> tuple[float, float]:
    correct = predicted_labels == true_labels
    confidence_source = confidence[correct] if np.any(correct) else confidence
    margin_source = margin[correct] if np.any(correct) else margin
    confidence_threshold = float(
        np.clip(np.percentile(confidence_source, 10), *confidence_bounds)
    )
    margin_threshold = float(np.clip(np.percentile(margin_source, 10), *margin_bounds))
    return confidence_threshold, margin_threshold
