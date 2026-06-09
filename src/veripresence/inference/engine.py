from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from veripresence.config import PreprocessingConfig, QualityConfig
from veripresence.data.preprocessing import FacePreprocessor
from veripresence.inference.quality import FaceQuality, FaceQualityChecker
from veripresence.models.artifact import ModelArtifact
from veripresence.models.classifier import probability_summary
from veripresence.models.features import FaceFeatureExtractor


@dataclass(frozen=True)
class Recognition:
    identity: str
    accepted: bool
    confidence: float
    margin: float
    bbox: tuple[int, int, int, int]
    quality: FaceQuality
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "accepted": self.accepted,
            "confidence": self.confidence,
            "margin": self.margin,
            "bbox": list(self.bbox),
            "quality": self.quality.to_dict(),
            "reason": self.reason,
        }


class RecognitionEngine:
    def __init__(
        self,
        model_path: str | Path,
        preprocessing_config: PreprocessingConfig,
        quality_config: QualityConfig | None = None,
        unknown_label: str = "unknown",
    ) -> None:
        self.artifact = ModelArtifact.load(model_path)
        effective_config = PreprocessingConfig(
            **{
                **preprocessing_config.__dict__,
                "face_size": self.artifact.face_size,
            }
        )
        self.preprocessor = FacePreprocessor(effective_config)
        self.extractor = FaceFeatureExtractor(self.artifact.face_size)
        self.quality_checker = FaceQualityChecker(quality_config or QualityConfig())
        self.unknown_label = unknown_label

    def recognize_image(
        self, image: np.ndarray, allow_full_frame_fallback: bool = False
    ) -> list[Recognition]:
        crops = self.preprocessor.extract_faces(image)
        if not crops and allow_full_frame_fallback:
            crops = [
                self.preprocessor.extract_primary_face(
                    image, allow_full_frame_fallback=True
                )
            ]
        if not crops:
            return []

        features = self.extractor.transform([crop.image for crop in crops])
        labels, confidence, margin = probability_summary(
            self.artifact.estimator, features
        )
        recognitions = []
        for crop, label, score, score_margin in zip(
            crops, labels, confidence, margin, strict=True
        ):
            quality = self.quality_checker.check(crop.image)
            threshold_passed = (
                float(score) >= self.artifact.confidence_threshold
                and float(score_margin) >= self.artifact.margin_threshold
            )
            accepted = quality.acceptable and threshold_passed
            reason = None
            if not quality.acceptable:
                reason = "poor_quality"
            elif not threshold_passed:
                reason = "low_confidence"
            recognitions.append(
                Recognition(
                    identity=str(label) if accepted else self.unknown_label,
                    accepted=accepted,
                    confidence=float(score),
                    margin=float(score_margin),
                    bbox=crop.bbox,
                    quality=quality,
                    reason=reason,
                )
            )
        return recognitions

    def recognize_bytes(
        self, payload: bytes, allow_full_frame_fallback: bool = False
    ) -> list[Recognition]:
        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Uploaded content is not a decodable image.")
        return self.recognize_image(image, allow_full_frame_fallback)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.artifact.metadata,
            "schema_version": self.artifact.schema_version,
            "confidence_threshold": self.artifact.confidence_threshold,
            "margin_threshold": self.artifact.margin_threshold,
        }
