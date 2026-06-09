from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from veripresence.config import QualityConfig


@dataclass(frozen=True)
class FaceQuality:
    acceptable: bool
    score: float
    brightness: float
    contrast: float
    sharpness: float
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceptable": self.acceptable,
            "score": self.score,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "sharpness": self.sharpness,
            "issues": list(self.issues),
        }


class FaceQualityChecker:
    def __init__(self, config: QualityConfig) -> None:
        self.config = config

    def check(self, face: np.ndarray) -> FaceQuality:
        brightness = float(np.mean(face))
        contrast = float(np.std(face))
        sharpness = float(cv2.Laplacian(face, cv2.CV_64F).var())

        issues = []
        if brightness < self.config.min_brightness:
            issues.append("too_dark")
        elif brightness > self.config.max_brightness:
            issues.append("too_bright")
        if contrast < self.config.min_contrast:
            issues.append("low_contrast")
        if sharpness < self.config.min_sharpness:
            issues.append("blurry")

        brightness_score = 1.0 - min(abs(brightness - 127.5) / 127.5, 1.0)
        contrast_score = min(contrast / max(self.config.min_contrast, 1.0), 1.0)
        sharpness_score = min(sharpness / max(self.config.min_sharpness, 1.0), 1.0)
        score = float(np.mean([brightness_score, contrast_score, sharpness_score]))

        return FaceQuality(
            acceptable=not issues or not self.config.enabled,
            score=score,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            issues=tuple(issues),
        )
